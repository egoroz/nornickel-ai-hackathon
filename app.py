"""Streamlit-приложение классификации шлифов.

  streamlit run app.py --server.maxUploadSize 1024
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.classify import OreAnalyzer
from src.config import TALC_THRESHOLD
from src.pipeline import analyze_path
from src.report import mask_to_labelme_json, metrics_table, save_pdf

st.set_page_config(page_title="Классификация шлифов", layout="wide")


@st.cache_resource(show_spinner=False)
def get_analyzer() -> OreAnalyzer:
    return OreAnalyzer()


st.caption("Классификация руд по фотографиям полированных шлифов")

with st.sidebar:
    uploads = st.file_uploader(
        "Изображения (TIFF / PNG / JPEG / BMP)",
        type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True) or []

    st.header("Параметры")
    talc_thr = st.slider("Порог оталькованности, % зон талька", 1, 30,
                         int(TALC_THRESHOLD * 100), key="p_talc") / 100
    fine_thr = st.slider("Порог «тонкого» зерна", 0.2, 0.8, 0.5, 0.05, key="p_fine")
    run = st.button("Анализировать", type="primary", use_container_width=True)

if run:
    with st.spinner("Загрузка моделей…"):
        analyzer = get_analyzer()
    analyzer.talc_threshold = talc_thr
    analyzer.fine_grain_thr = fine_thr

    paths = []
    for up in uploads:
        tmp = Path(tempfile.mkdtemp()) / up.name
        tmp.write_bytes(up.getbuffer())
        paths.append(tmp)
    if not paths:
        st.warning("Загрузите хотя бы один файл")
        st.stop()

    results = {}
    prog = st.progress(0.0, "Обработка…")
    for i, p in enumerate(paths):
        prog.progress(i / len(paths), f"{p.name} ({i + 1}/{len(paths)})")
        try:
            results[p.name] = analyze_path(Path(p), analyzer)
        except Exception as e:
            st.error(f"{p.name}: {e}")
    prog.empty()
    st.session_state["batch"] = results

if "batch" in st.session_state and st.session_state["batch"]:
    results = st.session_state["batch"]

    if len(results) > 1:
        st.subheader(f"Сводка: {len(results)} изображений")
        rows = [dict(файл=n, класс=r.ore_class,
                     **{"зоны талька, %": round(r.talc_zone_share * 100, 1),
                        "тальк (вкрапл.), %": round(r.talc_share * 100, 1),
                        "тонкие, %": round(r.fine_dominance * 100),
                        "сульфиды, %": round(r.sulfide_share * 100, 1)})
                for n, r in results.items()]
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button("Сводка (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "batch_summary.csv", "text/csv")

    name = (list(results)[0] if len(results) == 1 else
            st.selectbox("Открыть результат", list(results), key="open_res"))
    res = results[name]

    st.subheader(f"{name} — {res.conclusion}")

    c1, c2 = st.columns([3, 2])
    with c1:
        if res.layers is not None and res.source_bgr is not None:
            lc = st.columns(5)
            show = dict(
                ordinary=lc[0].checkbox("Обычные", True, key="l_ord"),
                fine=lc[1].checkbox("Тонкие", True, key="l_fine"),
                talc=lc[2].checkbox("Тальк", True, key="l_talc"),
                zones=lc[3].checkbox("Зоны талька", True, key="l_zone"),
                magnetite=lc[4].checkbox("Магнетит", False, key="l_mag"),
            )
            shown = OreAnalyzer.compose_overlay(res.source_bgr, res.layers, show)
        else:
            shown = res.overlay
        st.caption("зелёный — обычные срастания, красный — тонкие, синий — тальк, "
                   "контуром — зоны; зум колесом мыши")
        disp = shown
        if max(disp.shape[:2]) > 1400:  # даунскейл только для отображения
            s = 1400 / max(disp.shape[:2])
            disp = cv2.resize(disp, (int(disp.shape[1] * s), int(disp.shape[0] * s)),
                              interpolation=cv2.INTER_AREA)
        import plotly.express as px
        fig = px.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), binary_string=True)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=620,
                          xaxis_visible=False, yaxis_visible=False, dragmode="pan")
        st.plotly_chart(fig, use_container_width=True,
                        config=dict(scrollZoom=True), key="main_overlay")

        if getattr(res, "tile_map", None) is not None:
            with st.expander("Карта классов по тайлам"):
                st.image(cv2.cvtColor(res.tile_map, cv2.COLOR_BGR2RGB),
                         caption="рамки: зелёная — рядовая, красная — труднообогатимая, "
                                 "синяя — оталькованная, серая — пусто",
                         use_container_width=True)
        if res.talc_proba is not None:
            with st.expander("Карта уверенности талька"):
                figp = px.imshow(res.talc_proba, color_continuous_scale="Blues",
                                 zmin=0, zmax=1)
                figp.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=420,
                                   xaxis_visible=False, yaxis_visible=False)
                st.plotly_chart(figp, use_container_width=True, key="talc_proba")

    with c2:
        tbl = metrics_table(res)
        st.dataframe(tbl, hide_index=True, use_container_width=True)

        st.download_button("Метрики (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                           f"{Path(name).stem}_metrics.csv", "text/csv",
                           use_container_width=True)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            save_pdf(res, Path(f.name), source=name)
            pdf_bytes = Path(f.name).read_bytes()
        st.download_button("Отчёт (PDF)", pdf_bytes,
                           f"{Path(name).stem}_report.pdf", "application/pdf",
                           use_container_width=True)
        ok, buf = cv2.imencode(".jpg", res.overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        st.download_button("Оверлей (JPG)", buf.tobytes(),
                           f"{Path(name).stem}_overlay.jpg", "image/jpeg",
                           use_container_width=True)
        if getattr(res, "color_mask", None) is not None:
            ok, mbuf = cv2.imencode(".png", res.color_mask)
            st.download_button("Маска (PNG)", mbuf.tobytes(),
                               f"{Path(name).stem}_mask.png", "image/png",
                               use_container_width=True)
        if getattr(res, "talc_zones_mask", None) is not None:
            h, w = res.talc_zones_mask.shape
            js = mask_to_labelme_json(res.talc_zones_mask, name, h, w)
            st.download_button("Разметка зон (labelme)", js.encode("utf-8"),
                               f"{Path(name).stem}.json", "application/json",
                               use_container_width=True,
                               help="Файл открывается в labelme рядом с фото: "
                                    "полигоны можно править")

        st.caption(f"модель талька: {res.params.get('talc_model')}, "
                   f"порог зон: {talc_thr}")
else:
    st.info("Загрузите изображения слева и нажмите «Анализировать».")
