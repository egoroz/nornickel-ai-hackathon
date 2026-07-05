"""Кнопочная разметка изображений. Ответы сразу пишутся в CSV.

  streamlit run label_app.py -- --manifest tile_labeling/manifest.csv \
      --labels "рядовая,труднообогатимая,оталькованная,пусто/не уверен" \
      --numeric "доля талька, %" --out tile_labeling/labels.csv
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--numeric", default=None,
                   help="имя необязательного числового поля (колонка value)")
    return p.parse_args(sys.argv[1:])


args = parse_args()
LABELS = [s.strip() for s in args.labels.split(",")]
OUT = Path(args.out)
st.set_page_config(page_title="Разметка", layout="wide")

mpath = Path(args.manifest)
if not mpath.exists():
    st.error(f"Манифест не найден: {mpath}\n\n"
             "Сначала собери датасет: .venv/bin/python tools/build_dataset.py")
    st.stop()

manifest = pd.read_csv(mpath)
manifest["id"] = manifest["id"].astype(str)
manifest["image_path"] = manifest["image_path"].map(
    lambda p: p if Path(p).is_absolute() else str((mpath.parent / p).resolve()))

labels_df = (pd.read_csv(OUT, dtype={"id": str})
             if OUT.exists() else
             pd.DataFrame(columns=["id", "label", "value", "ts"]))
answers = {r.id: r for r in labels_df.itertuples()}


def save(item_id: str, label: str, value):
    global labels_df
    labels_df = labels_df[labels_df.id != item_id]
    labels_df = pd.concat([labels_df, pd.DataFrame([dict(
        id=item_id, label=label, value=value,
        ts=dt.datetime.now().isoformat(timespec="seconds"))])])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(OUT, index=False)


n_done = manifest.id.isin(labels_df.id).sum()
st.sidebar.markdown(f"### {n_done} / {len(manifest)} размечено")
st.sidebar.progress(n_done / max(len(manifest), 1))

# блок создаётся до кнопок: st.rerun() обрывает прогон, и виджеты ниже
# по коду теряли бы состояние
st.sidebar.markdown("### Отображение")
auto_c = st.sidebar.checkbox("автоконтраст", value=True, key="disp_autoc")
gamma = st.sidebar.slider("гамма (выше = светлее)", 0.3, 2.5, 1.0, 0.1,
                          key="disp_gamma")

def fmt(i):
    r = manifest.iloc[i]
    a = answers.get(r.id)
    mark = f"✓ {a.label}" if a is not None else "○"
    return f"{r.id}  [{mark}]"

if "pos" not in st.session_state:
    undone = manifest.index[~manifest.id.isin(labels_df.id)]
    st.session_state.pos = int(undone[0]) if len(undone) else 0

pos = st.sidebar.selectbox("Элемент", options=range(len(manifest)),
                           format_func=fmt, index=st.session_state.pos)
if pos != st.session_state.pos:
    st.session_state.pos = pos

c1, c2 = st.sidebar.columns(2)
if c1.button("← пред."):
    st.session_state.pos = max(0, st.session_state.pos - 1)
    st.rerun()
if c2.button("след. →"):
    st.session_state.pos = min(len(manifest) - 1, st.session_state.pos + 1)
    st.rerun()
if st.sidebar.button("⏭ к первому неразмеченному"):
    undone = manifest.index[~manifest.id.isin(labels_df.id)]
    if len(undone):
        st.session_state.pos = int(undone[0])
        st.rerun()

row = manifest.iloc[st.session_state.pos]
prev = answers.get(row.id)

value = None
if args.numeric:
    default = float(prev.value) if (prev is not None and pd.notna(prev.value)) else 0.0
    value = st.number_input(args.numeric, min_value=0.0, max_value=100.0,
                            value=default, step=1.0)

cols = st.columns(len(LABELS))
for c, lbl in zip(cols, LABELS):
    is_current = prev is not None and prev.label == lbl
    if c.button(("✓ " if is_current else "") + lbl,
                use_container_width=True,
                type="primary" if is_current else "secondary"):
        save(row.id, lbl, value)
        done_ids = set(pd.read_csv(OUT, dtype={"id": str}).id)
        undone = manifest.index[~manifest.id.isin(done_ids)]
        st.session_state.pos = int(undone[0]) if len(undone) \
            else min(st.session_state.pos + 1, len(manifest) - 1)
        st.rerun()

if prev is not None:
    st.caption(f"текущий ответ: **{prev.label}**"
               + (f", {args.numeric} = {prev.value}" if args.numeric and pd.notna(prev.value) else ""))

import numpy as np
import plotly.express as px
from PIL import Image

img = Image.open(row.image_path).convert("RGB")
if auto_c or gamma != 1.0:
    a = np.asarray(img).astype(np.float32)
    if auto_c:
        lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
        a = np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255)
    if gamma != 1.0:
        a = (a / 255) ** (1 / gamma) * 255
    img = Image.fromarray(a.astype(np.uint8))
fig = px.imshow(img, binary_string=True)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=720,
                  xaxis_visible=False, yaxis_visible=False, dragmode="pan")
st.plotly_chart(fig, use_container_width=True,
                config=dict(scrollZoom=True, displayModeBar=True))

extra = {k: row[k] for k in manifest.columns if k not in ("id", "image_path")}
if extra:
    st.caption(" · ".join(f"{k}: {v}" for k, v in extra.items()))
