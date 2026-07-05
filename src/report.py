"""Отчёты: таблица метрик, CSV, PDF, лог параметров запуска."""
from __future__ import annotations

import json
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .classify import AnalysisResult

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def metrics_table(res: AnalysisResult) -> pd.DataFrame:
    rows = [
        ("Класс руды", res.ore_class),
        ("Доля талька (вкрапления), %", f"{res.talc_share*100:.1f}"),
        ("Зоны скопления талька, % площади", f"{res.talc_zone_share*100:.1f}"),
        ("Доля сульфидов (всего), %", f"{res.sulfide_share*100:.1f}"),
        ("Серые/средние фазы (магнетит и др.), %", f"{res.gray_share*100:.1f}"),
        ("Нерудные фазы, %", f"{res.nonore_share*100:.1f}"),
        ("Обычные срастания, % площади", f"{res.ordinary_share*100:.1f}"),
        ("Тонкие срастания, % площади", f"{res.fine_share*100:.1f}"),
        ("Тонкие среди всех срастаний, %", f"{res.fine_dominance*100:.1f}"),
        ("Сульфидная мелочь вне сростков, %", f"{res.fines_share*100:.2f}"),
        ("Среднее замещение зерна, %", f"{res.mean_replacement*100:.0f}"),
        ("Медианная площадь зерна, px", f"{res.median_grain_px:.0f}"),
        ("Вкрапленников найдено", str(res.n_grains)),
    ]
    return pd.DataFrame(rows, columns=["Метрика", "Значение"])


def save_csv(res: AnalysisResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_table(res).to_csv(path, index=False)


def save_json(res: AnalysisResult, path: Path, source: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(
        source=source, ts=dt.datetime.now().isoformat(timespec="seconds"),
        ore_class=res.ore_class, talc_share=res.talc_share,
        sulfide_share=res.sulfide_share, ordinary_share=res.ordinary_share,
        fine_share=res.fine_share, fine_dominance=res.fine_dominance,
        n_grains=res.n_grains, conclusion=res.conclusion, params=res.params,
    ), ensure_ascii=False, indent=2))


def _register_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            pdfmetrics.registerFont(TTFont("DejaVu", p))
            return "DejaVu"
    return "Helvetica"


def save_pdf(res: AnalysisResult, path: Path, source: str = "",
             overlay_max_px: int = 1400) -> None:
    """Одностраничный отчёт: оверлей + таблица + заключение."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as pdfcanvas

    path.parent.mkdir(parents=True, exist_ok=True)
    font = _register_font()
    c = pdfcanvas.Canvas(str(path), pagesize=A4)
    W, H = A4

    c.setFont(font, 15)
    c.drawString(18 * mm, H - 18 * mm, "Отчёт: классификация руды по шлифу")
    c.setFont(font, 9)
    c.drawString(18 * mm, H - 24 * mm,
                 f"{source}   ·   {dt.datetime.now():%Y-%m-%d %H:%M}")

    y = H - 30 * mm
    if res.overlay is not None:
        ov = res.overlay
        s = overlay_max_px / max(ov.shape[:2])
        if s < 1:
            ov = cv2.resize(ov, (int(ov.shape[1] * s), int(ov.shape[0] * s)))
        rgb = cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)
        img_w = W - 36 * mm
        img_h = img_w * rgb.shape[0] / rgb.shape[1]
        if img_h > 105 * mm:  # иначе таблица и заключение уйдут за низ страницы
            img_h = 105 * mm
            img_w = img_h * rgb.shape[1] / rgb.shape[0]
        from PIL import Image
        c.drawImage(ImageReader(Image.fromarray(rgb)),
                    18 * mm + (W - 36 * mm - img_w) / 2, y - img_h,
                    img_w, img_h)
        y -= img_h + 6 * mm
        c.setFont(font, 8)
        c.drawString(18 * mm, y, "зелёный — обычные срастания · красный — тонкие · синий — тальк")
        y -= 8 * mm

    c.setFont(font, 10)
    for _, (k, v) in metrics_table(res).iterrows():
        c.drawString(18 * mm, y, f"{k}:")
        c.drawRightString(W - 18 * mm, y, str(v))
        y -= 6 * mm

    y -= 4 * mm
    c.setFont(font, 11)
    for line in _wrap(c, res.conclusion, font, 11, W - 36 * mm):
        c.drawString(18 * mm, y, line)
        y -= 6 * mm
    c.save()


def _wrap(c, text: str, font: str, size: int, max_w: float) -> list[str]:
    """Перенос по фактической ширине строки, а не по числу символов."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if c.stringWidth(cand, font, size) > max_w and cur:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def mask_to_labelme_json(mask, image_name: str, h: int, w: int,
                         label: str = "t", eps: float = 3.0,
                         min_area: int = 800) -> str:
    """Маска -> labelme-JSON с полигонами (для доразметки)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    shapes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 3:
            continue
        shapes.append(dict(label=label,
                           points=[[float(x), float(y)] for x, y in approx[:, 0, :]],
                           group_id=None, shape_type="polygon", flags={}))
    return json.dumps(dict(version="5.4.1", flags={}, shapes=shapes,
                           imagePath=image_name, imageData=None,
                           imageHeight=h, imageWidth=w), ensure_ascii=False)
