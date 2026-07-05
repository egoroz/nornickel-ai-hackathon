"""Анализ изображения шлифа: фазы -> срастания -> тальк -> сорт.
Правило: зоны талька > 10% -> оталькованная, иначе по преобладающему
типу срастаний.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import (ARTIFACTS, CLS_ORDINARY, CLS_REFRACTORY, CLS_TALC,
                     COLOR_FINE, COLOR_ORDINARY, COLOR_TALC, TALC_THRESHOLD)


@dataclass
class AnalysisResult:
    ore_class: str
    talc_share: float           # тальк-вкрапления / площадь
    sulfide_share: float
    ordinary_share: float       # площадь обычных срастаний / площадь
    fine_share: float           # то же для тонких
    fine_dominance: float       # доля тонких среди срастаний
    n_grains: int
    overlay: np.ndarray | None
    conclusion: str
    talc_zone_share: float = 0.0
    gray_share: float = 0.0       # магнетит и пр. серые фазы
    nonore_share: float = 0.0
    fines_share: float = 0.0      # сульфидная мелочь вне сростков
    median_grain_px: float = 0.0
    mean_replacement: float = 0.0
    tile_map: np.ndarray | None = None
    talc_proba: np.ndarray | None = None
    phase_cat: np.ndarray | None = None    # 0 фон, 1 серое, 2 сульфид, 3 тальк (даунскейл x4)
    color_mask: np.ndarray | None = None
    talc_zones_mask: np.ndarray | None = None
    layers: dict | None = None             # маски слоёв для интерактивного оверлея
    source_bgr: np.ndarray | None = None
    params: dict = field(default_factory=dict)


class OreAnalyzer:
    """Загружает модели один раз, анализирует изображения/тайлы."""

    def __init__(self, artifacts: Path = ARTIFACTS,
                 talc_threshold: float = TALC_THRESHOLD,
                 fine_grain_thr: float = 0.5):
        from .intergrowth import load_model as load_ig
        from .talc import load_model as load_talc, load_unet, UNET_PATH_DEFAULT

        self.talc_threshold = talc_threshold
        self.fine_grain_thr = fine_grain_thr
        self.ig_model = load_ig(artifacts / "intergrowth" / "model.joblib")
        self.talc_model = load_talc(artifacts / "talc" / "model.joblib")
        self.talc_unet = load_unet(artifacts / "talc" / "unet.pt")

    def analyze(self, bgr: np.ndarray, centers: np.ndarray | None = None,
                make_overlay: bool = True) -> AnalysisResult:
        from .intergrowth import (analyze_image, group_regions,
                                  image_fine_share, predict_grains,
                                  smooth_grain_proba)
        from .talc import predict_mask, predict_proba_map, predict_proba_map_unet

        res = analyze_image(bgr, centers=centers)
        sulf = res["sulfide_mask"]
        grains = res["grains"]
        env = res["envelopes"]
        from .sulfides import label_sulfide_clusters
        phase = res["phase"]
        sulf_ids = np.flatnonzero(label_sulfide_clusters(res["centers"]))
        gray_px = (~np.isin(phase, sulf_ids)) & (phase != 0)
        gray_share = float(gray_px.mean())

        proba_ig = predict_grains(self.ig_model, grains)
        # вердикт по области, а не по осколкам: близкие зёрна голосуют вместе
        proba_ig = smooth_grain_proba(grains, proba_ig,
                                      group_regions(env, grains))
        shares = image_fine_share(grains, proba_ig, self.fine_grain_thr)

        if self.talc_unet is not None:
            proba = predict_proba_map_unet(bgr, self.talc_unet, scale=4)
        else:
            proba = predict_proba_map(bgr, self.talc_model)
        # тальк ищется везде, кроме самих сульфидных пикселей
        exclude = sulf.astype(np.uint8)
        talc_mask, talc_share, talc_zones = predict_mask(
            bgr, self.talc_model, exclude_mask=exclude, proba=proba)

        img_area = float(bgr.shape[0] * bgr.shape[1])
        ordinary_share = shares["area_ordinary"] / img_area
        fine_share_img = shares["area_fine"] / img_area
        fine_dom = shares["fine_share"]

        zone_share = float(talc_zones.mean())
        ore_class = self._rule(zone_share, fine_dom)
        conclusion = self._conclusion(ore_class, zone_share, fine_dom)

        overlay = None
        color_mask = None
        phase_cat = None
        layers = None
        if make_overlay:
            overlay = self.make_overlay(bgr, sulf, env, grains, proba_ig,
                                        talc_mask, talc_zones=talc_zones,
                                        thr=self.fine_grain_thr)
            color_mask = self.make_overlay(np.zeros_like(bgr), sulf, env, grains,
                                           proba_ig, talc_mask,
                                           talc_zones=talc_zones, alpha=1.0,
                                           thr=self.fine_grain_thr)
            cat = np.zeros(bgr.shape[:2], np.uint8)
            cat[(~np.isin(phase, sulf_ids)) & (phase != 0)] = 1
            cat[sulf > 0] = 2
            cat[talc_zones > 0] = 3
            phase_cat = cat[::4, ::4].copy()

            fine_ids = {g.label_id for g, p in zip(grains, proba_ig)
                        if p >= self.fine_grain_thr}
            ord_ids = {g.label_id for g in grains} - fine_ids
            layers = dict(
                ordinary=np.isin(env, list(ord_ids)) & (sulf > 0),
                fine=np.isin(env, list(fine_ids)) & (sulf > 0),
                talc=talc_mask > 0,
                zones=talc_zones > 0,
                magnetite=(cat == 1),
            )
            # слои и фон храним в экранном размере: интерактив лёгкий,
            # полноразмерными остаются overlay/color_mask для экспорта
            s = 1400 / max(bgr.shape[:2])
            if s < 1:
                dw, dh = int(bgr.shape[1] * s), int(bgr.shape[0] * s)
                bgr_disp = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
                layers = {k: cv2.resize(v.astype(np.uint8), (dw, dh),
                                        interpolation=cv2.INTER_NEAREST) > 0
                          for k, v in layers.items()}
            else:
                bgr_disp = bgr

        return AnalysisResult(
            ore_class=ore_class, talc_share=talc_share,
            layers=layers, source_bgr=bgr_disp if make_overlay else None,
            sulfide_share=float(sulf.mean()),
            ordinary_share=ordinary_share, fine_share=fine_share_img,
            fine_dominance=fine_dom, n_grains=len(grains),
            gray_share=gray_share, nonore_share=float(1.0 - sulf.mean() - gray_share),
            fines_share=float(((sulf > 0) & (env == 0)).mean()),
            median_grain_px=float(np.median([g.area for g in grains])) if grains else 0.0,
            mean_replacement=(float(np.average([1 - g.features[0] for g in grains],
                                               weights=[g.area for g in grains]))
                              if grains else 0.0),
            overlay=overlay, conclusion=conclusion, talc_proba=proba,
            talc_zone_share=zone_share, phase_cat=phase_cat,
            color_mask=color_mask, talc_zones_mask=talc_zones,
            params=dict(talc_threshold=self.talc_threshold,
                        fine_grain_thr=self.fine_grain_thr,
                        talc_model="unet" if self.talc_unet is not None else "lgbm"),
        )

    def _rule(self, talc_zone_share: float, fine_dom: float) -> str:
        if talc_zone_share > self.talc_threshold:
            return CLS_TALC
        return CLS_REFRACTORY if fine_dom > 0.5 else CLS_ORDINARY

    def _conclusion(self, ore_class: str, talc_zones: float, fine_dom: float) -> str:
        kind = "тонкие" if fine_dom > 0.5 else "обычные"
        return (f"Руда классифицирована как {ore_class}: "
                f"зоны оталькования — {talc_zones*100:.0f}% площади, "
                f"преобладают {kind} срастания "
                f"(тонких — {fine_dom*100:.0f}% рудной массы).")

    @staticmethod
    def compose_overlay(bgr, layers, show, alpha: float = 0.28) -> np.ndarray:
        """Оверлей из выбранных слоёв (для интерактивного включения в UI)."""
        color = np.zeros_like(bgr)
        if show.get("magnetite") and layers.get("magnetite") is not None:
            color[layers["magnetite"]] = (160, 160, 160)
        if show.get("zones"):
            color[layers["zones"]] = (170, 140, 70)
        if show.get("ordinary"):
            color[layers["ordinary"]] = COLOR_ORDINARY
        if show.get("fine"):
            color[layers["fine"]] = COLOR_FINE
        if show.get("talc"):
            color[layers["talc"]] = COLOR_TALC
        colored = color.sum(axis=2) > 0
        out = bgr.copy()
        out[colored] = (alpha * color[colored]
                        + (1 - alpha) * bgr[colored]).astype(np.uint8)
        if show.get("zones") and layers["zones"].any():
            cont, _ = cv2.findContours(layers["zones"].astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, (255, 200, 80), 2)
        return out

    @staticmethod
    def make_overlay(bgr, sulf, envelopes, grains, proba_ig, talc_mask,
                     talc_zones=None, alpha: float = 0.28,
                     thr: float = 0.5) -> np.ndarray:
        """Полупрозрачные заливки + контуры зон: фактура остаётся видна."""
        color = np.zeros_like(bgr)
        if talc_zones is not None:
            color[talc_zones > 0] = (170, 140, 70)
        fine_ids = {g.label_id for g, p in zip(grains, proba_ig) if p >= thr}
        ord_ids = {g.label_id for g in grains} - fine_ids
        fine_px = np.isin(envelopes, list(fine_ids)) & (sulf > 0)
        ord_px = np.isin(envelopes, list(ord_ids)) & (sulf > 0)
        color[ord_px] = COLOR_ORDINARY
        color[fine_px] = COLOR_FINE
        color[talc_mask > 0] = COLOR_TALC
        colored = (color.sum(axis=2) > 0)
        out = bgr.copy()
        out[colored] = (alpha * color[colored] + (1 - alpha) * bgr[colored]).astype(np.uint8)
        if talc_zones is not None and talc_zones.any():
            cont, _ = cv2.findContours(talc_zones.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cont, -1, (255, 200, 80), 2)
        return out
