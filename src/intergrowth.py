"""Срастания «обычные» vs «тонкие» на уровне вкрапленников.
Оболочка зерна = морфологическое замыкание маски сульфидов + заливка дыр;
признаки масштабно-инвариантны; обучение на слабых метках (класс фото).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

FEATURE_NAMES = [
    "sulf_fill",      # доля сульфида в оболочке зерна — главный маркер замещения
    "gray_fill",      # доля серой фазы (магнетит) в оболочке
    "dark_fill",      # доля тёмной фазы в оболочке
    "sulf_solidity",  # площадь сульфида / выпуклая оболочка сульфида
    "frag_density",   # log10(число осколков сульфида на единицу площади зерна)
    "thick_rel",      # средняя полутолщина сульфида / sqrt(площади зерна)
    "extent",         # площадь зерна / bbox
    "log_area_rel",   # log10(площадь зерна / площадь изображения)
    "core_fill",      # доля ярчайшего кластера («ядра») в оболочке
    "core_frag",      # log10(фрагментов ядра на площадь зерна)
]
CLOSE_PX = 31
MIN_GRAIN_AREA = 1500


@dataclass
class Grain:
    label_id: int
    area: int
    bbox: tuple[int, int, int, int]
    features: np.ndarray


def grain_envelopes(sulfide_mask: np.ndarray, close_px: int = CLOSE_PX,
                    min_area: int = MIN_GRAIN_AREA) -> np.ndarray:
    """Карта меток вкрапленников (0 = фон)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    closed = cv2.morphologyEx(sulfide_mask, cv2.MORPH_CLOSE, k)
    filled = ndi.binary_fill_holes(closed).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < min_area)
    lab[np.isin(lab, small)] = 0
    return lab


def grain_features(bgr: np.ndarray, sulfide_mask: np.ndarray,
                   phase_labels: np.ndarray, sulf_cluster_ids: np.ndarray,
                   envelopes: np.ndarray | None = None) -> list[Grain]:
    """Признаки каждого вкрапленника."""
    if envelopes is None:
        envelopes = grain_envelopes(sulfide_mask)
    img_area = float(sulfide_mask.size)
    dark_id = 0  # кластеры отсортированы по яркости
    grains = []
    for lid in np.unique(envelopes):
        if lid == 0:
            continue
        ys, xs = np.where(envelopes == lid)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        env = (envelopes[y0:y1, x0:x1] == lid)
        area = int(env.sum())
        sulf = (sulfide_mask[y0:y1, x0:x1] > 0) & env
        ph = phase_labels[y0:y1, x0:x1]
        sulf_area = float(sulf.sum())
        sulf_fill = sulf_area / area
        gray_fill = float((~np.isin(ph, sulf_cluster_ids) & (ph != dark_id) & env).sum()) / area
        dark_fill = float(((ph == dark_id) & env).sum()) / area

        core = (ph == int(phase_labels.max())) & env
        core_area = float(core.sum())
        core_fill = core_area / area
        if core_area > 0:
            n_cf, _ = cv2.connectedComponents(core.astype(np.uint8), connectivity=8)
            core_frag = np.log10(max(n_cf - 1, 1) / area * 1e4 + 1e-6)
        else:
            core_frag = -6.0

        if sulf_area > 0:
            pts = cv2.findNonZero(sulf.astype(np.uint8))
            hull_area = float(cv2.contourArea(cv2.convexHull(pts))) + 1.0
            solidity = sulf_area / hull_area
            n_frag, _ = cv2.connectedComponents(sulf.astype(np.uint8), connectivity=8)
            frag_density = np.log10(max(n_frag - 1, 1) / area * 1e4 + 1e-6)
            dist = cv2.distanceTransform(sulf.astype(np.uint8), cv2.DIST_L2, 3)
            thick_rel = float(dist[sulf].mean()) / np.sqrt(area)
        else:
            solidity, frag_density, thick_rel = 0.0, 0.0, 0.0

        feats = np.array([
            sulf_fill, gray_fill, dark_fill, solidity, frag_density,
            thick_rel, area / float((y1 - y0) * (x1 - x0)),
            np.log10(area / img_area),
            core_fill, core_frag,
        ], dtype=np.float32)
        grains.append(Grain(int(lid), area, (int(x0), int(y0),
                                             int(x1 - x0), int(y1 - y0)), feats))
    return grains


def analyze_image(bgr: np.ndarray, centers: np.ndarray | None = None):
    """Сульфиды + фазовая карта + вкрапленники одного изображения."""
    from .sulfides import (assign_clusters, fit_phase_model,
                           label_sulfide_clusters, segment_sulfides)
    if centers is None:
        centers = fit_phase_model(bgr)
    mask, _ = segment_sulfides(bgr, centers=centers)
    phase = assign_clusters(bgr, centers)
    sulf_ids = np.flatnonzero(label_sulfide_clusters(centers))
    envelopes = grain_envelopes(mask)
    grains = grain_features(bgr, mask, phase, sulf_ids, envelopes)
    return dict(sulfide_mask=mask, phase=phase, centers=centers,
                envelopes=envelopes, grains=grains)


def build_grain_dataset(index: pd.DataFrame, max_per_image: int = 40,
                        seed: int = 0) -> pd.DataFrame:
    from .config import CLS_ORDINARY, CLS_REFRACTORY

    rng = np.random.RandomState(seed)
    sub = index[index.cls.isin([CLS_ORDINARY, CLS_REFRACTORY])]
    if "label_conflict" in sub.columns:
        n0 = len(sub)
        sub = sub[~sub.label_conflict]
        print(f"исключено конфликтных дублей: {n0 - len(sub)}")
    rows = []
    for _, r in sub.iterrows():
        img = cv2.imread(r.path)
        if img is None:
            continue
        res = analyze_image(img)
        grains = res["grains"]
        if len(grains) > max_per_image:
            grains = [grains[i] for i in
                      rng.choice(len(grains), max_per_image, replace=False)]
        for g in grains:
            rows.append(dict(
                path=r.path, file=r.file, part=r.part, cls=r.cls,
                sample_id=r.sample_id, group=r.get("group", r.sample_id),
                grain_id=g.label_id, area=g.area,
                x=g.bbox[0], y=g.bbox[1], w=g.bbox[2], h=g.bbox[3],
                weak_label=int(r.cls == CLS_REFRACTORY),
                **dict(zip(FEATURE_NAMES, g.features)),
            ))
    return pd.DataFrame(rows)


def train_classifier(df: pd.DataFrame, val_groups: set | None = None, seed: int = 0):
    import lightgbm as lgb

    if val_groups is None:
        val_groups = set()
    tr = df[~df.group.isin(val_groups)]
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1)
    model.fit(tr[FEATURE_NAMES].values, tr.weak_label.values,
              sample_weight=np.sqrt(tr.area.values))
    return model


def predict_grains(model, grains: list[Grain]) -> np.ndarray:
    if not grains:
        return np.zeros(0)
    return model.predict_proba(np.stack([g.features for g in grains]))[:, 1]


GROUP_PX = 101  # зёрна ближе этого расстояния — одна область


def group_regions(envelopes: np.ndarray, grains: list[Grain],
                  group_px: int = GROUP_PX) -> np.ndarray:
    """id области для каждого зерна: близкие оболочки склеиваются."""
    if not grains:
        return np.zeros(0, int)
    m = (envelopes > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (group_px, group_px))
    _, reg = cv2.connectedComponents(cv2.dilate(m, k), connectivity=8)
    out = np.zeros(len(grains), int)
    for i, g in enumerate(grains):
        x, y, w, h = g.bbox
        sub = envelopes[y:y + h, x:x + w] == g.label_id
        ys, xs = np.nonzero(sub)
        out[i] = int(reg[y + ys[0], x + xs[0]])
    return out


def smooth_grain_proba(grains: list[Grain], proba: np.ndarray,
                       regions: np.ndarray) -> np.ndarray:
    """Вероятность области (средневзвешенная по площади) вместо пёстрых
    вердиктов по осколкам одного скопления."""
    out = proba.astype(float).copy()
    for rid in np.unique(regions):
        idx = np.flatnonzero(regions == rid)
        w = np.array([grains[i].area for i in idx], float)
        out[idx] = float(np.average(proba[idx], weights=w))
    return out


def image_fine_share(grains: list[Grain], proba: np.ndarray, thr: float = 0.5) -> dict:
    """Агрегация: доли площадей обычных/тонких срастаний (по оболочкам зёрен)."""
    if not grains:
        return dict(area_ordinary=0.0, area_fine=0.0, fine_share=0.0)
    areas = np.array([g.area for g in grains], dtype=float)
    fine = proba >= thr
    a_fine, a_ord = float(areas[fine].sum()), float(areas[~fine].sum())
    total = a_fine + a_ord
    return dict(area_ordinary=a_ord, area_fine=a_fine,
                fine_share=a_fine / total if total else 0.0)


def save_model(model, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    import joblib
    return joblib.load(path)
