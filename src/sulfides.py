"""Сегментация сульфидов: KMeans пикселей в осях (яркость, R-B),
кластеры размечаются правилами. Сульфид = ярче и теплее матрицы;
это отделяет его от серого магнетита той же яркости при любом
балансе белого камеры.
"""
from __future__ import annotations

import cv2
import numpy as np

N_CLUSTERS = 5
MIN_GRAY_GAP = 25
MIN_WARM_GAP = 12
REL_WARMTH = 0.55
REL_GRAY = 0.45
NEUTRAL_REL_GRAY = 0.72


def pixel_features(bgr: np.ndarray) -> np.ndarray:
    """(N,2) float32: яркость и теплота каждого пикселя."""
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    warmth = r - b
    return np.stack([gray.ravel(), warmth.ravel()], axis=1)


def fit_phase_model(bgr: np.ndarray, sample_step: int = 4,
                    n_clusters: int = N_CLUSTERS, seed: int = 0) -> np.ndarray:
    """Центры кластеров (n,2) по яркости; яркий хвост пересэмплирован,
    иначе на панорамах сульфиды (<1% пикселей) не получают кластера."""
    from sklearn.cluster import MiniBatchKMeans

    sub = bgr[::sample_step, ::sample_step]
    feats = pixel_features(sub)
    bright = feats[feats[:, 0] > np.percentile(feats[:, 0], 99.0)]
    boost = max(int(0.08 * len(feats) / max(len(bright), 1)), 1)
    feats = np.concatenate([feats] + [bright] * boost)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=5,
                         batch_size=4096).fit(feats)
    centers = km.cluster_centers_
    return centers[np.argsort(centers[:, 0])]


def label_sulfide_clusters(centers: np.ndarray) -> np.ndarray:
    """Булев вектор «кластер — сульфид». При нейтральном балансе белого
    (R-B ~ 0 у всех фаз) теплота неинформативна — только яркость."""
    gray_c, warm_c = centers[:, 0], centers[:, 1]
    matrix_gray = gray_c[0]
    g_max, w_max = gray_c.max(), warm_c.max()
    w_span = float(warm_c.max() - warm_c.min())
    is_sulf = np.zeros(len(centers), dtype=bool)
    neutral = w_span < 15
    for i in range(len(centers)):
        if gray_c[i] < matrix_gray + MIN_GRAY_GAP:
            continue
        if neutral:
            is_sulf[i] = (g_max >= matrix_gray + 60
                          and gray_c[i] >= NEUTRAL_REL_GRAY * g_max)
        else:
            bright_enough = gray_c[i] >= REL_GRAY * g_max
            warm_enough = warm_c[i] >= max(REL_WARMTH * w_max,
                                           warm_c[0] + MIN_WARM_GAP)
            is_sulf[i] = bright_enough and warm_enough
    return is_sulf


def assign_clusters(bgr: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Каждому пикселю — индекс ближайшего центра."""
    feats = pixel_features(bgr)
    d = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1).astype(np.uint8).reshape(bgr.shape[:2])


def segment_sulfides(
    bgr: np.ndarray,
    centers: np.ndarray | None = None,
    min_area: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Маска сульфидов (uint8 0/1) + центры. `centers` передаются заранее,
    когда тайлы панорамы должны сегментироваться согласованно."""
    if centers is None:
        centers = fit_phase_model(bgr)
    labels = assign_clusters(bgr, centers)
    sulf_ids = np.flatnonzero(label_sulfide_clusters(centers))
    mask = np.isin(labels, sulf_ids).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if min_area > 1:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < min_area)
        mask[np.isin(lab, small)] = 0
    return mask, centers


def phase_map(bgr: np.ndarray, centers: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Фазовая карта: индексы кластеров по яркости."""
    if centers is None:
        centers = fit_phase_model(bgr)
    return assign_clusters(bgr, centers), centers
