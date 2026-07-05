"""Детекция талька: пиксельный классификатор.

Признаки ранговые/текстурные (устойчивы к экспозиции и балансу белого):
gray_rank, warm_rank — квантильные ранги яркости и R-B внутри изображения;
local_std — шероховатость; dark/vdark_dens — плотность тёмного (p25/p10);
grad_dens — крапчатость; *_wide/xwide — те же в широких окнах.
Инференс на 1/4 разрешения, сульфиды исключаются заранее.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FEATURE_NAMES = ["gray_rank", "warm_rank", "local_std", "dark_dens",
                 "vdark_dens", "grad_dens", "dark_dens_wide", "std_wide",
                 "dark_dens_xwide", "vdark_xwide"]
SCALE = 4
WIN_STD = 9
WIN_DENS = 15
WIN_WIDE = 61
WIN_XWIDE = 121
SMOOTH = 31

# сомнительные автомаски — исключены из обучения до правки экспертом
SUSPECT_STEMS = {"2550376-2 10x", "2550377-1 5x", "DSCN3056", "DSCN4717"}


def _rank_plane(plane: np.ndarray, ref: np.ndarray) -> np.ndarray:
    qs = np.quantile(ref, np.linspace(0, 1, 65))
    return (np.searchsorted(qs, plane, side="right") / len(qs)).astype(np.float32)


def build_feature_planes(bgr: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """(h, w, F) float32 на уменьшенном разрешении."""
    small = cv2.resize(bgr, (bgr.shape[1] // scale, bgr.shape[0] // scale),
                       interpolation=cv2.INTER_AREA)
    b = small[:, :, 0].astype(np.float32)
    g = small[:, :, 1].astype(np.float32)
    r = small[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    warm = r - b
    sub = gray[::4, ::4]

    gray_rank = _rank_plane(gray, sub)
    warm_rank = _rank_plane(warm, warm[::4, ::4])

    mean = cv2.boxFilter(gray, -1, (WIN_STD, WIN_STD))
    sq = cv2.boxFilter(gray * gray, -1, (WIN_STD, WIN_STD))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    # нормировка на размах яркости: std сопоставим между камерами
    span = max(float(np.percentile(sub, 99) - np.percentile(sub, 1)), 1.0)
    local_std = (std / span).astype(np.float32)

    p25, p10 = np.percentile(sub, 25), np.percentile(sub, 10)
    dark_dens = cv2.boxFilter((gray < p25).astype(np.float32), -1, (WIN_DENS, WIN_DENS))
    vdark_dens = cv2.boxFilter((gray < p10).astype(np.float32), -1, (WIN_DENS, WIN_DENS))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy) / span
    grad_dens = cv2.boxFilter((mag > 0.5).astype(np.float32), -1, (WIN_DENS, WIN_DENS))

    dark_bin = (gray < p25).astype(np.float32)
    vdark_bin = (gray < p10).astype(np.float32)
    dark_wide = cv2.boxFilter(dark_bin, -1, (WIN_WIDE, WIN_WIDE))
    std_wide = cv2.boxFilter(std, -1, (WIN_WIDE, WIN_WIDE)) / span
    dark_xwide = cv2.boxFilter(dark_bin, -1, (WIN_XWIDE, WIN_XWIDE))
    vdark_xwide = cv2.boxFilter(vdark_bin, -1, (WIN_XWIDE, WIN_XWIDE))

    return np.stack([gray_rank, warm_rank, local_std, dark_dens,
                     vdark_dens, grad_dens, dark_wide, std_wide,
                     dark_xwide, vdark_xwide], axis=-1)


def gt_mask_path(stem: str, project_root: Path,
                 new_name: str | None = None) -> Path | None:
    """Маска эксперта (data/dataset/talc_masks) приоритетнее автоматической."""
    for cand_stem in filter(None, [new_name, stem]):
        p = project_root / "data" / "dataset" / "talc_masks" / f"{Path(cand_stem).stem}.png"
        if p.exists():
            return p
    auto = project_root / "artifacts" / "talc_masks" / f"{stem}.png"
    return auto if auto.exists() else None


def load_name_mapping(project_root: Path) -> dict:
    """original path → нормализованное имя файла в data/dataset."""
    mp = project_root / "data" / "dataset" / "photos" / "metadata.csv"
    if not mp.exists():
        return {}
    df = pd.read_csv(mp)
    return {r.original: Path(r.file).name for r in df.itertuples()}


def build_pixel_dataset(index: pd.DataFrame, n_per_image: int = 12000,
                        seed: int = 0, n_negative_images: int = 60,
                        n_per_negative: int = 5000):
    """Пиксели (вне сульфидов) с размеченных фото + негативы с
    рядовых/труднообогатимых: без них тёмная матрица путается с тальком."""
    from .config import CLS_ORDINARY, CLS_REFRACTORY, PROJECT_ROOT
    from .sulfides import segment_sulfides

    rng = np.random.RandomState(seed)
    name_map = load_name_mapping(PROJECT_ROOT)
    ann = index[index.annot_path.notna()]
    X, y, grp = [], [], []

    neg = index[index.cls.isin([CLS_ORDINARY, CLS_REFRACTORY])]
    if "label_conflict" in neg.columns:
        neg = neg[~neg.label_conflict]  # дубли отальк-фото в «рядовых» — не негативы
    neg = neg.sample(n=min(n_negative_images, len(neg)), random_state=rng)
    for _, r in neg.iterrows():
        img = cv2.imread(r.path)
        if img is None:
            continue
        sulf, _ = segment_sulfides(img)
        feats = build_feature_planes(img)
        h, w = feats.shape[:2]
        valid = ~(cv2.resize(sulf, (w, h)) > 0)
        ys, xs = np.where(valid)
        k = min(n_per_negative, len(ys))
        sel = rng.choice(len(ys), k, replace=False)
        X.append(feats[ys[sel], xs[sel]])
        y.append(np.zeros(k, dtype=bool))
        grp += [r.group if "group" in r else r.sample_id] * k

    for _, r in ann.iterrows():
        stem = Path(r.file).stem
        mp = gt_mask_path(stem, PROJECT_ROOT, new_name=name_map.get(r.path))
        expert = mp is not None and "dataset" in str(mp)
        if stem in SUSPECT_STEMS and not expert:
            continue
        if mp is None:
            continue
        img = cv2.imread(r.path)
        gt = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127)
        sulf, _ = segment_sulfides(img)
        feats = build_feature_planes(img)
        h, w = feats.shape[:2]
        gt_s = cv2.resize(gt.astype(np.uint8), (w, h)) > 0
        sulf_s = cv2.resize(sulf, (w, h)) > 0
        valid = ~sulf_s
        ys, xs = np.where(valid)
        k = min(n_per_image, len(ys))
        sel = rng.choice(len(ys), k, replace=False)
        X.append(feats[ys[sel], xs[sel]])
        y.append(gt_s[ys[sel], xs[sel]])
        grp += [r.group if "group" in r else r.sample_id] * k
    return np.concatenate(X), np.concatenate(y), np.array(grp)


def train(X: np.ndarray, y: np.ndarray, seed: int = 0):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08,
                               num_leaves=31, subsample=0.8,
                               colsample_bytree=0.9, random_state=seed,
                               verbose=-1)
    model.fit(X, y)
    return model


def predict_proba_map(bgr: np.ndarray, model, scale: int = SCALE) -> np.ndarray:
    """Сглаженная карта вероятностей талька на уменьшенном разрешении."""
    feats = build_feature_planes(bgr, scale)
    h, w = feats.shape[:2]
    proba = model.predict_proba(feats.reshape(-1, feats.shape[-1]))[:, 1]
    return cv2.boxFilter(proba.reshape(h, w).astype(np.float32), -1,
                         (SMOOTH, SMOOTH))


MASK_THR = 0.30
SPECK_PCTL = 25


def speck_mask(bgr: np.ndarray) -> np.ndarray:
    """Тёмные вкрапления: пиксели темнее p25 изображения."""
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    thr = np.percentile(gray[::4, ::4], SPECK_PCTL)
    return (gray < thr).astype(np.uint8)


def predict_mask(bgr: np.ndarray, model, exclude_mask: np.ndarray | None = None,
                 thr: float = MASK_THR, scale: int = SCALE,
                 proba: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """(вкрапления uint8 0/1, их доля, зоны оталькования uint8 0/1).
    Вкрапления = зоны (порог по сглаженной карте) & тёмные пиксели."""
    if proba is None:
        proba = predict_proba_map(bgr, model, scale)
    # на панорамах вероятности сжаты (текстура задавлена jpeg) —
    # порог адаптивно опускается к 0.6*max, но не ниже 0.15
    pmax = float(proba.max())
    thr_eff = thr if pmax >= 0.5 else max(0.15, 0.6 * pmax)
    m = (proba >= thr_eff).astype(np.uint8)
    region = cv2.resize(m, (bgr.shape[1], bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    full = (region & speck_mask(bgr)).astype(np.uint8)
    if exclude_mask is not None:
        full &= (exclude_mask == 0)
        region = (region & (exclude_mask == 0)).astype(np.uint8)
    share = float(full.mean())
    return full, share, region


def save_model(model, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    import joblib
    return joblib.load(path)


def hausdorff_95(pred: np.ndarray, gt: np.ndarray, scale: int = 4) -> float:
    """HD95 между границами масок, px полного разрешения."""
    p = pred[::scale, ::scale].astype(np.uint8)
    g = gt[::scale, ::scale].astype(np.uint8)
    if p.max() == 0 and g.max() == 0:
        return 0.0
    if p.max() == 0 or g.max() == 0:
        return float(max(p.shape) * scale)
    pb = p - cv2.erode(p, np.ones((3, 3), np.uint8))
    gb = g - cv2.erode(g, np.ones((3, 3), np.uint8))
    dt_g = cv2.distanceTransform(1 - gb, cv2.DIST_L2, 5)
    dt_p = cv2.distanceTransform(1 - pb, cv2.DIST_L2, 5)
    d_pg = dt_g[pb > 0]
    d_gp = dt_p[gb > 0]
    if len(d_pg) == 0 or len(d_gp) == 0:
        return float(max(p.shape) * scale)
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)) * scale)


UNET_PATH_DEFAULT = "artifacts/talc/unet.pt"


def load_unet(path: Path):
    """None, если весов или зависимостей нет — работает бустинг-базлайн."""
    if not Path(path).exists():
        return None
    try:
        import segmentation_models_pytorch as smp
        import torch
    except ImportError:
        print("segmentation-models-pytorch не установлен — тальк считает бустинг")
        return None
    model = smp.Unet("resnet18", encoder_weights=None, classes=1)
    model.load_state_dict(torch.load(path, map_location="cpu",
                                     weights_only=True))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def predict_proba_map_unet(bgr: np.ndarray, unet, tile: int = 512,
                           overlap: int = 64, scale: int = 1) -> np.ndarray:
    """Карта вероятностей U-Net тайлами (GPU при наличии), в масштабе 1/scale."""
    import torch
    device = next(unet.parameters()).device
    img = bgr if scale == 1 else cv2.resize(
        bgr, (bgr.shape[1] // scale, bgr.shape[0] // scale),
        interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    out = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    step = tile - overlap
    with torch.no_grad():
        for y0 in range(0, max(H - overlap, 1), step):
            for x0 in range(0, max(W - overlap, 1), step):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                patch = rgb[y0:y1, x0:x1]
                ph, pw = patch.shape[:2]
                pad_h = (32 - ph % 32) % 32
                pad_w = (32 - pw % 32) % 32
                if pad_h or pad_w:
                    patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w,
                                               cv2.BORDER_REFLECT)
                x = torch.from_numpy(
                    patch.transpose(2, 0, 1)[None]).float().to(device) / 255.
                p = torch.sigmoid(unet(x))[0, 0].cpu().numpy()[:ph, :pw]
                out[y0:y1, x0:x1] += p
                cnt[y0:y1, x0:x1] += 1
    return out / np.maximum(cnt, 1)
