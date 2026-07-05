"""Индекс исходных данных и парсер синих контуров талька.

Контуры эксперта часто не замкнуты (линия обрывается у границы кадра или
у сульфида) — маска строится через «стены»: штрих + мостики конец-конец /
конец-граница / конец-сульфид, затем выбор стороны заливки.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import (DIR_PANORAMAS, DIR_PART1, DIR_PART2, IMG_EXTS,
                     CLS_ORDINARY, CLS_REFRACTORY, CLS_TALC)

_PART1_DIRS = {
    "Рядовые руды": CLS_ORDINARY,
    "Труднообогатимые руды": CLS_REFRACTORY,
    "Оталькованные руды": CLS_TALC,
}
_PART2_DIRS = {
    "рядовые": CLS_ORDINARY,
    "тонкие": CLS_REFRACTORY,
    "оталькованные": CLS_TALC,
}
ANNOT_SUBDIR = "Области оталькования"
_SAMPLE_RE = re.compile(r"^(\d{6,8})\b")


def _sample_id(stem: str) -> str:
    m = _SAMPLE_RE.match(stem)
    return m.group(1) if m else stem


def _list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def build_index() -> pd.DataFrame:
    """DataFrame по всем размеченным фото: path, cls, part, sample_id, annot_path."""
    rows = []
    for dname, cls in _PART1_DIRS.items():
        d = DIR_PART1 / dname
        annot_dir = d / ANNOT_SUBDIR
        for p in _list_images(d):
            annot = annot_dir / p.name
            rows.append(dict(
                path=str(p), file=p.name, cls=cls, part="ч1",
                sample_id=_sample_id(p.stem),
                annot_path=str(annot) if annot.exists() else None,
            ))
    for dname, cls in _PART2_DIRS.items():
        for p in _list_images(DIR_PART2 / dname):
            rows.append(dict(
                path=str(p), file=p.name, cls=cls, part="ч2",
                sample_id=p.stem, annot_path=None,
            ))
    return pd.DataFrame(rows)


def list_panoramas() -> list[Path]:
    return _list_images(DIR_PANORAMAS)


def _dhash(path: str, size: int = 8) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        im.draft("L", (size * 8, size * 8))
        g = np.asarray(im.convert("L").resize((size + 1, size)), dtype=int)
    return (g[:, 1:] > g[:, :-1]).ravel()


def add_groups(index: pd.DataFrame, max_hamming: int = 4) -> pd.DataFrame:
    """Колонка group: dhash-дубли и фото одного образца — одна группа,
    иначе валидация утекает (в данных >100 дублей, часть между классами)."""
    hashes = np.stack([_dhash(p) for p in index.path])
    parent = list(range(len(index)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(len(index)):
        d = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        for j in np.flatnonzero(d <= max_hamming):
            union(i, i + 1 + j)

    by_sample: dict[str, int] = {}
    for i, sid in enumerate(index.sample_id):
        if sid in by_sample:
            union(i, by_sample[sid])
        else:
            by_sample[sid] = i

    index = index.copy()
    index["group"] = [f"g{find(i)}" for i in range(len(index))]
    # разные классы в одной группе дублей — слабой метке верить нельзя
    span = index.groupby("group")["cls"].nunique()
    index["label_conflict"] = index.group.map(span) > 1
    return index


STROKE_BLUE_MARGIN = 40   # B - max(R,G) порог синего штриха
GAP_CLOSE = 15
R_ENDPOINT = 300          # мостик конец-конец, px
R_BORDER = 150            # конец-граница кадра
R_SULFIDE = 250           # конец-сульфид
BRIDGE_W = 7


@dataclass
class TalcParse:
    mask: np.ndarray           # области оталькования, uint8 0/1
    stroke: np.ndarray         # синий штрих
    talc_share: float
    n_endpoints: int = 0
    n_pair: int = 0
    n_border: int = 0
    n_sulfide: int = 0
    n_unresolved: int = 0
    ok: bool = True            # False — остались неприкрытые концы


def extract_stroke(annot_bgr: np.ndarray) -> np.ndarray:
    b = annot_bgr[:, :, 0].astype(int)
    g = annot_bgr[:, :, 1].astype(int)
    r = annot_bgr[:, :, 2].astype(int)
    return ((b - np.maximum(g, r)) > STROKE_BLUE_MARGIN).astype(np.uint8)


def _skeleton_endpoints(binary: np.ndarray) -> list[tuple[int, int]]:
    from skimage.morphology import skeletonize
    sk = skeletonize(binary > 0)
    nb = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    ys, xs = np.where(sk & (nb == 2))
    return list(zip(xs.tolist(), ys.tolist()))


def _nearest_border_point(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    d = [(x, (0, y)), (w - 1 - x, (w - 1, y)), (y, (x, 0)), (h - 1 - y, (x, h - 1))]
    return min(d)[1]


def talc_mask_from_annotation(
    orig_bgr: np.ndarray,
    annot_bgr: np.ndarray,
    sulfide_mask: np.ndarray | None = None,
) -> TalcParse:
    h, w = annot_bgr.shape[:2]
    stroke = extract_stroke(annot_bgr)
    walls = cv2.morphologyEx(
        stroke, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GAP_CLOSE, GAP_CLOSE)))

    eps = _skeleton_endpoints(walls)
    used = [False] * len(eps)
    n_pair = n_border = n_sulf = n_unres = 0

    # мостики конец-конец (жадно ближайшие пары)
    pairs = sorted(
        (np.hypot(eps[i][0] - eps[j][0], eps[i][1] - eps[j][1]), i, j)
        for i in range(len(eps)) for j in range(i + 1, len(eps)))
    for d, i, j in pairs:
        if d > R_ENDPOINT:
            break
        if not used[i] and not used[j]:
            cv2.line(walls, eps[i], eps[j], 1, BRIDGE_W)
            used[i] = used[j] = True
            n_pair += 2

    # конец-граница кадра
    for i, (x, y) in enumerate(eps):
        if used[i]:
            continue
        if min(x, w - 1 - x, y, h - 1 - y) < R_BORDER:
            cv2.line(walls, (x, y), _nearest_border_point(x, y, w, h), 1, BRIDGE_W)
            used[i] = True
            n_border += 1

    # конец-сульфид: линия часто обрывается у светлой фазы
    dangling = [eps[i] for i in range(len(eps)) if not used[i]]
    if dangling:
        if sulfide_mask is None:
            from .sulfides import segment_sulfides
            sulfide_mask, _ = segment_sulfides(orig_bgr)
        n_cc, cc = cv2.connectedComponents(sulfide_mask, connectivity=8)
        sy, sx = np.nonzero(sulfide_mask)
        keep_cc = set()
        for (x, y) in dangling:
            if len(sx) == 0:
                n_unres += 1
                continue
            d2 = (sx - x) ** 2 + (sy - y) ** 2
            k = int(d2.argmin())
            if d2[k] > R_SULFIDE ** 2:
                n_unres += 1
                continue
            cv2.line(walls, (x, y), (int(sx[k]), int(sy[k])), 1, BRIDGE_W)
            keep_cc.add(int(cc[sy[k], sx[k]]))
            n_sulf += 1
        if keep_cc:
            walls |= np.isin(cc, list(keep_cc)).astype(np.uint8)

    walls[0, :] = walls[-1, :] = 1
    walls[:, 0] = walls[:, -1] = 1

    n_lab, lab = cv2.connectedComponents((1 - walls).astype(np.uint8), connectivity=4)
    sizes = np.bincount(lab.ravel(), minlength=n_lab)
    sizes[0] = 0

    # выбор стороны заливки: обводится тёмная крапчатая область
    if sulfide_mask is None:
        from .sulfides import segment_sulfides
        sulfide_mask, _ = segment_sulfides(orig_bgr)
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    dark_thr = float(np.percentile(gray[::4, ::4], 35))

    min_region = int(0.0005 * h * w)
    border = np.zeros((h, w), bool)
    border[:3, :] = border[-3:, :] = border[:, :3] = border[:, -3:] = True
    stats = {}
    for lid in range(1, n_lab):
        if sizes[lid] < min_region:
            continue
        m = lab == lid
        g = gray[m]
        md = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        stats[lid] = dict(
            dark=float((g < dark_thr).mean()),
            sulf=float(sulfide_mask[m].mean()),
            area=int(sizes[lid]),
            touches_border=bool((md & border).any()),
        )

    n_scc, scc = cv2.connectedComponents(stroke, connectivity=8)
    talc_ids: set[int] = set()
    for sid in range(1, n_scc):
        ring = cv2.dilate((scc == sid).astype(np.uint8),
                          np.ones((41, 41), np.uint8)).astype(bool)
        adj = [lid for lid in stats if (lab[ring] == lid).any()]
        adj = [lid for lid in adj if stats[lid]["sulf"] < 0.5]
        if not adj:
            continue
        # замкнутый контур: внутренность (не касается рамки) и есть область
        enclosed = [lid for lid in adj if not stats[lid]["touches_border"]]
        if len(enclosed) == 1:
            talc_ids.add(enclosed[0])
            continue
        if enclosed and len(enclosed) < len(adj):
            for lid in enclosed:
                talc_ids.add(lid)
            continue
        # открытый контур: берём тёмную сторону
        adj.sort(key=lambda lid: stats[lid]["dark"], reverse=True)
        best = adj[0]
        if len(adj) == 1 or stats[best]["dark"] - stats[adj[1]]["dark"] > 0.06:
            talc_ids.add(best)
        else:
            talc_ids.add(min(adj[:2], key=lambda lid: stats[lid]["area"]))

    mask = np.isin(lab, list(talc_ids)).astype(np.uint8) if talc_ids else np.zeros((h, w), np.uint8)
    mask |= (cv2.dilate(mask, np.ones((21, 21), np.uint8)) & stroke).astype(np.uint8)
    mask_pure = (mask & (1 - sulfide_mask)).astype(np.uint8)  # сульфиды — не тальк

    return TalcParse(
        mask=mask_pure, stroke=stroke, talc_share=float(mask_pure.mean()),
        n_endpoints=len(eps), n_pair=n_pair, n_border=n_border,
        n_sulfide=n_sulf, n_unresolved=n_unres, ok=(n_unres == 0),
    )


def make_talc_masks(out_dir: Path, qc_dir: Path | None = None) -> pd.DataFrame:
    from .sulfides import segment_sulfides

    out_dir.mkdir(parents=True, exist_ok=True)
    if qc_dir:
        qc_dir.mkdir(parents=True, exist_ok=True)
    idx = build_index()
    rows = []
    ann = idx[idx.annot_path.notna()]
    for _, r in ann.iterrows():
        orig = cv2.imread(r.path)
        annot = cv2.imread(r.annot_path)
        sulf, _ = segment_sulfides(orig)
        tp = talc_mask_from_annotation(orig, annot, sulfide_mask=sulf)
        stem = Path(r.file).stem
        cv2.imwrite(str(out_dir / f"{stem}.png"), tp.mask * 255)
        if qc_dir:
            vis = orig.copy()
            ov = vis.copy()
            ov[tp.mask > 0] = (255, 120, 60)
            vis = cv2.addWeighted(ov, 0.45, vis, 0.55, 0)
            vis[tp.stroke > 0] = (0, 0, 255)
            tag = "" if tp.ok else " UNRESOLVED"
            cv2.putText(vis, f"{stem} talc={tp.talc_share*100:.1f}%{tag}",
                        (10, 40), 0, 1.2, (0, 0, 0), 6)
            cv2.putText(vis, f"{stem} talc={tp.talc_share*100:.1f}%{tag}",
                        (10, 40), 0, 1.2, (255, 255, 255), 2)
            cv2.imwrite(str(qc_dir / f"{stem}.jpg"),
                        cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2)),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append(dict(file=r.file, talc_share=tp.talc_share,
                         n_endpoints=tp.n_endpoints, n_pair=tp.n_pair,
                         n_border=tp.n_border, n_sulfide=tp.n_sulfide,
                         n_unresolved=tp.n_unresolved, ok=tp.ok))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from .config import ARTIFACTS

    stats = make_talc_masks(ARTIFACTS / "talc_masks", ARTIFACTS / "qc" / "talc")
    stats.to_csv(ARTIFACTS / "talc_masks" / "stats.csv", index=False)
    print(stats.to_string())
    print(f"\nok: {stats.ok.sum()}/{len(stats)}, "
          f"median talc share: {stats.talc_share.median()*100:.1f}%")
