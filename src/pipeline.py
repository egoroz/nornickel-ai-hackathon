"""Батч-пайплайн: фото целиком, панорамы тайлами (pyvips, без полной
декомпрессии). Фазовая модель фитится по уменьшенной панораме один раз,
чтобы тайлы сегментировались согласованно.

  python -m src.pipeline --input <файл|папка> --out <папка>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .classify import AnalysisResult, OreAnalyzer
from .config import IMG_EXTS

TILE = 2048        # единый размер тайла в проекте (как в разметке)
OVERLAY_SCALE = 8
PANORAMA_MPX = 30  # крупнее — тайловый режим


def _vips_to_np(v) -> np.ndarray:
    arr = np.ndarray(buffer=v.write_to_memory(), dtype=np.uint8,
                     shape=[v.height, v.width, v.bands])
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)


def analyze_panorama(path: Path, analyzer: OreAnalyzer,
                     tile: int = TILE, progress=None) -> AnalysisResult:
    import pyvips
    from .intergrowth import (analyze_image, group_regions, image_fine_share,
                              predict_grains, smooth_grain_proba)
    from .sulfides import fit_phase_model
    from .talc import predict_mask, predict_proba_map, predict_proba_map_unet

    img = pyvips.Image.new_from_file(str(path))
    W, H = img.width, img.height

    thumb = _vips_to_np(pyvips.Image.thumbnail(str(path), 2000))
    centers = fit_phase_model(thumb, sample_step=2)
    from .sulfides import label_sulfide_clusters
    sulf_ids = np.flatnonzero(label_sulfide_clusters(centers))

    ov_w, ov_h = W // OVERLAY_SCALE, H // OVERLAY_SCALE
    overlay = np.zeros((ov_h, ov_w, 3), np.uint8)

    acc = dict(sulf=0.0, ordinary=0.0, fine=0.0, talc_px=0.0, zone_px=0.0,
               valid=0.0, total=0.0, n_grains=0, fines_px=0.0, repl_w=0.0,
               gray=0.0)
    grain_areas: list = []
    tile_rows = []
    tiles = [(x0, y0) for y0 in range(0, H, tile) for x0 in range(0, W, tile)]
    for i, (x0, y0) in enumerate(tiles):
        tw, th = min(tile, W - x0), min(tile, H - y0)
        if tw < 256 or th < 256:
            continue
        bgr = _vips_to_np(img.crop(x0, y0, tw, th))

        res = analyze_image(bgr, centers=centers)
        sulf, env, grains = res["sulfide_mask"], res["envelopes"], res["grains"]
        proba_ig = predict_grains(analyzer.ig_model, grains)
        proba_ig = smooth_grain_proba(grains, proba_ig,
                                      group_regions(env, grains))
        shares = image_fine_share(grains, proba_ig, analyzer.fine_grain_thr)

        if analyzer.talc_unet is not None:
            tp = predict_proba_map_unet(bgr, analyzer.talc_unet, scale=4)
        else:
            tp = predict_proba_map(bgr, analyzer.talc_model)
        exclude = (sulf > 0).astype(np.uint8)
        talc_mask, _, talc_zones = predict_mask(bgr, analyzer.talc_model,
                                                exclude_mask=exclude, proba=tp)

        valid = exclude == 0
        acc["talc_px"] += float(talc_mask.sum())
        acc["zone_px"] += float(talc_zones.sum())
        acc["valid"] += float(valid.sum())
        acc["total"] += float(tw * th)
        acc["sulf"] += float(sulf.sum())
        acc["gray"] += float(((~np.isin(res["phase"], sulf_ids))
                              & (res["phase"] != 0)).sum())
        acc["ordinary"] += shares["area_ordinary"]
        acc["fine"] += shares["area_fine"]
        acc["n_grains"] += len(grains)
        acc["fines_px"] += float(((sulf > 0) & (env == 0)).sum())
        for g in grains:
            grain_areas.append(g.area)
            acc["repl_w"] += (1 - g.features[0]) * g.area
        t_zone = float(talc_zones.mean())
        t_fine = shares["fine_share"]
        t_growth = (shares["area_ordinary"] + shares["area_fine"]) / float(tw * th)
        if t_zone > analyzer.talc_threshold:
            t_cls = "оталькованная"
        elif t_growth < 0.005:
            t_cls = "пусто"
        else:
            t_cls = "труднообогатимая" if t_fine > 0.5 else "рядовая"
        tile_rows.append(dict(x0=x0, y0=y0, w=tw, h=th, cls=t_cls,
                              talc_zone=t_zone, fine_dom=t_fine))

        ov_tile = OreAnalyzer.make_overlay(bgr, sulf, env, grains, proba_ig,
                                           talc_mask, talc_zones=talc_zones,
                                           thr=analyzer.fine_grain_thr)
        x0s, y0s = x0 // OVERLAY_SCALE, y0 // OVERLAY_SCALE
        x1s = min(x0s + tw // OVERLAY_SCALE, ov_w)
        y1s = min(y0s + th // OVERLAY_SCALE, ov_h)
        if x1s > x0s and y1s > y0s:
            overlay[y0s:y1s, x0s:x1s] = cv2.resize(ov_tile, (x1s - x0s, y1s - y0s),
                                                   interpolation=cv2.INTER_AREA)
        if progress:
            progress((i + 1) / len(tiles))

    talc_share = acc["talc_px"] / max(acc["total"], 1.0)
    zone_share = acc["zone_px"] / max(acc["total"], 1.0)
    growth_total = acc["ordinary"] + acc["fine"]
    fine_dom = acc["fine"] / growth_total if growth_total else 0.0
    ore_class = analyzer._rule(zone_share, fine_dom)

    cmap = {"рядовая": (0, 200, 0), "труднообогатимая": (0, 0, 220),
            "оталькованная": (255, 80, 0), "пусто": (128, 128, 128)}
    tile_map = overlay.copy()
    for r in tile_rows:
        x0s, y0s = r["x0"] // OVERLAY_SCALE, r["y0"] // OVERLAY_SCALE
        x1s, y1s = x0s + r["w"] // OVERLAY_SCALE, y0s + r["h"] // OVERLAY_SCALE
        cv2.rectangle(tile_map, (x0s + 1, y0s + 1), (x1s - 2, y1s - 2),
                      cmap.get(r["cls"], (255, 255, 255)), 2)

    total = max(acc["total"], 1.0)
    return AnalysisResult(
        ore_class=ore_class, talc_share=talc_share,
        gray_share=acc["gray"] / total,
        nonore_share=max(1.0 - (acc["sulf"] + acc["gray"]) / total, 0.0),
        sulfide_share=acc["sulf"] / max(acc["total"], 1.0),
        ordinary_share=acc["ordinary"] / max(acc["total"], 1.0),
        fine_share=acc["fine"] / max(acc["total"], 1.0),
        fine_dominance=fine_dom, n_grains=acc["n_grains"],
        talc_zone_share=zone_share, tile_map=tile_map,
        fines_share=acc["fines_px"] / max(acc["total"], 1.0),
        median_grain_px=float(np.median(grain_areas)) if grain_areas else 0.0,
        mean_replacement=(acc["repl_w"] / sum(grain_areas)) if grain_areas else 0.0,
        overlay=overlay,
        conclusion=analyzer._conclusion(ore_class, zone_share, fine_dom),
        params=dict(tile=tile, overlay_scale=OVERLAY_SCALE,
                    talc_model="unet" if analyzer.talc_unet is not None else "lgbm",
                    talc_threshold=analyzer.talc_threshold),
    )


def analyze_path(path: Path, analyzer: OreAnalyzer, progress=None) -> AnalysisResult:
    """Авторежим: маленькое фото целиком, панорама — тайлами."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        mpx = im.size[0] * im.size[1] / 1e6
    if mpx > PANORAMA_MPX:
        return analyze_panorama(path, analyzer, progress=progress)
    bgr = cv2.imread(str(path))
    return analyzer.analyze(bgr)


def process(path: Path, out_dir: Path, analyzer: OreAnalyzer) -> AnalysisResult:
    from .report import save_csv, save_json, save_pdf
    t0 = time.time()
    res = analyze_path(path, analyzer)
    stem = path.stem
    d = out_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "overlay.jpg"), res.overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if res.tile_map is not None:
        cv2.imwrite(str(d / "tile_map.jpg"), res.tile_map, [cv2.IMWRITE_JPEG_QUALITY, 90])
    save_csv(res, d / "metrics.csv")
    save_json(res, d / "result.json", source=str(path))
    save_pdf(res, d / "report.pdf", source=path.name)
    print(f"{path.name}: {res.ore_class}, тальк {res.talc_share*100:.1f}%, "
          f"тонкие {res.fine_dominance*100:.0f}% [{time.time()-t0:.0f} c]")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="файл или папка")
    ap.add_argument("--out", required=True)
    ap.add_argument("--talc-threshold", type=float, default=None)
    a = ap.parse_args()

    analyzer = OreAnalyzer()
    if a.talc_threshold is not None:
        analyzer.talc_threshold = a.talc_threshold

    src = Path(a.input)
    files = ([src] if src.is_file() else
             sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS))
    out = Path(a.out)
    rows = []
    for p in files:
        try:
            res = process(p, out, analyzer)
            rows.append(dict(file=p.name, ore_class=res.ore_class,
                             talc_zones=res.talc_zone_share, talc=res.talc_share,
                             fine_dom=res.fine_dominance,
                             sulfides=res.sulfide_share))
        except Exception as e:
            print(f"!! {p.name}: {e}")
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
        print(f"\nсводка: {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
