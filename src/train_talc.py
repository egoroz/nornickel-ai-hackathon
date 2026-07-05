"""Тальк: сборка пиксельного датасета и оценка MAE/IoU/HD95.

  python -m src.train_talc --dataset-only   # только pixels.npz
  python -m src.train_talc --eval           # оценка модели из artifacts/talc/
  python -m src.train_talc --train          # локальное обучение

Валидация группами по образцам.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import ARTIFACTS, PROJECT_ROOT
from .data import add_groups, build_index
from .sulfides import segment_sulfides
from .talc import (build_feature_planes, build_pixel_dataset, gt_mask_path,
                   predict_mask, save_model, train)

OUT = ARTIFACTS / "talc"
VAL_FRAC = 0.25
SEED = 0


def main() -> None:
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_only = "--dataset-only" in sys.argv
    do_train = "--train" in sys.argv

    idx = add_groups(build_index())
    ann = idx[idx.annot_path.notna()].reset_index(drop=True)

    rng = np.random.RandomState(SEED)
    groups = sorted(ann.group.unique())
    val_groups = set(rng.choice(groups, max(int(len(groups) * VAL_FRAC), 3),
                                replace=False))
    tr_idx = ann[~ann.group.isin(val_groups)]
    va_idx = ann[ann.group.isin(val_groups)]
    print(f"фото: train {len(tr_idx)} / val {len(va_idx)} "
          f"(групп: {len(groups) - len(val_groups)}/{len(val_groups)})")

    tr_full = idx[~idx.group.isin(val_groups)]
    X, y, grp = build_pixel_dataset(tr_full)
    print(f"пикселей на обучение: {len(X)}, доля талька: {y.mean():.2f}")
    np.savez_compressed(OUT / "pixels.npz", X=X, y=y, groups=grp)
    if dataset_only:
        return

    if do_train:
        model = train(X, y, seed=SEED)
        save_model(model, OUT / "model.joblib")
    else:
        from .talc import load_model as _lm
        mp = OUT / "model.joblib"
        assert mp.exists(), "нет artifacts/talc/model.joblib"
        model = _lm(mp)

    # MAE долей на валидационных фото
    from .talc import SUSPECT_STEMS, load_name_mapping
    name_map = load_name_mapping(PROJECT_ROOT)
    rows = []
    for _, r in va_idx.iterrows():
        stem = Path(r.file).stem
        mp = gt_mask_path(stem, PROJECT_ROOT, new_name=name_map.get(r.path))
        img = cv2.imread(r.path)
        gt_region = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127)
        sulf, _ = segment_sulfides(img)
        specks, speck_share, zones = predict_mask(img, model, exclude_mask=sulf)
        gt_zone_share = float(gt_region.mean())
        zb, gb = zones.astype(bool), gt_region
        inter, union = float((zb & gb).sum()), float((zb | gb).sum())
        from .talc import hausdorff_95, speck_mask
        hd = hausdorff_95(zones, gt_region.astype(np.uint8))
        gt_speck = float((gt_region & (speck_mask(img) > 0) & (sulf == 0)).mean())
        rows.append(dict(file=r.file, suspect=stem in SUSPECT_STEMS,
                         gt=gt_zone_share, pred=float(zones.mean()),
                         err=abs(gt_zone_share - float(zones.mean())),
                         speck_err=abs(gt_speck - speck_share),
                         iou=inter / union if union else 1.0,
                         hd95=hd))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "val_shares.csv", index=False)
    mae = float(df.err.mean())
    clean = df[~df.suspect]
    mae_clean = float(clean.err.mean()) if len(clean) else mae
    print(df.to_string(index=False))
    print(f"\nMAE доли ЗОН: {mae*100:.2f} п.п.; без сомнительных масок: "
          f"{mae_clean*100:.2f} п.п.; MAE вкраплений: {df.speck_err.mean()*100:.2f} п.п.; "
          f"IoU зон (медиана): {df.iou.median():.2f}; HD95: {df.hd95.median():.0f} px")
    (OUT / "metrics.json").write_text(json.dumps(dict(
        mae_share=mae, mae_share_clean=mae_clean,
        median_iou=float(df.iou.median()),
        median_hd95=float(df.hd95.median()),
        n_val=len(df),
        feature_importance=dict(zip(
            ["gray_rank", "warm_rank", "local_std", "dark_dens",
             "vdark_dens", "grad_dens"],
            model.feature_importances_.tolist()))), ensure_ascii=False, indent=2))
    print("сохранено в", OUT)


if __name__ == "__main__":
    main()
