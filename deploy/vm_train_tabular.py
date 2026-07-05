"""Обучение табличных моделей (срастания + тальк) на VM команды.

Без бандлов: входы собираются прямо из чистого датасета
~/shlif/data/dataset (после ручной чистки классов metadata.csv —
источник истины; маски талька — data/dataset/talc_masks).

Запуск на VM:  ~/shlif/venv/bin/python ~/shlif/deploy/vm_train_tabular.py
Выход:  ~/shlif/out/{model_grains.joblib, model_talc.joblib, tabular_metrics.json}
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score)

HOME = Path.home() / "shlif"
TRAIN = HOME / "train"
OUT = HOME / "out"
SEED, VAL_FRAC = 0, 0.2
FEATURES = ["sulf_fill", "gray_fill", "dark_fill", "sulf_solidity",
            "frag_density", "thick_rel", "extent", "log_area_rel",
            "core_fill", "core_frag"]


def gpu_available() -> bool:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def make_model(use_gpu: bool, n_estimators: int, lr: float):
    if use_gpu:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=n_estimators, learning_rate=lr,
                             max_depth=7, subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", device="cuda",
                             random_state=SEED, eval_metric="logloss")
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=lr,
                              num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED,
                              verbose=-1)


def to_cpu(model):
    """XGB, обученный на CUDA, перед сохранением переводим на CPU-инференс."""
    if hasattr(model, "set_params") and "device" in model.get_params():
        model.set_params(device="cpu")
    return model


def _talc_group(stem: str) -> str:
    """Группа для сплита: номер образца из имени ch1, иначе сам stem."""
    parts = stem.split("_")
    if len(parts) > 1 and parts[1].split("-")[0].isdigit():
        return parts[1].split("-")[0]
    return stem


def build_inputs_from_dataset():
    """Пересборка grains.csv и pixels.npz из чистого датасета (без бандлов)."""
    ds = HOME / "data" / "dataset"
    # зёрна — штатный билдер по metadata.csv (уже без удалённых фото);
    # пересборка только если metadata новее готового grains.csv (~25 мин)
    g = TRAIN / "grains.csv"
    md_mtime = (ds / "photos" / "metadata.csv").stat().st_mtime
    if not g.exists() or g.stat().st_mtime < md_mtime:
        g.unlink(missing_ok=True)
        subprocess.run([sys.executable,
                        str(HOME / "deploy" / "vm_build_grains.py")], check=True)
    else:
        print("grains.csv свежее metadata — пересборка зёрен пропущена")

    # тальк: негативы из metadata + позитивы по экспертным маскам talc_masks
    sys.path.insert(0, str(HOME))
    from src.talc import build_pixel_dataset
    md = pd.read_csv(ds / "photos" / "metadata.csv")
    rows = [dict(path=str(ds / f), file=Path(f).name, cls=c,
                 sample_id=str(s), group=_talc_group(Path(f).stem),
                 label_conflict=bool(lc), annot_path=None)
            for f, c, s, lc in zip(md.file, md.cls, md.sample_id,
                                   md.label_conflict)]
    img_dirs = [ds / "talc_annotation", ds / "photos" / "otalkovannaya",
                ds / "photos" / "ryadovaya", ds / "photos" / "trudnoobogatimaya"]
    n_pos = 0
    for mp in sorted((ds / "talc_masks").glob("*.png")):
        img = next((p for d in img_dirs if d.exists()
                    for ext in (".jpg", ".jpeg", ".png", ".bmp")
                    if (p := d / f"{mp.stem}{ext}").exists()), None)
        if img is None:
            print(f"  маска без фото, пропуск: {mp.name}")
            continue
        rows.append(dict(path=str(img), file=img.name, cls="оталькованная",
                         sample_id=mp.stem, group=_talc_group(mp.stem),
                         label_conflict=False, annot_path=str(img)))
        n_pos += 1
    idx = pd.DataFrame(rows)
    print(f"тальк: {n_pos} размеченных фото + негативы из {len(md)} чистых")
    X, y, grp = build_pixel_dataset(idx)
    np.savez_compressed(TRAIN / "pixels.npz", X=X, y=y, groups=grp)
    print(f"pixels.npz: {len(X)} пикселей, доля талька {y.mean():.3f}")

    # производные входы кладём в датасет: notebooks/datasphere_tabular.ipynb
    # читает их из train_inputs/ (в DataSphere нет src/ для пересборки)
    ti = ds / "train_inputs"
    ti.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(TRAIN / "grains.csv", ti / "grains.csv")
    shutil.copy2(TRAIN / "pixels.npz", ti / "pixels.npz")


def main():
    OUT.mkdir(exist_ok=True)
    TRAIN.mkdir(exist_ok=True)
    build_inputs_from_dataset()
    use_gpu = gpu_available()
    print("GPU:", use_gpu)
    metrics = {"gpu": use_gpu}

    df = pd.read_csv(TRAIN / "grains.csv")
    rng = np.random.RandomState(SEED)
    groups = sorted(df.group.unique())
    val_groups = set(rng.choice(groups, int(len(groups) * VAL_FRAC), replace=False))
    tr, va = df[~df.group.isin(val_groups)], df[df.group.isin(val_groups)]
    print(f"grains: train {len(tr)} / val {len(va)}")

    t0 = time.time()
    model_g = make_model(use_gpu, 600, 0.05)
    model_g.fit(tr[FEATURES].values, tr.weak_label.values,
                sample_weight=np.sqrt(tr.area.values))
    t_g = time.time() - t0

    proba = to_cpu(model_g).predict_proba(va[FEATURES].values)[:, 1]
    acc = float(((proba >= 0.5) == va.weak_label).mean())
    auc = float(roc_auc_score(va.weak_label, proba))
    img = []
    for (path, cls), sub in va.groupby(["path", "cls"]):
        p = model_g.predict_proba(sub[FEATURES].values)[:, 1]
        a = sub.area.values.astype(float)
        img.append((cls == "труднообогатимая", a[p >= 0.5].sum() / a.sum()))
    img = pd.DataFrame(img, columns=["y", "fine_share"])
    f1 = float(f1_score(img.y, img.fine_share > 0.5, average="macro"))
    auc_img = float(roc_auc_score(img.y, img.fine_share))
    print(f"зёрна: acc={acc:.3f} AUC={auc:.3f} | изображения: "
          f"F1={f1:.3f} AUC={auc_img:.3f} | {t_g:.0f}с")
    print(confusion_matrix(img.y, img.fine_share > 0.5))
    joblib.dump(model_g, OUT / "model_grains.joblib")
    metrics["grains"] = dict(acc=acc, auc=auc, image_f1=f1, image_auc=auc_img,
                             train_sec=round(t_g, 1), n_train=len(tr), n_val=len(va))

    d = np.load(TRAIN / "pixels.npz", allow_pickle=True)
    X, y, grp = d["X"], d["y"], d["groups"]
    ug = np.unique(grp)
    val_g = set(np.random.RandomState(SEED).choice(
        ug, max(int(len(ug) * 0.2), 2), replace=False))
    vm = np.isin(grp, list(val_g))
    t0 = time.time()
    model_t = make_model(use_gpu, 200, 0.08)
    model_t.fit(X[~vm], y[~vm])
    t_t = time.time() - t0
    p = to_cpu(model_t).predict_proba(X[vm])[:, 1]
    auc_t = float(roc_auc_score(y[vm], p))
    acc_t = float(((p >= 0.5) == y[vm]).mean())
    print(f"тальк-пиксели: acc={acc_t:.3f} AUC={auc_t:.3f} | {t_t:.0f}с")
    joblib.dump(model_t, OUT / "model_talc.joblib")
    metrics["talc_pixels"] = dict(acc=acc_t, auc=auc_t, train_sec=round(t_t, 1),
                                  n=int(len(X)))

    (OUT / "tabular_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2))
    print("saved:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
