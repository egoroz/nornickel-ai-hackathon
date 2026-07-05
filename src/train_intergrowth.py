"""Срастания: сборка grains.csv и оценка модели.

  python -m src.train_intergrowth --dataset-only   # только grains.csv
  python -m src.train_intergrowth --eval           # оценка модели из artifacts/
  python -m src.train_intergrowth --train          # локальное обучение
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score)

from .config import ARTIFACTS, CLS_ORDINARY, CLS_REFRACTORY
from .data import add_groups, build_index
from .intergrowth import (FEATURE_NAMES, build_grain_dataset,
                          save_model, train_classifier)

OUT = ARTIFACTS / "intergrowth"
VAL_FRAC = 0.2
SEED = 0


def load_expert_labels() -> pd.DataFrame | None:
    """Экспертные метки зёрен: join по source-файлу и пересечению bbox."""
    p = ARTIFACTS / "labeling" / "components" / "labels.csv"
    m = ARTIFACTS / "labeling" / "components" / "manifest.csv"
    if not (p.exists() and m.exists()):
        return None
    lab = pd.read_csv(p, dtype={"id": str})
    lab = lab[lab.label.isin(["обычное", "тонкое"])]
    man = pd.read_csv(m, dtype={"id": str})
    df = lab.merge(man[["id", "source", "comp_id"]], on="id", how="inner")
    return df if len(df) else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rebuild = "--rebuild" in sys.argv
    dataset_only = "--dataset-only" in sys.argv
    do_train = "--train" in sys.argv

    grains_path = OUT / "grains.csv"
    if grains_path.exists() and not rebuild:
        df = pd.read_csv(grains_path)
        print(f"кэш зёрен: {len(df)}")
    else:
        idx = add_groups(build_index())
        idx.to_csv(OUT / "index.csv", index=False)
        df = build_grain_dataset(idx)
        df.to_csv(grains_path, index=False)
        print(f"зёрен: {len(df)} с {df.file.nunique()} фото")
    if dataset_only:
        print("датасет готов; обучение — в DataSphere (datasphere_tabular.ipynb)")
        return

    rng = np.random.RandomState(SEED)
    groups = sorted(df.group.unique())
    val_groups = set(rng.choice(groups, int(len(groups) * VAL_FRAC), replace=False))
    val = df[df.group.isin(val_groups)]
    print(f"train {len(df) - len(val)} / val {len(val)} зёрен; val фото: {val.file.nunique()}")

    if do_train:
        model = train_classifier(df, val_groups, seed=SEED)
        save_model(model, OUT / "model.joblib")
    else:
        from .intergrowth import load_model
        mp = OUT / "model.joblib"
        assert mp.exists(), ("нет artifacts/intergrowth/model.joblib — обучи в "
                             "DataSphere и положи файл сюда")
        model = load_model(mp)

    proba_val = model.predict_proba(val[FEATURE_NAMES].values)[:, 1]
    grain_acc = float(((proba_val >= 0.5) == val.weak_label).mean())

    # изображение: преобладание тонких по площади оболочек vs метка папки
    img_rows = []
    for (path, cls), sub in val.groupby(["path", "cls"]):
        p = model.predict_proba(sub[FEATURE_NAMES].values)[:, 1]
        areas = sub.area.values.astype(float)
        fine_share = float(areas[p >= 0.5].sum() / areas.sum()) if areas.sum() else 0.0
        img_rows.append(dict(path=path, cls=cls, fine_share=fine_share,
                             pred=CLS_REFRACTORY if fine_share > 0.5 else CLS_ORDINARY))
    img = pd.DataFrame(img_rows)
    y_true = (img.cls == CLS_REFRACTORY).astype(int)
    y_pred = (img.pred == CLS_REFRACTORY).astype(int)
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    auc_grain = float(roc_auc_score(val.weak_label, proba_val))
    auc_img = float(roc_auc_score(y_true, img.fine_share))
    print(f"\nзёрна (слабые метки): acc={grain_acc:.3f}, AUC={auc_grain:.3f}")
    print(f"изображения (val): macro-F1={f1:.3f}, AUC={auc_img:.3f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["обычные", "тонкие"]))

    metrics = dict(grain_acc=grain_acc, image_macro_f1=f1,
                   grain_auc=auc_grain, image_auc=auc_img,
                   n_train=int(len(df) - len(val)), n_val=int(len(val)),
                   n_val_images=len(img),
                   confusion=confusion_matrix(y_true, y_pred).tolist(),
                   feature_importance=dict(zip(FEATURE_NAMES,
                                               model.feature_importances_.tolist())))

    # экспертные метки зёрен: честная оценка (маппинг по bbox старых компонент)
    exp = load_expert_labels()
    if exp is not None:
        comp = pd.read_csv(OUT / "components.csv") if (OUT / "components.csv").exists() else None
        if comp is not None:
            exp = exp.merge(
                comp[["path", "comp_id", "x", "y", "w", "h"]].rename(columns={"path": "source"}),
                on=["source", "comp_id"], how="inner")
            hits, correct = 0, 0
            for src, sub in exp.groupby("source"):
                gsub = df[df.path == src]
                if gsub.empty:
                    continue
                p = model.predict_proba(gsub[FEATURE_NAMES].values)[:, 1]
                for _, e in sub.iterrows():
                    cx, cy = e.x + e.w / 2, e.y + e.h / 2
                    inside = gsub[(gsub.x <= cx) & (cx < gsub.x + gsub.w) &
                                  (gsub.y <= cy) & (cy < gsub.y + gsub.h)]
                    if inside.empty:
                        continue
                    gi = inside.index[0]
                    pred_fine = p[gsub.index.get_loc(gi)] >= 0.5
                    hits += 1
                    correct += int(pred_fine == (e.label == "тонкое"))
            if hits:
                metrics["expert_grain_acc"] = correct / hits
                metrics["expert_grain_n"] = hits
                print(f"экспертные метки зёрен: acc={correct / hits:.3f} (n={hits})")

    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    img.to_csv(OUT / "val_images.csv", index=False)
    print("сохранено в", OUT)


if __name__ == "__main__":
    main()
