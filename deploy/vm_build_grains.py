"""grains.csv из датасета: индекс по metadata.csv -> build_grain_dataset.
Выход: ~/shlif/train/grains.csv
"""
import sys
from pathlib import Path

import pandas as pd

HOME = Path.home() / "shlif"
sys.path.insert(0, str(HOME))
from src.intergrowth import build_grain_dataset

DS = HOME / "data" / "dataset"
md = pd.read_csv(DS / "photos" / "metadata.csv")

index = pd.DataFrame(dict(
    path=[str(DS / f) for f in md.file],
    file=[Path(f).name for f in md.file],
    cls=md.cls, part=md.part, sample_id=md.sample_id.astype(str),
    # группа: образец (ч1) + дубликаты по содержимому
    group=[f"{s}|{d}" if str(s).isdigit() else d
           for s, d in zip(md.sample_id, md.dup_group)],
    label_conflict=md.label_conflict,
))
# фото одного образца или одного содержимого должны иметь ОДНУ группу:
# склейка через dup_group -> первый group
first = index.groupby(index.group.str.split("|").str[-1])["group"].transform("first")
index["group"] = first

df = build_grain_dataset(index)
out = HOME / "train" / "grains.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"зёрен: {len(df)} с {df.file.nunique()} фото → {out}")
