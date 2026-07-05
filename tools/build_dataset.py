"""Сборка общего датасета data/dataset/ из data/raw (структуру описывает
генерируемый README.md). Фото хардлинкуются; дубли удаляются физически.

  python tools/build_dataset.py [--tiles-per-pano 40]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (ARTIFACTS, CLS_ORDINARY, CLS_REFRACTORY, CLS_TALC,
                        PROJECT_ROOT)
from src.data import build_index, list_panoramas

DS = PROJECT_ROOT / "data" / "dataset"
CLASS_DIRS = {CLS_ORDINARY: "ryadovaya",
              CLS_REFRACTORY: "trudnoobogatimaya",
              CLS_TALC: "otalkovannaya"}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


TYPICAL_MAGS = {2, 2.5, 4, 5, 10, 20, 25, 40, 50, 63, 100}


def parse_mag(stem: str) -> tuple[str, str]:
    """Увеличение из имени файла → ('10x', stem без токена).

    Встречаются обе записи: «10х» и «х5»/«x 10»; номера образцов бывают
    дробными («2.24 x10») — верим только кандидату с типичным для
    микроскопа значением увеличения.
    """
    cands = []
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*[xх]", stem, re.I):
        cands.append((float(m.group(1).replace(",", ".")), m.span()))
    for m in re.finditer(r"[xх]\s*(\d+(?:[.,]\d+)?)", stem, re.I):
        cands.append((float(m.group(1).replace(",", ".")), m.span()))
    typical = [c for c in cands if c[0] in TYPICAL_MAGS]
    if not typical:
        return "", stem
    val, span = typical[0]
    mag = f"{int(val)}x" if val == int(val) else f"{val}x"
    return mag, stem[:span[0]] + " " + stem[span[1]:]


def norm_name(part: str, stem: str, used: set) -> tuple[str, str]:
    mag, s = parse_mag(stem)
    s = re.sub(r"аншлиф", " ", s, flags=re.I)
    s = re.sub(r"[(),]", " ", s)
    s = re.sub(r"[\s._]+", "-", s.strip(" -_."))
    s = re.sub(r"-+", "-", s).strip("-")
    tag = "ch1" if part == "ч1" else "ch2"
    suffix = f"_{mag}" if mag else ""
    name = f"{tag}_{s}{suffix}.jpg"
    while name in used:  # редкие коллизии после чистки имён
        name = name.replace(".jpg", "x.jpg")
    used.add(name)
    return name, mag


def build_photos() -> pd.DataFrame:
    """Фото + метаданные. Дубли и почти-дубли ФИЗИЧЕСКИ не попадают в датасет:
    в photos/ кладётся один представитель на группу похожести (md5-дубли +
    dhash≤4). Приоритет представителя: с разметкой талька > ч1 > по имени.
    Удалённые перечислены в photos/removed_duplicates.csv (и в EDA).
    """
    import hashlib
    from PIL import Image

    idx = build_index()
    used: set = set()
    rows = []
    for _, r in idx.iterrows():
        stem = Path(r.file).stem
        name, mag = norm_name(r.part, stem, used)
        with Image.open(r.path) as im:
            w, h = im.size
        rows.append(dict(
            file=f"photos/{CLASS_DIRS[r.cls]}/{name}", cls=r.cls, part=r.part,
            sample_id=r.sample_id, magnification=mag,
            camera=("neutral_4160" if w >= 4000 else "yellow_2272"),
            width=w, height=h,
            original=r.path, annot_original=r.annot_path or "",
        ))
    df = pd.DataFrame(rows)

    # группы похожести: md5 + dhash<=4
    def _md5(path):
        hh = hashlib.md5()
        with open(path, "rb") as f:
            for ch in iter(lambda: f.read(1 << 20), b""):
                hh.update(ch)
        return hh.hexdigest()

    def _dhash(path, size=8):
        with Image.open(path) as im:
            im.draft("L", (size * 8, size * 8))
            g = np.asarray(im.convert("L").resize((size + 1, size)), dtype=int)
        return (g[:, 1:] > g[:, :-1]).ravel()

    md5s = [_md5(p) for p in df.original]
    hashes = np.stack([_dhash(p) for p in df.original])
    parent = list(range(len(df)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    by_md5: dict = {}
    for i, m in enumerate(md5s):
        if m in by_md5:
            union(i, by_md5[m])
        else:
            by_md5[m] = i
    for i in range(len(df)):
        d = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        for j in np.flatnonzero(d <= 4):
            union(i, i + 1 + j)
    df["dup_group"] = [f"d{find(i)}" for i in range(len(df))]
    span = df.groupby("dup_group")["cls"].nunique()
    df["label_conflict"] = df.dup_group.map(span) > 1

    # представитель группы: с разметкой талька > ч1 > имя
    df["_prio"] = ((df.annot_original != "").astype(int) * 2
                   + (df.part == "ч1").astype(int))
    df = df.sort_values(["dup_group", "_prio", "file"],
                        ascending=[True, False, True])
    kept = df.drop_duplicates("dup_group").copy()
    removed = df[~df.index.isin(kept.index)].copy()

    for _, r in kept.iterrows():
        _link(Path(r.original), DS / r.file)
    kept["removed_dups"] = kept.dup_group.map(
        removed.groupby("dup_group").apply(
            lambda g: ";".join(Path(x).name for x in g.file)))
    kept = kept.drop(columns=["_prio"]).sort_values("file")
    kept.to_csv(DS / "photos" / "metadata.csv", index=False)
    removed[["dup_group", "file", "cls", "part", "original"]].to_csv(
        DS / "photos" / "removed_duplicates.csv", index=False)

    # подчистить файлы, выпавшие из metadata (после смены логики)
    keep_set = set(kept.file)
    for f in DS.glob("photos/*/*.jpg"):
        if str(f.relative_to(DS)) not in keep_set:
            f.unlink()

    print("увеличения:", kept.magnification.replace("", "нет").value_counts().to_dict())
    print(f"уникальных фото в датасете: {len(kept)}; удалено похожих: {len(removed)} "
          f"(список: photos/removed_duplicates.csv); "
          f"конфликтных групп: {int(kept.label_conflict.sum())}")
    return kept


def build_talc_annotation(mapping: pd.DataFrame) -> None:
    """labelme-проект по ч1-оталькованным: фон = копия с линиями заказчика."""
    d = DS / "talc_annotation"
    d.mkdir(parents=True, exist_ok=True)
    ann = mapping[mapping.annot_original != ""]
    for _, r in ann.iterrows():
        name = Path(r.file).name
        _link(Path(r.annot_original), d / name)
        jp = d / f"{Path(name).stem}.json"
        if not jp.exists():
            im = cv2.imread(str(d / name))
            jp.write_text(json.dumps(dict(
                version="5.4.1", flags={}, shapes=[], imagePath=name,
                imageData=None, imageHeight=im.shape[0],
                imageWidth=im.shape[1]), ensure_ascii=False), encoding="utf-8")
    # миграция уже сделанных разметок из старого проекта
    old = ARTIFACTS / "labeling" / "talc_labelme_scratch"
    if old.exists():
        by_orig_stem = {Path(r.original).stem: Path(r.file).name
                        for _, r in ann.iterrows()}
        for op in old.glob("*.json"):
            doc = json.loads(op.read_text(encoding="utf-8"))
            if not doc.get("shapes"):
                continue
            new_img = by_orig_stem.get(op.stem)
            if not new_img:
                continue
            doc["imagePath"] = new_img
            (d / f"{Path(new_img).stem}.json").write_text(
                json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            print(f"  перенёс разметку: {op.name} → {Path(new_img).stem}.json")
    print(f"talc_annotation: {len(ann)} фото")


def build_panorama_tiles(per_pano: int, tile: int = 2048, seed: int = 0) -> None:
    import pyvips
    rng = np.random.RandomState(seed)
    d = DS / "panorama_tiles"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_md5: dict = {}
    for p in list_panoramas():
        import hashlib
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen_md5:
            print(f"  {p.name} — дубликат {seen_md5[h]}, тайлы не режем")
            continue
        seen_md5[h] = p.name
        img = pyvips.Image.new_from_file(str(p))
        W, H = img.width, img.height
        v = pyvips.Image.thumbnail(str(p), max(W // 16, 1))
        thumb = np.ndarray(buffer=v.write_to_memory(), dtype=np.uint8,
                           shape=[v.height, v.width, v.bands])[:, :, :3]
        gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
        bright_thr = np.percentile(gray, 97)
        cands = []
        for y0 in range(0, H - tile + 1, tile):
            for x0 in range(0, W - tile + 1, tile):
                sub = gray[y0 // 16:(y0 + tile) // 16, x0 // 16:(x0 + tile) // 16]
                if float((sub > bright_thr).mean()) > 0.002:
                    cands.append((x0, y0))
        rng.shuffle(cands)
        for x0, y0 in cands[:per_pano]:
            tid = f"pano{p.stem}_x{x0}_y{y0}"
            outp = d / f"{tid}.jpg"
            if not outp.exists():
                # сырой кроп 1:1, без правок яркости/контраста
                crop = img.crop(x0, y0, tile, tile)
                arr = np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8,
                                 shape=[crop.height, crop.width, crop.bands])[:, :, :3]
                cv2.imwrite(str(outp), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(dict(id=tid, image_path=f"../panorama_tiles/{tid}.jpg",
                             pano=p.name, x0=x0, y0=y0, tile=tile,
                             norm="raw"))
    pd.DataFrame(rows).to_csv(d / "manifest.csv", index=False)

    # тайлы, выпавшие из манифеста (панорамы-дубликаты и т.п.) — убрать
    keep = {r["id"] + ".jpg" for r in rows}
    for f in d.glob("*.jpg"):
        if f.name not in keep:
            f.unlink()

    # манифест для классификации тайлов (label_app)
    td = DS / "tile_labeling"
    td.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(td / "manifest.csv", index=False)
    print(f"panorama_tiles: {len(rows)} тайлов")


def build_grain_labeling(n_crops: int = 400, seed: int = 0) -> None:
    """Кропы вкрапленников из artifacts/intergrowth/grains.csv (зёрна v2)."""
    grains_csv = ARTIFACTS / "intergrowth" / "grains.csv"
    d = DS / "grain_labeling"
    (d / "crops").mkdir(parents=True, exist_ok=True)
    if not grains_csv.exists():
        print("grain_labeling: нет grains.csv "
              "(python -m src.train_intergrowth --dataset-only)")
        return
    df = pd.read_csv(grains_csv)
    # один представитель на группу дублей содержимого (не кликать одно дважды)
    md_meta = pd.read_csv(DS / "photos" / "metadata.csv")
    orig2dup = dict(zip(md_meta.original, md_meta.dup_group))
    df["dup_group"] = df.path.map(orig2dup)
    keep_path = df.groupby("dup_group").path.transform("first")
    df = df[df.path == keep_path]
    rng = np.random.RandomState(seed)
    picks = []
    for _, sub in df.groupby(["cls", "part"]):
        k = min(n_crops // df.groupby(["cls", "part"]).ngroups, len(sub))
        w = np.sqrt(sub.area.values)
        picks.append(sub.sample(n=k, weights=w / w.sum(), random_state=rng))
    sel = pd.concat(picks)
    rows = []
    for path, sub in sel.groupby("path"):
        img = cv2.imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        for _, r in sub.iterrows():
            pad = 60
            x0, y0 = max(int(r.x) - pad, 0), max(int(r.y) - pad, 0)
            x1 = min(int(r.x + r.w) + pad, W)
            y1 = min(int(r.y + r.h) + pad, H)
            crop = img[y0:y1, x0:x1].copy()
            cv2.rectangle(crop, (int(r.x) - x0, int(r.y) - y0),
                          (int(r.x + r.w) - x0, int(r.y + r.h) - y0),
                          (0, 0, 255), 3)
            if max(crop.shape[:2]) > 900:
                s = 900 / max(crop.shape[:2])
                crop = cv2.resize(crop, (int(crop.shape[1] * s),
                                         int(crop.shape[0] * s)))
            cid = f"{Path(path).stem}__{int(r.grain_id)}".replace(" ", "-")
            cv2.imwrite(str(d / "crops" / f"{cid}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append(dict(id=cid, image_path=f"crops/{cid}.jpg",
                             source=path, grain_id=int(r.grain_id)))
    pd.DataFrame(rows).to_csv(d / "manifest.csv", index=False)
    print(f"grain_labeling: {len(rows)} кропов (из зёрен v2)")


def make_self_contained() -> None:
    """Кладём в датасет всё для разметки без основного репозитория."""
    shutil.copy(PROJECT_ROOT / "tools" / "label_app.py", DS / "label_app.py")
    (DS / "requirements-labeling.txt").write_text(
        "streamlit\nplotly\npandas\npillow\nlabelme\n", encoding="utf-8")


README = """# Общий датасет команды — «Скажи мне, кто твой шлиф»

**Как размечать — читай [GUIDE.md](GUIDE.md)** (единственная инструкция:
установка на Windows/Ubuntu, синхронизация, критерии классов и полигонов,
все команды). Этот файл — только про структуру папки.

## Что где лежит

| Папка/файл | Что это |
|---|---|
| `GUIDE.md` | инструкция разметчика — начни с неё |
| `label_app.py`, `requirements-labeling.txt` | приложение кнопочной разметки и его зависимости |
| `photos/<класс>/` | фото руд (ч1+ч2), уникальные (дубли удалены), нормализованные имена: `ch1_2550374-2_10x.jpg` = часть 1, образец 2550374 шлиф 2, увеличение 10x |
| `photos/metadata.csv` | метаданные фото: класс, часть, образец, увеличение, камера, размеры, исходник, dup_group/label_conflict; расширяема |
| `photos/removed_duplicates.csv` | список удалённых дублей/почти-дублей (что и вместо чего) |
| `panorama_tiles/` | тайлы панорам 2048×2048 — сырые кропы 1:1 (координаты в `manifest.csv`; яркость крути только ползунками) |
| `talc_annotation/` | labelme-проект талька по ч1 (фон = копии с синими линиями заказчика) |
| `tile_labeling/` | задание А: `manifest.csv` + результат `labels.csv` |
| `grain_labeling/` | задание Б: кропы зёрен + `manifest.csv` + результат `labels.csv` |
| `talc_masks/` | (генерируется скриптами репозитория) маски из labelme-разметки |

## Куда пишется разметка

`tile_labeling/labels.csv`, `grain_labeling/labels.csv`, `*.json` в
`talc_annotation/` и `panorama_tiles/`. Если папка синхронизируется
(Google Drive) — больше ничего делать не нужно.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-per-pano", type=int, default=40)
    a = ap.parse_args()
    DS.mkdir(parents=True, exist_ok=True)
    mapping = build_photos()
    print(f"photos: {len(mapping)} файлов")
    build_talc_annotation(mapping)
    build_panorama_tiles(a.tiles_per_pano)
    build_grain_labeling()
    make_self_contained()
    (DS / "README.md").write_text(README, encoding="utf-8")
    print(f"\nготово: {DS}\nдокументация: {DS / 'README.md'}")
