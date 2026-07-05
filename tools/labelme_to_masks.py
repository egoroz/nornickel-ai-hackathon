"""Импорт labelme-разметки в бинарные маски.

t/talc -> talc_masks/<stem>.png, s/sulfide -> sulfide_masks/;
флаг no_talc (или reviewed без t-полигонов) -> нулевая маска-негатив.
Авто-фигуры (g, t_auto) в маски не идут.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ARTIFACTS, PROJECT_ROOT

DS = PROJECT_ROOT / "data" / "dataset"
SRCS = [DS / "talc_annotation",      # ч1 с подсказками-линиями (приоритет)
        DS / "panorama_tiles",        # тайлы панорам
        DS / "photos" / "otalkovannaya",   # тёмные ч2-фото без разметки
        DS / "photos" / "ryadovaya",
        DS / "photos" / "trudnoobogatimaya"]
OUT_T = DS / "talc_masks"
OUT_S = DS / "sulfide_masks"
QC = ARTIFACTS / "qc" / "talc_expert"

TALC_LABELS = {"t", "talc"}
SULF_LABELS = {"s", "sulfide"}
AUTO_LABELS = {"g", "t_auto"}  # предразметка, в маски не идёт


def render(doc: dict, labels: set) -> np.ndarray | None:
    h, w = doc["imageHeight"], doc["imageWidth"]
    shapes = [s for s in doc["shapes"]
              if s.get("label", "").lower() in labels
              and s.get("shape_type") == "polygon"]
    if not shapes:
        return None
    mask = np.zeros((h, w), np.uint8)
    for sh in shapes:
        cv2.fillPoly(mask, [np.array(sh["points"], dtype=np.int32)], 1)
    return mask


if __name__ == "__main__":
    QC.mkdir(parents=True, exist_ok=True)
    n_t = n_s = 0
    seen: set = set()
    for src in SRCS:
        if not src.exists():
            continue
        for jp in sorted(src.glob("*.json")):
            if jp.stem in seen:
                continue
            doc = json.loads(jp.read_text(encoding="utf-8"))
            flags = doc.get("flags") or {}
            no_talc = any(v for k, v in flags.items()
                          if k.lower() in ("no_talc", "нет_талька", "notalc"))
            reviewed = any(v for k, v in flags.items()
                           if k.lower() == "reviewed")
            human_shapes = [s for s in doc.get("shapes", [])
                            if s.get("label", "").lower() not in AUTO_LABELS]
            if not human_shapes and not no_talc and not reviewed:
                continue  # не размечено (а не «пусто»)
            seen.add(jp.stem)
            mt = render(doc, TALC_LABELS)
            if mt is None and (no_talc or reviewed):
                # «талька нет» — валидная нулевая маска
                h, w = doc["imageHeight"], doc["imageWidth"]
                mt = np.zeros((h, w), np.uint8)
            ms = render(doc, SULF_LABELS)
            img = cv2.imread(str(src / doc["imagePath"]))
            if mt is not None:
                OUT_T.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(OUT_T / f"{jp.stem}.png"), mt * 255)
                n_t += 1
                if img is not None:
                    ov = img.copy()
                    ov[mt > 0] = (255, 120, 60)
                    vis = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
                    cv2.putText(vis, f"{jp.stem} talc={mt.mean()*100:.1f}%",
                                (10, 40), 0, 1.2, (255, 255, 255), 2)
                    cv2.imwrite(str(QC / f"{jp.stem}.jpg"),
                                cv2.resize(vis, (img.shape[1] // 2,
                                                 img.shape[0] // 2)),
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ms is not None:
                OUT_S.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(OUT_S / f"{jp.stem}.png"), ms * 255)
                n_s += 1
    print(f"тальк: {n_t} масок → {OUT_T}")
    if n_s:
        print(f"сульфиды: {n_s} масок → {OUT_S}")
    print(f"QC-оверлеи: {QC}")
