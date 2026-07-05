"""Обучение U-Net талька на VM команды (Tesla T4), зеркало
notebooks/datasphere_talc_unet.ipynb.

БЕЗ бандлов: пары (фото, экспертная маска) берутся прямо из датасета —
data/dataset/talc_masks + поиск фото по stem. Аугментации закрывают
требования кейса: свет/контраст/гамма, блюр, масштаб 80–500 мкм, повороты.

Запуск на VM:  ~/shlif/venv/bin/python ~/shlif/deploy/vm_train_unet.py
Выход:  ~/shlif/out/{talc_unet.pt, unet_metrics.json}
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HOME = Path.home() / "shlif"
DSET = HOME / "data" / "dataset"
OUT = HOME / "out"
CROP, EPOCHS, SEED = 512, 40, 0

IMG_DIRS = [DSET / "talc_annotation", DSET / "photos" / "otalkovannaya",
            DSET / "photos" / "ryadovaya", DSET / "photos" / "trudnoobogatimaya"]


def find_img(stem):
    for d in IMG_DIRS:
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def group_of(stem):
    parts = stem.split("_")
    if len(parts) > 1 and parts[1].split("-")[0].isdigit():
        return parts[1].split("-")[0]
    return stem


def main():
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    pairs = {}
    for mp in sorted((DSET / "talc_masks").glob("*.png")):
        ip = find_img(mp.stem)
        if ip is None:
            print("маска без фото (удалено?):", mp.name)
            continue
        pairs[mp.stem] = (str(ip), str(mp))
    stems = sorted(pairs)
    groups = {s: group_of(s) for s in stems}
    uniq = sorted(set(groups.values()))
    rng = np.random.RandomState(SEED)
    val_groups = set(rng.choice(uniq, max(int(len(uniq) * 0.25), 3), replace=False))
    train_stems = [s for s in stems if groups[s] not in val_groups]
    val_stems = [s for s in stems if groups[s] in val_groups]
    print(f"train {len(train_stems)} / val {len(val_stems)} фото "
          f"(групп {len(uniq) - len(val_groups)}/{len(val_groups)})")

    import albumentations as A
    train_aug = A.Compose([
        A.RandomScale(scale_limit=(-0.6, 1.0), p=0.7),   # масштаб ~0.4x–2x
        A.PadIfNeeded(CROP, CROP, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomCrop(CROP, CROP),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.RandomBrightnessContrast(0.35, 0.35, p=0.9),
        A.RandomGamma((60, 160), p=0.5),
        A.HueSaturationValue(15, 40, 30, p=0.8),
        A.RGBShift(25, 25, 25, p=0.5),
        A.GaussianBlur(blur_limit=(3, 9), p=0.3),
        A.GaussNoise(p=0.3),
    ])

    class DS(Dataset):
        def __init__(self, stems, aug=None, crops=16):
            self.items = [(s, i) for s in stems
                          for i in range(crops if aug else 1)]
            self.aug, self.cache = aug, {}

        def _load(self, s):
            if s not in self.cache:
                ip, mp = pairs[s]
                img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
                m = (cv2.imread(mp, 0) > 127).astype(np.float32)
                self.cache[s] = (img, m)
            return self.cache[s]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            s, _ = self.items[i]
            img, m = self._load(s)
            if self.aug:
                r = self.aug(image=img, mask=m)
                img, m = r["image"], r["mask"]
            else:
                H, W = (img.shape[0] // 32) * 32, (img.shape[1] // 32) * 32
                img, m = img[:H, :W], m[:H, :W]
            x = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.
            return x, torch.from_numpy(np.ascontiguousarray(m)).float()

    import segmentation_models_pytorch as smp
    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1).to(device)
    dice = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    train_dl = DataLoader(DS(train_stems, train_aug), batch_size=8,
                          shuffle=True, num_workers=2)
    val_ds = DS(val_stems)

    def val_mae():
        model.eval()
        errs, ious = [], []
        with torch.no_grad():
            for x, m in DataLoader(val_ds, batch_size=1):
                p = torch.sigmoid(model(x.to(device)))[0, 0].cpu().numpy()
                gt = m[0].numpy()
                errs.append(abs(p.mean() - gt.mean()))
                pb, gb = p > 0.5, gt > 0.5
                u = (pb | gb).sum()
                ious.append((pb & gb).sum() / u if u else 1.0)
        model.train()
        return float(np.mean(errs)), float(np.median(ious))

    best, log = 1e9, []
    t0 = time.time()
    for ep in range(EPOCHS):
        tot = 0.0
        for x, m in train_dl:
            x, m = x.to(device), m.to(device).unsqueeze(1)
            out = model(x)
            loss = 0.5 * bce(out, m) + 0.5 * dice(out, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(x)
        sched.step()
        mae, iou = val_mae()
        star = ""
        if mae < best:
            best = mae
            torch.save(model.state_dict(), OUT / "talc_unet.pt")
            star = " <-- saved"
        line = (f"ep {ep:02d} loss {tot/len(train_dl.dataset):.3f} "
                f"MAE {mae*100:.2f}пп IoU {iou:.2f}{star}")
        print(line, flush=True)
        log.append(dict(epoch=ep, loss=tot / len(train_dl.dataset),
                        mae=mae, iou=iou))
    (OUT / "unet_metrics.json").write_text(json.dumps(dict(
        best_mae=best, epochs=EPOCHS, minutes=round((time.time() - t0) / 60, 1),
        log=log), ensure_ascii=False, indent=2))
    print(f"best MAE {best*100:.2f} пп за {(time.time()-t0)/60:.0f} мин")


if __name__ == "__main__":
    main()
