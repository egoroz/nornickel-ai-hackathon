"""Пути и константы проекта."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "shlif"
ARTIFACTS = PROJECT_ROOT / "artifacts"

DIR_PANORAMAS = DATA_ROOT / "Панорамы"
DIR_PART1 = DATA_ROOT / "Фото руд по сортам. ч1"
DIR_PART2 = DATA_ROOT / "Фото руд по сортам. ч2"

# Классы руд (итоговая классификация изображения)
CLS_ORDINARY = "рядовая"          # обычные срастания преобладают
CLS_REFRACTORY = "труднообогатимая"  # тонкие срастания преобладают
CLS_TALC = "оталькованная"        # доля талька > порога

TALC_THRESHOLD = 0.10  # экспертное правило: тальк > 10% площади

# Цвета оверлея (BGR) по ТЗ: зелёный = обычные, красный = тонкие, синий = тальк
COLOR_ORDINARY = (0, 200, 0)
COLOR_FINE = (0, 0, 220)
COLOR_TALC = (255, 80, 0)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
