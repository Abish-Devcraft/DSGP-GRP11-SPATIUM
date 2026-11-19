from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = DATA / "images"
ABO = DATA / "abo-glb-3d-models"
OUT = DATA / "output"
OUT.mkdir(exist_ok=True)

#make folder names uniform
def canon(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")

#map similar folders
LABEL_MAP = {
    "almirah": "wardrobe",
    "couch": "sofa",
    "drawer": "dresser",
}

CONTEXT_ONLY = {"room_scenes"}

def final_label(label: str) -> str:
    c = canon(label)
    return LABEL_MAP.get(c, c)