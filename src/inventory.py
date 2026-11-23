import hashlib
import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm
from config import DATA, IMAGES, OUT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

files = [p for p in IMAGES.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
print(f"Found {len(files)} image files. Building inventory...")

rows = []
for p in tqdm(files):
    rel = p.relative_to(DATA).as_posix()
    parts = p.relative_to(IMAGES).parts
    source = parts[0] if len(parts) > 1 else "unknown"
    label = p.parent.name
    kb = round(p.stat().st_size / 1024, 2)

    try:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            w, h = im.size
            phash = str(imagehash.phash(im.convert("RGB")))
        ok = True
    except Exception:
        w, h, sha, phash, ok = 0, 0, "", "", False

    rows.append({
        "rel": rel,
        "source": source,
        "label": label,
        "kb": kb,
        "w": w,
        "h": h,
        "sha": sha,
        "phash": phash,
        "ok": ok,
    })

inv = pd.DataFrame(rows)
inv.to_csv(OUT / "inventory.csv", index=False)
print("Saved inventory.csv:", inv.shape)