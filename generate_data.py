import random
import pandas as pd
from solver import solve

random.seed(42)
N = 10000
rows = []

for _ in range(N):
    # room
    L = round(random.triangular(2.2, 6.5, 3.2), 2)
    W = round(random.triangular(2.2, 5.5, 2.8), 2)
    H = round(random.uniform(2.4, 3.2), 2)

    # door (bottom wall) and window
    door_w = random.choice([0.8, 0.9, 1.0])
    door_off = round(random.uniform(0, L - door_w), 2)
    window_area = round(random.uniform(0.5, 3.0), 2)

    # clearance profile
    clearance = random.choice([0.6, 0.75, 0.9])

    # 0 to 3 existing items
    existing = []
    for _ in range(random.randint(0, 3)):
        ew = round(random.uniform(0.6, 2.0), 2)
        ed = round(random.uniform(0.5, 1.2), 2)
        ex = round(random.uniform(0, max(L - ew, 0)), 2)
        ey = round(random.uniform(0, max(W - ed, 0)), 2)
        existing.append((ex, ey, ew, ed))

    # new furniture
    f_w = round(random.uniform(0.8, 2.6), 2)
    f_d = round(random.uniform(0.5, 1.2), 2)
    f_h = round(random.uniform(0.4, 2.2), 2)

    # teacher decides
    fits, _ = solve(L, W, f_w, f_d, clearance, door_off, door_w, existing)
    existing_area = sum(w * d for _, _, w, d in existing)
    used_ratio = (existing_area + f_w * f_d) / (L * W)
    label = 1 if (fits and used_ratio <= 0.55) else 0

    # save positions of up to 3 existing items (zeros if absent)
    pos = {}
    for i in range(3):
        if i < len(existing):
            px, py, pw, pdp = existing[i]
        else:
            px = py = pw = pdp = 0.0
        pos[f"e{i+1}_x"], pos[f"e{i+1}_y"] = px, py
        pos[f"e{i+1}_w"], pos[f"e{i+1}_d"] = pw, pdp

    rows.append({
        "length_m": L, "width_m": W, "height_m": H,
        "door_off": door_off, "door_w": door_w, "window_area": window_area,
        "clearance": clearance,
        "n_existing": len(existing), "existing_area": round(existing_area, 3),
        "f_w": f_w, "f_d": f_d, "f_h": f_h,
        **pos,
        "label": label,
    })

df = pd.DataFrame(rows)
df.to_csv("data/layout_raw.csv", index=False)
print(df.shape)
print(df["label"].value_counts(normalize=True))