import numpy as np
import pandas as pd


def position_features(df):
    """Features that use WHERE the existing items are."""
    n = len(df)
    areas = np.zeros((n, 3))
    door_overlap = np.zeros(n)
    min_dist = np.full(n, 99.0)

    door_off = df.door_off.to_numpy()
    door_w = df.door_w.to_numpy()
    dcx = door_off + door_w / 2          # door centre x (door is on the y=0 wall)

    for i in (1, 2, 3):
        x = df[f"e{i}_x"].to_numpy()
        y = df[f"e{i}_y"].to_numpy()
        w = df[f"e{i}_w"].to_numpy()
        d = df[f"e{i}_d"].to_numpy()
        present = w > 0
        areas[:, i - 1] = w * d

        # overlap with the square of floor in front of the door
        ox = np.clip(np.minimum(x + w, door_off + door_w) - np.maximum(x, door_off), 0, None)
        oy = np.clip(np.minimum(y + d, door_w) - np.maximum(y, 0), 0, None)
        door_overlap += np.where(present, ox * oy, 0)

        # distance from door centre to the item's centre
        dist = np.sqrt((x + w / 2 - dcx) ** 2 + (y + d / 2) ** 2)
        min_dist = np.where(present, np.minimum(min_dist, dist), min_dist)

    return {
        "largest_existing_area": areas.max(axis=1),
        "door_zone_overlap": door_overlap,
        "min_dist_to_door": np.where(min_dist == 99.0, 0.0, min_dist),
    }


def make_features(df):
    X = pd.DataFrame(index=df.index)

    # room geometry
    room_area = df.length_m * df.width_m
    X["room_area"] = room_area
    X["perimeter"] = 2 * (df.length_m + df.width_m)
    X["aspect_ratio"] = (df[["length_m", "width_m"]].max(axis=1)
                         / df[["length_m", "width_m"]].min(axis=1))

    # existing items
    X["n_existing"] = df.n_existing
    X["existing_area"] = df.existing_area
    free = (room_area - df.existing_area).clip(lower=0.01)
    X["free_floor_area"] = free
    X["free_floor_ratio"] = free / room_area

    # new furniture
    foot = df.f_w * df.f_d
    X["footprint"] = foot
    X["longest_side"] = df[["f_w", "f_d"]].max(axis=1)
    X["furn_to_room"] = foot / room_area
    X["furn_to_free"] = foot / free

    # fit with clearance
    c = df.clearance
    X["clearance"] = c
    slack_a = (df.length_m - (df.f_w + 2 * c)).combine(
        df.width_m - (df.f_d + 2 * c), min)
    slack_b = (df.length_m - (df.f_d + 2 * c)).combine(
        df.width_m - (df.f_w + 2 * c), min)
    X["best_slack"] = slack_a.combine(slack_b, max)

    # door and window
    X["door_margin"] = df.door_w - df[["f_w", "f_d", "f_h"]].min(axis=1) - 0.05
    X["door_off"] = df.door_off
    X["window_ratio"] = df.window_area / room_area

    # NEW: position features
    for name, values in position_features(df).items():
        X[name] = values

    return X


if __name__ == "__main__":
    df = pd.read_csv("data/layout_clean.csv")
    X = make_features(df)
    y = df["label"]
    print(X.shape)
    print(X.corrwith(y).sort_values())
    X.assign(label=y).to_csv("data/layout_features.csv", index=False)