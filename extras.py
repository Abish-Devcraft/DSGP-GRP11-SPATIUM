def doorway_test(f_w, f_d, f_h, door_w, door_h=2.1, margin=0.05):
    """Carry the item with its longest side along the passage.
    The two smaller sides must clear the door width and height."""
    a, b, _ = sorted([f_w, f_d, f_h])        # two smallest sides
    ok = (a <= door_w - margin and b <= door_h - margin) or \
         (b <= door_w - margin and a <= door_h - margin)
    return {"passes": ok, "smallest_two": (a, b),
            "reason": None if ok else
            f"item cross-section {a:.2f} x {b:.2f} m does not clear a "
            f"{door_w:.2f} x {door_h:.2f} m door"}


def comfort_score(length_m, width_m, windows):
    """windows = list of (wall, area_m2), wall in N/S/E/W. Returns 0-100."""
    floor = length_m * width_m
    score = 50.0
    total_glass = sum(a for _, a in windows)

    # daylight: window-to-floor ratio between 0.10 and 0.25
    if 0.10 <= total_glass / floor <= 0.25:
        score += 20

    # cross-ventilation
    walls = {w for w, _ in windows}
    if {"N", "S"} <= walls or {"E", "W"} <= walls:
        score += 20                          # opposite walls
    elif len(walls) >= 2:
        score += 10                          # two walls (adjacent)

    # west sun penalty
    west = sum(a for w, a in windows if w == "W")
    score -= 60 * west / floor

    return round(max(0, min(100, score)), 1)


if __name__ == "__main__":
    print(doorway_test(2.0, 0.9, 0.85, 0.9))       # sofa, 0.9 m door
    print(doorway_test(2.0, 1.0, 2.0, 0.8))        # wardrobe, 0.8 m door
    print(comfort_score(4.0, 3.5, [("N", 1.5), ("S", 1.2)]))   # good room
    print(comfort_score(4.0, 3.5, [("W", 3.0)]))               # bad room