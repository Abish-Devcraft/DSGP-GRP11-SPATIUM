import numpy as np

STEP = 0.2  # grid size in metres


def overlap(a, b):
    """Do two rectangles (x, y, width, depth) overlap?"""
    ax, ay, aw, ad = a
    bx, by, bw, bd = b
    return ax < bx + bw and bx < ax + aw and ay < by + bd and by < ay + ad


def solve(L, W, item_w, item_d, clearance, door_off, door_w, existing):
    """
    L, W       : room length (x) and width (y) in metres
    item_w/d   : size of the new furniture
    clearance  : walking space needed around existing items (0.6 / 0.75 / 0.9)
    door_off   : where the door starts along the bottom wall
    door_w     : door width
    existing   : list of (x, y, w, d) for items already in the room
    Returns (True, placement) or (False, None).
    """
    # the floor area in front of the door must stay free
    door_zone = (door_off, 0, door_w, door_w)

    # try the item as given, then turned 90 degrees
    for w, d in [(item_w, item_d), (item_d, item_w)]:
        for x in np.arange(0, L - w + 1e-9, STEP):
            for y in np.arange(0, W - d + 1e-9, STEP):
                box = (x, y, w, d)

                if overlap(box, door_zone):
                    continue  # blocks the door

                blocked = False
                for ex, ey, ew, ed in existing:
                    # grow the existing item by the clearance on all sides
                    grown = (ex - clearance, ey - clearance,
                             ew + 2 * clearance, ed + 2 * clearance)
                    if overlap(box, grown):
                        blocked = True
                        break
                if not blocked:
                    return True, (round(float(x), 2), round(float(y), 2), w, d)
    return False, None


if __name__ == "__main__":
    # 1. big room, small sofa -> should fit
    print(solve(4.2, 3.4, 2.0, 0.9, 0.6, 0.4, 0.9, []))
    # 2. sofa 2.5 m long in a 2.2 m room -> must NOT fit
    print(solve(2.2, 2.2, 2.5, 0.9, 0.6, 0.4, 0.9, []))
    # 3. small room, big existing wardrobe -> no space left
    print(solve(2.2, 2.2, 1.2, 0.6, 0.6, 0.4, 0.9, [(0, 0.9, 2.2, 1.3)]))