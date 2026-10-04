"""
Mean skeleton per letter, pooled across sessions.

Run: python -m gestureflow.preview
"""

import argparse
from contextlib import contextmanager

import cv2
import numpy as np
import pandas as pd

from gestureflow.core import preprocessing
from gestureflow.paths import LANDMARKS_CSV, PREVIEW_IMAGE

COLS = [f"{a}{i}" for i in range(21) for a in "xy"]
BG = 0
PAD = 22


@contextmanager
def big_canvas(size):
    """Render larger for a crisp preview. Geometry identical, restored after."""
    old = (preprocessing.CANVAS, preprocessing.BONE, preprocessing.JOINT)
    k = size / old[0]
    preprocessing.CANVAS = size
    preprocessing.BONE = max(1, round(old[1] * k))
    preprocessing.JOINT = max(1, round(old[2] * k))
    try:
        yield
    finally:
        preprocessing.CANVAS, preprocessing.BONE, preprocessing.JOINT = old


def mean_shape(coords):
    """Normalize every sample, average the coordinates, return (mean, spread)."""
    normed = np.stack([preprocessing.normalize(c) for c in coords])
    mean = normed.mean(axis=0)
    spread = float(np.linalg.norm(normed - mean, axis=2).mean())
    return mean, spread


def build_grid(cells, cols):
    h, w = cells[0].shape[:2]
    while len(cells) % cols:
        cells.append(np.full((h, w, 3), BG, np.uint8))

    rows = []
    for i in range(0, len(cells), cols):
        row = [np.full((h, PAD, 3), BG, np.uint8)]
        for c in cells[i:i + cols]:
            row += [c, np.full((h, PAD, 3), BG, np.uint8)]
        rows.append(np.hstack(row))

    bar = np.full((PAD, rows[0].shape[1], 3), BG, np.uint8)
    out = [bar]
    for r in rows:
        out += [r, bar]
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(LANDMARKS_CSV))
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="default: every session in the csv")
    ap.add_argument("--out", default=str(PREVIEW_IMAGE))
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--size", type=int, default=190)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    present = sorted(df["session"].unique())
    sessions = args.sessions or present

    missing = [s for s in sessions if s not in present]
    if missing:
        raise SystemExit(f"sessions not in csv: {missing}\nfound: {present}")

    df = df[df["session"].isin(sessions)]
    labels = sorted(df["label"].unique())
    print(f"pooling sessions: {', '.join(sessions)}\n")

    cells = []
    stats = []

    with big_canvas(args.size):
        for L in labels:
            rows = df[df["label"] == L]
            coords = rows[COLS].to_numpy(np.float32).reshape(-1, 21, 2)

            mean, spread = mean_shape(coords)
            tile = preprocessing.render(mean)
            per_session = {}
            for s in sessions:
                sub = rows[rows["session"] == s]
                if len(sub):
                    c = sub[COLS].to_numpy(np.float32).reshape(-1, 21, 2)
                    per_session[s] = mean_shape(c)[0]

            gap = None
            if len(per_session) == 2:
                a, b = per_session.values()
                gap = float(np.linalg.norm(a - b, axis=1).mean())

            cv2.putText(tile, str(L), (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(tile)
            stats.append((L, len(coords), spread, gap))

    # ------------------------------------------------------------ report
    has_gap = any(g is not None for _, _, _, g in stats)
    print(f"{'letter':>6}  {'n':>4}  {'spread':>8}" +
          ("     gap" if has_gap else ""))
    print("-" * (30 if has_gap else 22))
    for L, n, spread, gap in stats:
        line = f"{L:>6}  {n:>4}  {spread:>8.4f}"
        if has_gap:
            line += f"  {gap:>6.4f}" if gap is not None else "       -"
        print(line)

    worst = sorted(stats, key=lambda t: -t[2])[:5]
    print("\nhighest spread - least consistent gestures:")
    for L, _n, spread, _ in worst:
        print(f"  {L}: {spread:.4f}")

    if has_gap:
        gaps = [(g, L) for L, _, _, g in stats if g is not None]
        gaps.sort()
        print("\nsmallest gaps - those two sessions barely differ:")
        for g, L in gaps[:5]:
            print(f"  {L}: {g:.4f}")
        print(f"median gap: {np.median([g for g, _ in gaps]):.4f}")
        print("Below about 0.03 and the second session is close to a copy.")

    grid = build_grid(cells, args.cols)
    cv2.imwrite(args.out, grid)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
