"""
Landmarks -> model input. normalize + render + augment live here and NOWHERE else.

Training (gestureflow.train), live inference (gestureflow.core.inference) and the preview all
import from this module, so they can never disagree about what the CNN sees. Any change
to its output must be deliberate: tests/test_golden.py pins it byte for byte, and the
checkpoint stores config() so a model trained under different settings refuses to load.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------- config
CANVAS = 128        # output image is CANVAS x CANVAS x 3
MARGIN = 0.12       # fraction of the canvas left empty at the edges
BONE = 2            # line thickness
JOINT = 3           # circle radius

ALIGN_ROTATION = False


def config():
    """The settings a checkpoint was trained under; inference refuses a mismatch."""
    return {"canvas": CANVAS, "margin": MARGIN, "align_rotation": ALIGN_ROTATION}

# The 21 bones. Each pair is (start point, end point).
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                    # palm base
]

# BGR, because that is the order OpenCV uses.
COLOR = {}
for _i in range(1, 5):
    COLOR[_i] = (80, 80, 255)       # thumb   - red
for _i in range(5, 9):
    COLOR[_i] = (80, 220, 255)      # index   - orange
for _i in range(9, 13):
    COLOR[_i] = (80, 255, 120)      # middle  - green
for _i in range(13, 17):
    COLOR[_i] = (255, 200, 80)      # ring    - blue
for _i in range(17, 21):
    COLOR[_i] = (255, 110, 200)     # pinky   - magenta
COLOR[0] = (200, 200, 200)          # wrist   - grey


# ------------------------------------------------------------ normalize
def normalize(points):
    pts = np.asarray(points, dtype=np.float32)[:, :2].copy()

    wrist = pts[0].copy()
    pts = pts - wrist

    if ALIGN_ROTATION:
        vx, vy = pts[9]
        theta = (-np.pi / 2.0) - np.arctan2(vy, vx)
        c, s = np.cos(theta), np.sin(theta)
        pts = pts @ np.array([[c, -s], [s, c]], dtype=np.float32).T

    lengths = np.linalg.norm(pts, axis=1)
    scale = lengths.max()
    if scale < 1e-6:
        scale = 1e-6
    pts = pts / scale

    return pts.astype(np.float32)


# --------------------------------------------------------------- render
def render(points):
    img = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)

    half = CANVAS / 2.0
    scale = half * (1.0 - MARGIN)
    xy = np.asarray(points, dtype=np.float32) * scale + half
    xy = np.round(xy).astype(np.int32)

    for a, b in CONNECTIONS:
        cv2.line(img, tuple(xy[a]), tuple(xy[b]), COLOR[b], BONE)

    for i in range(21):
        cv2.circle(img, tuple(xy[i]), JOINT, COLOR[i], -1)

    return img


def to_input(img):
    """(H, W, 3) uint8 skeleton -> (3, H, W) float32 in 0..1, the layout the CNN takes."""
    return img.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0)

# -------------------------------------------------------------- augment


def augment(points, rng):
    pts = np.asarray(points, dtype=np.float32).copy()

    # rotation - small only, since orientation carries meaning here
    ang = np.deg2rad(rng.uniform(-15.0, 15.0))
    c, s = np.cos(ang), np.sin(ang)
    pts = pts @ np.array([[c, -s], [s, c]], dtype=np.float32).T

    # uniform scale
    pts *= float(rng.uniform(0.85, 1.15))

    # mild aspect jitter, stands in for camera perspective
    pts[:, 0] *= float(rng.uniform(0.92, 1.08))
    pts[:, 1] *= float(rng.uniform(0.92, 1.08))

    # per-joint noise, stands in for landmark detector wobble
    pts += rng.normal(0.0, 0.015, pts.shape).astype(np.float32)

    # translation
    pts += rng.uniform(-0.05, 0.05, (1, 2)).astype(np.float32)

    return pts.astype(np.float32)
