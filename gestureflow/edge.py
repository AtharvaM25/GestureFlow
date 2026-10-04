"""
Reference for Edge mode's in-browser preprocessing (frontend/src/lib/edge.ts).

The browser can't run OpenCV, so it draws the skeleton with a simpler rasterizer: bones are
every pixel within EDGE_BONE_RADIUS of the bone segment, joints are filled discs. This file is
that rasterizer in Python, written to the same float32/float64 arithmetic as the TypeScript,
so the two can be checked against each other on every sample:

    python -m gestureflow.edge        # measure agreement with the server, write the fixture

writes docs/edge_parity.json (how often Edge and Server mode give the same letter) and
frontend/src/lib/edge-fixture.json (the expected render of every sample, which
`pnpm test:edge` checks the TypeScript against).
"""

import hashlib
import json

import numpy as np

from gestureflow.core import preprocessing as P

EDGE_BONE_RADIUS = 1.5   # chosen to best match cv2.line(thickness=2); see docs/edge_parity.json
EDGE_JOINT_RADIUS = 3    # same as preprocessing.JOINT


def pixel_points(points):
    """Normalized points -> integer canvas coordinates, exactly as preprocessing.render does."""
    half = P.CANVAS / 2.0
    xy = np.asarray(points, dtype=np.float32) * (half * (1.0 - P.MARGIN)) + half
    return np.round(xy).astype(np.int32)


def render_edge(points):
    c = P.CANVAS
    img = np.zeros((c, c, 3), np.uint8)
    xy = pixel_points(points)
    r2 = EDGE_BONE_RADIUS * EDGE_BONE_RADIUS
    pad = int(np.ceil(EDGE_BONE_RADIUS))
    for a, b in P.CONNECTIONS:
        x0, y0 = int(xy[a][0]), int(xy[a][1])
        x1, y1 = int(xy[b][0]), int(xy[b][1])
        dx, dy = x1 - x0, y1 - y0
        length2 = dx * dx + dy * dy
        xa, xb = max(0, min(x0, x1) - pad), min(c - 1, max(x0, x1) + pad)
        ya, yb = max(0, min(y0, y1) - pad), min(c - 1, max(y0, y1) + pad)
        if xa > xb or ya > yb:
            continue
        py, px = np.mgrid[ya:yb + 1, xa:xb + 1].astype(np.float64)
        if length2 > 0:
            t = np.minimum(1.0, np.maximum(0.0, ((px - x0) * dx + (py - y0) * dy) / length2))
        else:
            t = np.zeros_like(px)
        ex = px - (x0 + t * dx)
        ey = py - (y0 + t * dy)
        img[ya:yb + 1, xa:xb + 1][ex * ex + ey * ey <= r2] = P.COLOR[b]
    jr = EDGE_JOINT_RADIUS
    for i in range(21):
        x, y = int(xy[i][0]), int(xy[i][1])
        xa, xb = max(0, x - jr), min(c - 1, x + jr)
        ya, yb = max(0, y - jr), min(c - 1, y + jr)
        if xa > xb or ya > yb:
            continue
        py, px = np.mgrid[ya:yb + 1, xa:xb + 1]
        img[ya:yb + 1, xa:xb + 1][(px - x) ** 2 + (py - y) ** 2 <= jr * jr] = P.COLOR[i]
    return img


def render_hash(img):
    return hashlib.sha256(img.tobytes()).hexdigest()


def temporal_fixture(n_streams=100, seed=0):
    """Random signing streams (held letters with flicker, low confidence and lost hands)
    and what LetterCommitter does with them, so the browser's port
    (frontend/src/lib/temporal.ts) can be checked frame by frame."""
    from gestureflow.core.temporal import LetterCommitter

    rng = np.random.default_rng(seed)
    streams = []
    for _ in range(n_streams):
        frames = []
        for _ in range(int(rng.integers(2, 8))):
            held = str(rng.choice(list("ABC")))
            for _ in range(int(rng.integers(3, 35))):
                r = rng.random()
                if r < 0.03:
                    frames.append(None)
                elif r < 0.10:
                    frames.append([str(rng.choice(list("ABC"))), 0.95])
                else:
                    frames.append([held, float(rng.choice([0.6, 0.85, 0.95, 0.99],
                                                          p=[.05, .05, .45, .45]))])
        c = LetterCommitter()
        out = []
        for f in frames:
            if f is None:
                c.no_hand()
                out.append([None, None, 0.0])
            else:
                s = c.update(*f)
                out.append([s.smoothed, s.committed, c.progress])
        streams.append({"frames": frames, "expected": out})
    return streams


def main():
    import pandas as pd
    import torch

    from gestureflow.core.inference import GesturePredictor
    from gestureflow.paths import EDGE_FIXTURE, EDGE_PARITY_REPORT, LANDMARKS_CSV, MODEL_PATH, REPO

    cols = [f"{a}{i}" for i in range(21) for a in "xy"]
    df = pd.read_csv(LANDMARKS_CSV)
    raw = df[cols].to_numpy(np.float32).reshape(-1, 21, 2)
    predictor = GesturePredictor(MODEL_PATH)

    hashes, agree, differing_pixels, disagreements = [], 0, 0.0, []
    with torch.no_grad():
        for start in range(0, len(raw), 256):
            server, edge = [], []
            for r in raw[start:start + 256]:
                pts = P.normalize(r)
                ref, mine = P.render(pts), render_edge(pts)
                hashes.append(render_hash(mine))
                differing_pixels += (ref != mine).any(-1).mean()
                server.append(P.to_input(ref))
                edge.append(P.to_input(mine))
            ps = predictor.model(torch.from_numpy(np.stack(server))).argmax(1).numpy()
            pe = predictor.model(torch.from_numpy(np.stack(edge))).argmax(1).numpy()
            agree += int((ps == pe).sum())
            for k in np.flatnonzero(ps != pe):
                disagreements.append({"row": start + int(k),
                                      "server": predictor.labels[ps[k]],
                                      "edge": predictor.labels[pe[k]]})

    report = {
        "samples": len(raw),
        "same_letter": agree,
        "agreement": round(agree / len(raw), 4),
        "pixels_differing_pct": round(differing_pixels / len(raw) * 100, 3),
        "bone_radius": EDGE_BONE_RADIUS,
        "disagreements": disagreements,
        "note": "Same checkpoint, OpenCV render (server) vs the browser rasterizer (edge).",
    }
    EDGE_PARITY_REPORT.write_text(json.dumps(report, indent=2))
    # whole-pipeline check for the browser: raw landmarks -> expected Edge-mode probabilities
    golden = json.loads((REPO / "tests" / "golden" / "core_golden.json").read_text())["cases"]
    with torch.no_grad():
        x = torch.from_numpy(np.stack([
            P.to_input(render_edge(P.normalize(np.asarray(c["raw"], np.float32))))
            for c in golden]))
        probs = torch.softmax(predictor.model(x), 1).numpy()
    pipeline = [{"label": c["label"], "raw": c["raw"], "probs": [round(float(v), 6) for v in pr]}
                for c, pr in zip(golden, probs, strict=True)]
    EDGE_FIXTURE.write_text(json.dumps({"csv": "data/landmarks.csv", "render_sha256": hashes,
                                        "temporal": temporal_fixture(), "pipeline": pipeline,
                                        "labels": predictor.labels}))
    print(f"edge vs server: same letter on {agree}/{len(raw)} samples "
          f"({agree / len(raw):.2%}); {report['pixels_differing_pct']}% of pixels differ")


if __name__ == "__main__":
    main()
