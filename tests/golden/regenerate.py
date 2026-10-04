"""
Rewrite tests/golden/core_golden.json from the current code.

Only run this when a change to preprocessing or the model is deliberate -- and retrain,
since the checkpoint was trained on the old output. The diff of the JSON is the review.

    python -m tests.golden.regenerate
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gestureflow.core import preprocessing
from gestureflow.core.inference import GesturePredictor
from gestureflow.paths import LANDMARKS_CSV, MODEL_PATH

COLS = [f"{a}{i}" for i in range(21) for a in "xy"]
OUT = Path(__file__).with_name("core_golden.json")
AUG_SEED = 1234


def sha(img):
    return hashlib.sha256(img.tobytes()).hexdigest()


def main():
    df = pd.read_csv(LANDMARKS_CSV)
    # First day1 frame of every letter; inputs are stored in the JSON, so adding new
    # recordings to landmarks.csv never changes this file.
    rows = (df[df.session == "day1"].sort_values("timestamp").groupby("label").head(1)
            .sort_values("label"))
    predictor = GesturePredictor(MODEL_PATH)

    cases = []
    for _, r in rows.iterrows():
        raw = r[COLS].to_numpy(np.float32).reshape(21, 2)
        pts = preprocessing.normalize(raw)
        aug = preprocessing.augment(pts, np.random.default_rng(AUG_SEED))
        pred = predictor.predict(raw)
        cases.append({
            "label": r["label"],
            "raw": raw.tolist(),
            "normalized": pts.tolist(),
            "render_sha256": sha(preprocessing.render(pts)),
            "augmented_seed1234": aug.tolist(),
            "augmented_render_sha256": sha(preprocessing.render(aug)),
            "probs": [round(float(p), 6) for p in pred.probs],
        })

    out = {"config": preprocessing.config(), "checkpoint_labels": predictor.labels,
           "cases": cases}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
