"""
Measure what calibration does for someone the model hasn't seen.

    python -m gestureflow.train --test-session day2 --out models/held_out_day2.pt
    python -m gestureflow.calibration_eval --checkpoint models/held_out_day2.pt --session day2

The checkpoint must be trained WITHOUT --session, which then plays the new user: the first
--frames frames of each letter are their calibration, the rest of the session is the test.
Writes docs/calibration_report.json. Calibration frames and test frames come from the same
sitting -- which is also how it is used (calibrate, then sign) -- so this is the favourable
case; across sittings the gain may be smaller.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from gestureflow.core import preprocessing
from gestureflow.core.calibration import ALPHA, TAU, Calibration, build_prototypes
from gestureflow.core.model import SkeletonCNN
from gestureflow.core.temporal import CONF_COMMIT
from gestureflow.paths import CALIBRATION_REPORT, LANDMARKS_CSV

COLS = [f"{a}{i}" for i in range(21) for a in "xy"]


@torch.no_grad()
def probs_and_embeddings(model, df):
    x = torch.from_numpy(np.stack([
        preprocessing.to_input(preprocessing.render(preprocessing.normalize(r)))
        for r in df[COLS].to_numpy(np.float32).reshape(-1, 21, 2)]))
    logits, emb = model.forward_with_embedding(x)
    return F.softmax(logits, 1).numpy(), F.normalize(emb, dim=1).numpy()


def scores(y, probs, labels):
    pred, conf = probs.argmax(1), probs.max(1)
    per = f1_score(y, pred, average=None, labels=range(len(labels)))
    committable = conf > CONF_COMMIT
    return {"accuracy": round(float(np.mean(pred == y)), 4),
            # a letter only commits above CONF_COMMIT, so accuracy alone can hide a calibration
            # that is right but never confident enough to be used
            "can_commit": round(float(np.mean(committable)), 4),
            "wrong_and_can_commit": round(float(np.mean(committable & (pred != y))), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "per_letter_f1": {label: round(float(v), 4)
                              for label, v in zip(labels, per, strict=True)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--session", default="day2", help="held-out session that plays the new user")
    ap.add_argument("--frames", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--out", default=str(CALIBRATION_REPORT))
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = ck["labels"]
    model = SkeletonCNN(len(labels))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    df = pd.read_csv(LANDMARKS_CSV)
    user = df[df.session == args.session].sort_values("timestamp")
    index = {label: i for i, label in enumerate(labels)}
    report = {"checkpoint": args.checkpoint, "session": args.session, "alpha_shipped": ALPHA,
              "tau": TAU, "commit_threshold": CONF_COMMIT,
              "note": "calibration and test frames are from the same sitting", "runs": []}

    for k in args.frames:
        calib_rows = user.groupby("label").head(k).index
        calib, test = user.loc[calib_rows], user.drop(calib_rows)
        _, calib_emb = probs_and_embeddings(model, calib)
        probs, emb = probs_and_embeddings(model, test)
        y = test.label.map(index).to_numpy()
        by_label = {label: list(calib_emb[(calib.label == label).to_numpy()]) for label in labels}
        protos = build_prototypes(by_label, labels)

        run = {"frames_per_letter": k, "test_frames": len(test),
               "without": scores(y, probs, labels), "alpha_sweep": {}}
        for alpha in (0.25, 0.5, 0.75, 1.0):
            blended = Calibration(protos, alpha=alpha).apply(probs, emb)
            run["alpha_sweep"][str(alpha)] = scores(y, blended, labels)
        report["runs"].append(run)
        before, after = run["without"], run["alpha_sweep"][str(ALPHA)]
        print(f"{k:2d} frames/letter: accuracy {before['accuracy']:.4f} -> "
              f"{after['accuracy']:.4f}, macro F1 {before['macro_f1']:.4f} -> "
              f"{after['macro_f1']:.4f}, K F1 {before['per_letter_f1']['K']:.2f} -> "
              f"{after['per_letter_f1']['K']:.2f}, can commit {before['can_commit']:.2f} -> "
              f"{after['can_commit']:.2f}, wrong & can commit {after['wrong_and_can_commit']:.3f}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
