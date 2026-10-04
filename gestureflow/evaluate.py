"""
Honest evaluation: train on one session, test on another that the model has never seen.

This is the protocol behind the numbers in docs/EVALUATION.md. It changes exactly one
thing from the repo's own training recipe -- the split -- so any difference in the result
is attributable to leakage and nothing else.

    python -m gestureflow.evaluate --epochs 25

Reports accuracy, macro F1 and per-class F1 on the held-out session, plus the confusion
pairs that dominate the error. Writes a JSON report for the model registry.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gestureflow.core.model import SkeletonCNN
from gestureflow.paths import EVAL_REPORT, LANDMARKS_CSV
from gestureflow.splits import describe_split, leave_one_group_out
from gestureflow.train import SkeletonDataset, balanced_loader, evaluate


def run_fold(train_df, test_df, labels, args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dl = balanced_loader(SkeletonDataset(train_df, labels, augment=True, seed=args.seed),
                               args.batch_size, seed=args.seed)
    test_dl = DataLoader(SkeletonDataset(test_df, labels), batch_size=128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SkeletonCNN(len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        correct = total = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        sched.step()
        if ep % args.log_every == 0 or ep in (1, args.epochs):
            _, acc, _, _ = evaluate(model, test_dl, device)
            print(f"  ep{ep:3d}  train={correct/total:.3f}  held-out={acc:.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    from sklearn.metrics import confusion_matrix, f1_score

    _, acc, preds, trues = evaluate(model, test_dl, device)
    macro = f1_score(trues, preds, average="macro")
    per_class = f1_score(trues, preds, average=None, labels=list(range(len(labels))))

    cm = confusion_matrix(trues, preds, labels=list(range(len(labels))))
    pairs = sorted(
        ((int(cm[i, j]), labels[i], labels[j])
         for i in range(len(labels)) for j in range(len(labels))
         if i != j and cm[i, j] > 0),
        reverse=True,
    )[:8]

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "per_class_f1": {labels[i]: round(float(per_class[i]), 4) for i in range(len(labels))},
        "top_confusions": [{"true": a, "pred": b, "n": n} for n, a, b in pairs],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "epochs": args.epochs,
        "seed": args.seed,
        "train_seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(LANDMARKS_CSV))
    ap.add_argument("--group", default="session", help="column to hold out (session, signer)")
    ap.add_argument("--hold-out", default=None,
                    help="run only this fold; default runs every group")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--out", default=str(EVAL_REPORT))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    labels = sorted(df["label"].unique())
    print(f"{len(df)} rows, {len(labels)} classes, "
          f"{args.group}s={sorted(df[args.group].unique())}\n")

    folds = {}
    for held, train_df, test_df in leave_one_group_out(df, args.group):
        if args.hold_out and held != args.hold_out:
            continue
        print(f"=== hold out {args.group}={held!r} ===")
        print(" ", describe_split("train", train_df, args.group))
        print(" ", describe_split("test", test_df, args.group))
        res = run_fold(train_df, test_df, labels, args)
        print(f"  RESULT  acc={res['accuracy']:.4f}  macroF1={res['macro_f1']:.4f}")
        worst = sorted(res["per_class_f1"].items(), key=lambda kv: kv[1])[:6]
        print("  worst classes: " + ", ".join(f"{k}={v:.2f}" for k, v in worst))
        print("  top confusions: "
              + ", ".join(f"{c['true']}->{c['pred']}:{c['n']}" for c in res["top_confusions"][:5]))
        print()
        folds[held] = res

    if not folds:
        raise SystemExit(f"no fold matched --hold-out {args.hold_out!r}")

    accs = [f["accuracy"] for f in folds.values()]
    report = {
        "protocol": f"leave-one-{args.group}-out",
        "csv": Path(args.csv).name,
        "n_folds": len(folds),
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "folds": folds,
        "caveats": [
            "Single seed per fold; no variance estimate.",
            "All data from one signer; cross-signer generalization is unmeasured.",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"mean accuracy over {len(folds)} fold(s): {report['mean_accuracy']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
