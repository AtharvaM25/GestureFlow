"""
Train the CNN.
Run: python -m gestureflow.train --epochs 40

The most recent session is held out as the test set, so the printed TEST accuracy is
measured on a sitting the model never saw. Once that number is known, retrain the model
you ship with --test-session none.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from gestureflow.core import preprocessing
from gestureflow.core.model import SkeletonCNN
from gestureflow.paths import LANDMARKS_CSV, MODEL_PATH
from gestureflow.splits import (
    describe_split,
    grouped_holdout,
    latest_group,
    leakage_report,
    time_tail,
)

COLS = [f"{a}{i}" for i in range(21) for a in "xy"]


# ---------------------------------------------------------------- data
class SkeletonDataset(Dataset):

    def __init__(self, df, labels, augment=False, seed=0):
        self.coords = df[COLS].to_numpy(np.float32).reshape(-1, 21, 2)
        self.y = df["label"].map(
            {L: i for i, L in enumerate(labels)}).to_numpy()
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        pts = preprocessing.normalize(self.coords[i])
        if self.augment:
            pts = preprocessing.augment(pts, self.rng)
        img = preprocessing.render(pts)
        return torch.from_numpy(preprocessing.to_input(img)), int(self.y[i])


def balanced_loader(dataset, batch_size, seed=0):
    """Draw every letter equally often, however many frames each has. Recording extra
    sittings for weak letters would otherwise tilt the model toward guessing them."""
    counts = np.bincount(dataset.y)
    weights = torch.as_tensor(1.0 / counts[dataset.y], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def make_split(df, test_session=None, val_frac=0.15):
    """Hold one whole session out as the test set; validate on the time tail of the rest.

    test_session=None picks the most recent session. "none" holds nothing out and trains
    on every session -- for the shipped model, once the held-out number has been measured.
    """
    if test_session is None:
        test_session = latest_group(df)
    if test_session == "none":
        pool, test = df, df.iloc[:0]
    else:
        pool, _, test = grouped_holdout(df, [test_session])
    train, val = time_tail(pool, val_frac)
    return train, val, test


# ------------------------------------------------------------ evaluate
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    loss_sum = correct = total = 0
    preds, trues = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += crit(out, y).item() * len(y)
        p = out.argmax(1)
        correct += (p == y).sum().item()
        total += len(y)
        preds.append(p.cpu().numpy())
        trues.append(y.cpu().numpy())

    return (loss_sum / total, correct / total,
            np.concatenate(preds), np.concatenate(trues))


# ---------------------------------------------------------------- main
def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(LANDMARKS_CSV))
    ap.add_argument("--out", default=str(MODEL_PATH))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-session", default=None,
                    help="session held out as the test set (default: most recent). "
                         "'none' trains on every session and reports no test accuracy.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    labels = sorted(df["label"].unique())
    print(f"{len(df)} samples, {len(labels)} classes")

    train_df, val_df, test_df = make_split(df, args.test_session)
    for name, split in (("train", train_df), ("val", val_df), ("test", test_df)):
        print(describe_split(name, split))

    # val shares sessions with train by design (checkpoint selection only), so the
    # leakage check covers test alone.
    problems = leakage_report(train_df, val_df.iloc[:0], test_df)
    for p in problems:
        print(p)
    if any(p.startswith("LEAK") for p in problems):
        raise SystemExit("refusing to train on a leaky split")

    train_dl = balanced_loader(SkeletonDataset(train_df, labels, augment=True), args.batch_size)
    val_dl = DataLoader(SkeletonDataset(val_df, labels),
                        batch_size=args.batch_size)
    test_dl = DataLoader(SkeletonDataset(test_df, labels),
                         batch_size=args.batch_size) if len(test_df) else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SkeletonCNN(len(labels)).to(device)
    print(f"device: {device}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = correct = total = 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = crit(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            run_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)

        sched.step()
        va_loss, va_acc, _, _ = evaluate(model, val_dl, device)

        flag = ""
        if va_acc > best_val:
            best_val = va_acc
            flag = "  <- saved"
            torch.save({
                "state_dict": model.state_dict(),
                "labels": labels,
                # key name predates the move to gestureflow.core; kept so
                # existing checkpoints still load
                "handnorm_config": preprocessing.config(),
            }, args.out)

        print(f"epoch {epoch:3d}  train {run_loss/total:.3f}/{correct/total:.3f}"
              f"   val {va_loss:.3f}/{va_acc:.3f}{flag}")

    print("\n" + "=" * 60)
    if test_dl is None:
        print(f"no held-out session; best val {best_val:.4f} (within-session, "
              f"not a generalization number)")
        return

    model.load_state_dict(torch.load(
        args.out, map_location=device, weights_only=True)["state_dict"])
    _, te_acc, preds, trues = evaluate(model, test_dl, device)
    print(f"TEST accuracy on unseen session {test_df['session'].iloc[0]!r}: "
          f"{te_acc:.4f}   (best val {best_val:.4f})")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(trues, preds, labels=list(range(len(labels))))
    pairs = [(cm[i, j], labels[i], labels[j])
             for i in range(len(labels)) for j in range(len(labels))
             if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)

    print("\nworst confusions (true -> predicted):")
    for n, a, b in pairs[:10]:
        print(f"  {a} -> {b}: {n}")
    print("\nIf one or two pairs dominate, that is a gesture design problem,")
    print("not a model problem. More epochs will not fix it.")


if __name__ == "__main__":
    main()
