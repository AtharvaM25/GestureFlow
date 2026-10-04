"""
Group-aware dataset splitting.

Replaces CNN.make_split, which sorted each class by (session, timestamp) and took the
tail. Measured consequence on landmarks.csv: val and test were 100% `day2` while `day2`
was also 38% of train. Burst capture writes every 3rd frame of a continuous sitting, so a
train frame and a test frame could be ~0.1s apart -- near-duplicate poses on opposite
sides of the split.

Every function here splits on a *group* key (session, and later signer) so that all frames
from one recording sitting land on exactly one side.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

__all__ = ["leave_one_group_out", "grouped_holdout", "time_tail", "latest_group",
           "describe_split", "leakage_report"]

DEFAULT_GROUP = "session"


def leave_one_group_out(
    df: pd.DataFrame, group: str = DEFAULT_GROUP
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield (held_out_name, train_df, test_df) once per group.

    This is the honest protocol: the held-out group contributes nothing to training.
    With two sessions it gives two folds; add sessions or signers and it scales.
    """
    groups = sorted(df[group].unique())
    if len(groups) < 2:
        raise ValueError(
            f"need >=2 distinct {group!r} values to hold one out, found {groups}. "
            f"Collect another session before quoting a generalization number."
        )
    for held in groups:
        test = df[df[group] == held].reset_index(drop=True)
        train = df[df[group] != held].reset_index(drop=True)
        yield str(held), train, test


def grouped_holdout(
    df: pd.DataFrame, test_groups: list[str], val_groups: list[str] | None = None,
    group: str = DEFAULT_GROUP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Explicit train/val/test by group membership. No group appears twice."""
    val_groups = val_groups or []
    overlap = set(test_groups) & set(val_groups)
    if overlap:
        raise ValueError(f"groups in both val and test: {sorted(overlap)}")

    present = set(df[group].unique())
    missing = (set(test_groups) | set(val_groups)) - present
    if missing:
        raise ValueError(f"{group}(s) not in data: {sorted(missing)}; have {sorted(present)}")

    test = df[df[group].isin(test_groups)].reset_index(drop=True)
    val = df[df[group].isin(val_groups)].reset_index(drop=True)
    train = df[~df[group].isin(set(test_groups) | set(val_groups))].reset_index(drop=True)
    return train, val, test


def time_tail(
    df: pd.DataFrame, frac: float = 0.15, group: str = DEFAULT_GROUP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the last `frac` of every (label, group) run by timestamp.

    The tail comes from the same sittings as the head, so it is correlated with it. Use it
    to pick a checkpoint, never to report accuracy -- that is what a held-out group is for.
    """
    head, tail = [], []
    for _, g in df.groupby(["label", group]):
        g = g.sort_values("timestamp")
        cut = len(g) - max(1, int(len(g) * frac))
        head.append(g.iloc[:cut])
        tail.append(g.iloc[cut:])
    return (pd.concat(head).reset_index(drop=True),
            pd.concat(tail).reset_index(drop=True))


def latest_group(df: pd.DataFrame, group: str = DEFAULT_GROUP) -> str:
    """The group recorded most recently, by its newest timestamp."""
    return str(df.groupby(group)["timestamp"].max().idxmax())


def describe_split(name: str, split: pd.DataFrame, group: str = DEFAULT_GROUP) -> str:
    if not len(split):
        return f"{name:>6}  (empty)"
    mix = {str(k): int(v) for k, v in split.groupby(group).size().items()}
    return f"{name:>6}  n={len(split):<5} classes={split['label'].nunique():<3} {group}s={mix}"


def leakage_report(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    group: str = DEFAULT_GROUP,
) -> list[str]:
    """Return a list of human-readable leakage findings. Empty list means clean."""
    problems: list[str] = []
    tr = set(train[group].unique()) if len(train) else set()

    for name, split in (("val", val), ("test", test)):
        if not len(split):
            continue
        shared = tr & set(split[group].unique())
        if shared:
            n_tr = len(train[train[group].isin(shared)])
            pct = 100.0 * n_tr / max(len(train), 1)
            problems.append(
                f"LEAK: {name} shares {group}(s) {sorted(shared)} with train "
                f"({n_tr} rows, {pct:.0f}% of train). Frames from one sitting are "
                f"highly correlated; this inflates {name} accuracy."
            )

    coord_cols = [c for c in train.columns if c[:1] in "xy" and c[1:].isdigit()]
    if coord_cols and len(test):
        merged = train[coord_cols].merge(test[coord_cols], how="inner")
        if len(merged):
            problems.append(f"LEAK: {len(merged)} exactly duplicated coordinate rows "
                            "across train/test.")

    missing = set(train["label"].unique()) ^ set(test["label"].unique()) if len(test) else set()
    if missing:
        problems.append(f"WARN: classes not present on both sides: {sorted(missing)}")

    return problems


if __name__ == "__main__":
    import argparse

    from gestureflow.paths import LANDMARKS_CSV

    ap = argparse.ArgumentParser(description="Inspect splits for leakage.")
    ap.add_argument("--csv", default=str(LANDMARKS_CSV))
    ap.add_argument("--group", default=DEFAULT_GROUP)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"{len(df)} rows, {df['label'].nunique()} classes, "
          f"{args.group}s={sorted(df[args.group].unique())}\n")

    print("--- repo's original make_split (tail of each class) ---")
    tr, va, te = [], [], []
    for _, g in df.groupby("label"):
        g = g.sort_values(["session", "timestamp"])
        n = len(g)
        i_v, i_t = int(n * 0.70), int(n * 0.85)
        tr.append(g.iloc[:i_v])
        va.append(g.iloc[i_v:i_t])
        te.append(g.iloc[i_t:])
    otr, ova, ote = (pd.concat(x) for x in (tr, va, te))
    for nm, s in (("train", otr), ("val", ova), ("test", ote)):
        print(" ", describe_split(nm, s, args.group))
    for p in leakage_report(otr, ova, ote, args.group) or ["  clean"]:
        print("  " + p)

    print("\n--- leave-one-group-out ---")
    for held, trn, tst in leave_one_group_out(df, args.group):
        print(f"  hold out {held!r}:")
        print("   ", describe_split("train", trn, args.group))
        print("   ", describe_split("test", tst, args.group))
        for p in leakage_report(trn, trn.iloc[:0], tst, args.group) or ["clean"]:
            print("    " + p)
