"""The training split must never put one recording session on both sides of train/test."""

import pandas as pd
import pytest

from gestureflow.splits import latest_group, leakage_report, time_tail
from gestureflow.train import make_split


def make_df(sessions=("day1", "day2", "day3"), labels="ABC", n=20):
    rows = []
    for s_i, s in enumerate(sessions):
        for label in labels:
            for i in range(n):
                rows.append({"label": label, "session": s,
                             "timestamp": 1000.0 * s_i + i,
                             "x0": float(i), "y0": float(s_i)})   # unique per session
    return pd.DataFrame(rows)


def test_latest_group_uses_timestamps_not_names():
    df = make_df(sessions=("zzz_old", "aaa_new"))
    assert latest_group(df) == "aaa_new"


def test_default_split_holds_out_latest_session_entirely():
    train, val, test = make_split(make_df())
    assert set(test["session"]) == {"day3"}
    assert "day3" not in set(train["session"]) | set(val["session"])
    assert not [p for p in leakage_report(train, val.iloc[:0], test) if p.startswith("LEAK")]


def test_explicit_test_session():
    train, _, test = make_split(make_df(), "day1")
    assert set(test["session"]) == {"day1"}
    assert set(train["session"]) == {"day2", "day3"}


def test_none_trains_on_everything():
    df = make_df()
    train, val, test = make_split(df, "none")
    assert len(test) == 0
    assert len(train) + len(val) == len(df)


def test_unknown_session_is_an_error():
    with pytest.raises(ValueError):
        make_split(make_df(), "day9")


def test_time_tail_takes_latest_frames_of_every_label_and_session():
    df = make_df(n=20)
    head, tail = time_tail(df, 0.15)
    assert len(head) + len(tail) == len(df)
    for (_, _), g in tail.groupby(["label", "session"]):
        assert len(g) == 3
    for key, g in tail.groupby(["label", "session"]):
        h = head[(head["label"] == key[0]) & (head["session"] == key[1])]
        assert g["timestamp"].min() > h["timestamp"].max()


def test_balanced_loader_draws_letters_equally():
    import numpy as np
    import torch

    from gestureflow.train import balanced_loader

    class Fake(torch.utils.data.Dataset):
        y = np.array([0] * 900 + [1] * 100)        # letter 1 has 9x fewer frames

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return 0, int(self.y[i])

    drawn = torch.cat([y for _, y in balanced_loader(Fake(), batch_size=500)]).numpy()
    share = np.bincount(drawn, minlength=2) / len(drawn)
    assert abs(share[1] - 0.5) < 0.06
