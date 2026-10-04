"""Tests for the confusion-aware collector. The camera loop is excluded -- it needs
physical hardware -- but every decision the collector makes is covered."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from gestureflow.collect import (
    HEADER,
    IncrementalWriter,
    build_plan,
    class_spread,
    load_f1,
)
from gestureflow.paths import LANDMARKS_CSV, REPO

COORD_COLS = [f"{a}{i}" for i in range(21) for a in "xy"]


# A plausible hand: wrist at origin, five fingers splayed. Random points would
# normalize to a huge spread and make every synthetic class look inconsistent.
_BASE_HAND = np.array(
    [[0, 0]] + [[dx, dy] for dx in (-30, -15, 0, 15, 30) for dy in (-25, -50, -75, -100)],
    dtype=np.float32,
) + np.array([320, 240], dtype=np.float32)


def make_df(spec, jitter=1.5, seed=0):
    """spec: {label: [(session, n_samples), ...]}. Samples are one consistent hand
    shape plus small jitter, so normalized spread stays low as it would in real data."""
    rows = []
    rng = np.random.default_rng(seed)
    for label, sessions in spec.items():
        offset = rng.normal(0, 12, (21, 2)).astype(np.float32)   # per-class shape
        for session, n in sessions:
            for i in range(n):
                pts = _BASE_HAND + offset + rng.normal(0, jitter, (21, 2)).astype(np.float32)
                rows.append({"label": label, "session": session, "timestamp": 1e9 + i,
                             "seq": i, **dict(zip(COORD_COLS, pts.reshape(-1), strict=True))})
    return pd.DataFrame(rows)


def test_fixture_produces_realistic_spread():
    """Guards the fixture itself: if synthetic hands drift to noise, the prioritization
    tests below stop testing what they claim to."""
    df = make_df({"A": [("d1", 50)]})
    s = class_spread(df[COORD_COLS].to_numpy(np.float32).reshape(-1, 21, 2))
    assert s < 0.12, f"fixture spread {s:.3f} is unrealistically high"


# ------------------------------------------------------------ crash safety
def test_writer_persists_each_row_immediately(tmp_path):
    """The bug this collector fixes: collecter.py buffered rows in memory and wrote only
    on a clean exit, so an interrupted sitting was lost."""
    csv = tmp_path / "out.csv"
    pts = np.arange(42, dtype=np.float32).reshape(21, 2)

    with IncrementalWriter(csv) as w:
        w.write("K", "day3", 0, pts)
        # Read from a separate handle while the writer is still open.
        mid = pd.read_csv(csv)
        assert len(mid) == 1, "row must be on disk before the writer closes"
        w.write("K", "day3", 1, pts)

    assert len(pd.read_csv(csv)) == 2


def test_writer_survives_process_kill(tmp_path):
    """Hard-kill a real subprocess mid-capture; previously written rows must survive."""
    csv = tmp_path / "out.csv"
    script = f"""
import sys, numpy as np
sys.path.insert(0, {str(REPO)!r})
from gestureflow.collect import IncrementalWriter
w = IncrementalWriter({str(csv)!r})
pts = np.zeros((21, 2), dtype=np.float32)
for i in range(3):
    w.write("K", "day3", i, pts)
sys.stdout.write("written"); sys.stdout.flush()
import os; os.kill(os.getpid(), 9)      # SIGKILL: no cleanup, no close()
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert proc.stdout == b"written"
    assert proc.returncode != 0, "process should have been killed, not exited cleanly"
    assert len(pd.read_csv(csv)) == 3, "fsync'd rows must survive SIGKILL"


def test_writer_appends_without_duplicating_header(tmp_path):
    csv = tmp_path / "out.csv"
    pts = np.zeros((21, 2), dtype=np.float32)
    for _ in range(2):
        with IncrementalWriter(csv) as w:
            w.write("A", "day1", 0, pts)
    assert csv.read_text().count("label,session") == 1
    assert list(pd.read_csv(csv).columns) == HEADER


# ------------------------------------------------------------ prioritization
def test_low_cross_session_f1_outranks_everything():
    """K is the real case: low spread, full sample count, but F1 0.18 on an unseen
    session. Spread-based ranking misses it; the plan must not."""
    df = make_df({"K": [("day1", 150), ("day2", 150), ("day3", 150)],
                  "M": [("day1", 100), ("day2", 100)]})
    plan = build_plan(df, {"K": 0.18, "M": 0.84}, target=150)
    assert plan[0].label == "K"
    assert "cross-session F1 0.18" in "; ".join(plan[0].reasons)


def test_session_coverage_counted():
    df = make_df({"A": [("day1", 150)], "B": [("day1", 150), ("day2", 150),
                                              ("day3", 150)]})
    plan = {n.label: n for n in build_plan(df, {}, target=150)}
    assert plan["A"].n_sessions == 1 and plan["A"].priority > 0
    assert plan["B"].n_sessions == 3 and plan["B"].priority == 0, \
        "a class meeting every target must not be queued"


def test_healthy_class_is_not_queued():
    df = make_df({"A": [("d1", 150), ("d2", 150), ("d3", 150)]})
    assert build_plan(df, {"A": 0.99}, target=150)[0].priority == 0


def test_plan_is_deterministic():
    df = make_df({"A": [("d1", 100)], "B": [("d1", 100)], "C": [("d1", 100)]})
    runs = [[n.label for n in build_plan(df, {}, 150)] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


# ------------------------------------------------------------ report parsing
def test_load_f1_averages_across_folds(tmp_path):
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps({
        "protocol": "leave-one-session-out",
        "folds": {"day1": {"per_class_f1": {"K": 0.20}},
                  "day2": {"per_class_f1": {"K": 0.40}}},
    }))
    f1, note = load_f1(rep)
    assert f1["K"] == pytest.approx(0.30)
    assert "2 fold(s)" in note


def test_missing_or_corrupt_report_degrades_gracefully(tmp_path):
    assert load_f1(tmp_path / "nope.json") == ({}, None)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_f1(bad) == ({}, None)


# ------------------------------------------------------------ spread metric
def test_spread_zero_for_identical_samples():
    pts = np.tile(np.arange(42, dtype=np.float32).reshape(1, 21, 2), (10, 1, 1))
    assert class_spread(pts) == pytest.approx(0.0, abs=1e-6)


def test_spread_matches_published_values():
    """M and N are the high-spread classes in the README table, computed the same way
    by gestureflow/preview.py. Guards against the metric silently changing."""
    df = pd.read_csv(LANDMARKS_CSV)
    got = {L: class_spread(g[COORD_COLS].to_numpy(np.float32).reshape(-1, 21, 2))
           for L, g in df.groupby("label") if L in ("M", "N", "Z")}
    assert got["M"] == pytest.approx(0.2592, abs=5e-4)
    assert got["N"] == pytest.approx(0.2286, abs=5e-4)
    assert got["Z"] == pytest.approx(0.0418, abs=5e-4)
