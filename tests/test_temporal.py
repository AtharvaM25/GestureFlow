"""LetterCommitter must behave exactly like the loop it was extracted from in M5_test.py."""

from collections import Counter, deque

import numpy as np

from gestureflow.core.temporal import LetterCommitter


def reference(frames):
    """The pre-refactor M5_test.py loop, verbatim apart from I/O. frames: (label, conf)
    tuples, or None for a frame with no hand. Returns (smoothed, committed) per frame."""
    stable = 0
    recent = deque(maxlen=7)
    out = []
    for f in frames:
        if f is None:
            recent.append(None)
            stable = 0
            out.append((None, None))
            continue
        label, conf = f
        recent.append(label if conf > 0.70 else None)
        votes = [v for v in recent if v is not None]
        smoothed = Counter(votes).most_common(1)[0][0] if votes else None
        committed = None
        if conf > 0.90 and smoothed == label:
            stable += 1
            if stable >= 15:
                committed = label
                stable = 0
                recent.clear()
        else:
            stable = 0
        out.append((smoothed, committed))
    return out


def run(frames):
    c = LetterCommitter()
    out = []
    for f in frames:
        if f is None:
            c.no_hand()
            out.append((None, None))
        else:
            s = c.update(*f)
            out.append((s.smoothed, s.committed))
    return out


def test_commits_after_fifteen_confident_frames():
    out = run([("A", 0.95)] * 15)
    assert [c for _, c in out] == [None] * 14 + ["A"]


def test_low_confidence_frame_resets_the_count():
    out = run([("A", 0.95)] * 14 + [("A", 0.80)] + [("A", 0.95)] * 14)
    assert all(c is None for _, c in out)


def test_lost_hand_resets_the_count():
    out = run([("A", 0.95)] * 10 + [None] + [("A", 0.95)] * 14)
    assert all(c is None for _, c in out)


def test_same_letter_twice_needs_two_full_holds():
    out = run([("L", 0.95)] * 30)
    assert [c for _, c in out].count("L") == 2


def test_matches_reference_on_random_streams():
    rng = np.random.default_rng(0)
    commits = 0
    for _ in range(200):
        # held letters with occasional flicker, low confidence and lost hands -- the
        # shape of real signing, so commits and resets both happen
        frames = []
        for _ in range(int(rng.integers(2, 8))):
            held = str(rng.choice(list("ABC")))
            for _ in range(int(rng.integers(3, 35))):
                r = rng.random()
                if r < 0.03:
                    frames.append(None)
                elif r < 0.10:
                    frames.append((str(rng.choice(list("ABC"))), 0.95))
                else:
                    frames.append((held, float(rng.choice([0.6, 0.85, 0.95, 0.99],
                                                          p=[.05, .05, .45, .45]))))
        expected = reference(frames)
        assert run(frames) == expected
        commits += sum(c is not None for _, c in expected)
    assert commits > 20, "streams too noisy to exercise the commit path"
