"""RecognitionSession must reproduce what the predictor and committer do on their own."""

import json
from pathlib import Path

import numpy as np
import pytest

from gestureflow.core.inference import GesturePredictor
from gestureflow.core.session import RecognitionSession
from gestureflow.core.temporal import LetterCommitter
from gestureflow.paths import MODEL_PATH

CASES = json.loads((Path(__file__).parent / "golden" / "core_golden.json").read_text())["cases"]


@pytest.fixture(scope="module")
def predictor():
    return GesturePredictor(MODEL_PATH)


def test_frame_matches_predictor(predictor):
    case = CASES[0]
    res = RecognitionSession(predictor).step(case["raw"])
    pred = predictor.predict(case["raw"])
    assert res.hand and res.raw == pred.label
    assert res.confidence == pytest.approx(pred.confidence)
    assert res.alternatives[0] == (pred.label, pytest.approx(pred.confidence))
    assert len(res.alternatives) == 3
    probs = [p for _, p in res.alternatives]
    assert probs == sorted(probs, reverse=True)
    assert res.latency_ms > 0


def test_no_hand(predictor):
    res = RecognitionSession(predictor).step(None)
    assert not res.hand and res.gesture is None and res.committed is None


def test_holding_a_confident_letter_commits_like_the_committer(predictor):
    # pick a golden case the checkpoint is confident about
    case = max(CASES, key=lambda c: max(c["probs"]))
    session = RecognitionSession(predictor)
    reference = LetterCommitter()
    pred = predictor.predict(case["raw"])
    for _ in range(LetterCommitter().commit_frames):
        res = session.step(case["raw"])
        ref = reference.update(pred.label, pred.confidence)
        assert (res.gesture, res.committed) == (ref.smoothed, ref.committed)
    assert res.committed == case["label"]
    assert res.stability == 0.0          # resets after a commit


def test_stability_rises_while_holding(predictor):
    case = max(CASES, key=lambda c: max(c["probs"]))
    session = RecognitionSession(predictor)
    values = [session.step(case["raw"]).stability for _ in range(5)]
    assert values == sorted(values) and values[-1] > 0
    assert np.isclose(values[-1], 5 / LetterCommitter().commit_frames)
