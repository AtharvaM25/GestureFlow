"""
Golden-vector test: the core must produce exactly what the pre-refactor code produced.

core_golden.json was captured from handnorm.py and CNN.SkeletonCNN before they moved into
gestureflow.core. Training, live inference and (later) the browser all depend on these
outputs being identical, so any drift fails here instead of silently degrading accuracy.
If a change is intentional, run `python -m tests.golden.regenerate` and retrain.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gestureflow.core import preprocessing
from gestureflow.core.inference import GesturePredictor
from gestureflow.paths import MODEL_PATH

GOLDEN = json.loads((Path(__file__).parent / "golden" / "core_golden.json").read_text())
CASES = GOLDEN["cases"]
IDS = [c["label"] for c in CASES]


def sha(img):
    return hashlib.sha256(img.tobytes()).hexdigest()


def f32(x):
    return np.asarray(x, dtype=np.float32)


@pytest.fixture(scope="module")
def predictor():
    return GesturePredictor(MODEL_PATH)


def test_config_unchanged():
    assert preprocessing.config() == GOLDEN["config"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_normalize_is_bit_identical(case):
    np.testing.assert_array_equal(preprocessing.normalize(f32(case["raw"])),
                                  f32(case["normalized"]))


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_render_is_byte_identical(case):
    assert sha(preprocessing.render(f32(case["normalized"]))) == case["render_sha256"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_augment_is_bit_identical(case):
    aug = preprocessing.augment(f32(case["normalized"]), np.random.default_rng(1234))
    np.testing.assert_array_equal(aug, f32(case["augmented_seed1234"]))
    assert sha(preprocessing.render(aug)) == case["augmented_render_sha256"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_prediction_matches(case, predictor):
    pred = predictor.predict(case["raw"])
    np.testing.assert_allclose(pred.probs, case["probs"], atol=2e-6)
    assert pred.label == GOLDEN["checkpoint_labels"][int(np.argmax(case["probs"]))]
    assert sha(pred.skeleton) == case["render_sha256"]
