"""The ONNX model used by Edge mode must match the PyTorch checkpoint it was exported from."""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from gestureflow.core import preprocessing
from gestureflow.core.inference import GesturePredictor
from gestureflow.paths import MODEL_PATH, ONNX_META_PATH, ONNX_PATH

GOLDEN = json.loads((Path(__file__).parent / "golden" / "core_golden.json").read_text())


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])


def test_export_is_from_the_current_checkpoint():
    meta = json.loads(ONNX_META_PATH.read_text())
    current = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    assert meta["source_checkpoint_sha256"] == current, \
        "models/gesture_cnn.pt changed: re-run `python -m gestureflow.export_onnx`"
    assert meta["preprocessing"] == preprocessing.config()
    assert meta["labels"] == GesturePredictor(MODEL_PATH).labels


def test_onnx_matches_pytorch_on_golden_samples(session):
    x = np.stack([
        preprocessing.to_input(preprocessing.render(np.asarray(c["normalized"], np.float32)))
        for c in GOLDEN["cases"]])
    logits = session.run(["logits"], {"skeleton": x})[0]
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    np.testing.assert_allclose(probs, [c["probs"] for c in GOLDEN["cases"]], atol=1e-5)
