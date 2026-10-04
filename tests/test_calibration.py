"""Personal calibration: prototype building, blending, and its effect on a recognition stream."""

import json
from pathlib import Path

import numpy as np
import pytest

from gestureflow.core.calibration import Calibration, build_prototypes
from gestureflow.core.inference import GesturePredictor
from gestureflow.core.session import RecognitionSession
from gestureflow.paths import MODEL_PATH

CASES = {c["label"]: c for c in
         json.loads((Path(__file__).parent / "golden" / "core_golden.json").read_text())["cases"]}


@pytest.fixture(scope="module")
def predictor():
    return GesturePredictor(MODEL_PATH)


def embeddings(predictor, swap=()):
    """3 copies of each letter's golden embedding; `swap` pairs exchange two letters'."""
    emb = {label: predictor.predict(c["raw"]).embedding for label, c in CASES.items()}
    for a, b in swap:
        emb[a], emb[b] = emb[b], emb[a]
    return {label: [e, e, e] for label, e in emb.items()}


def test_prototypes_are_unit_length_in_label_order(predictor):
    protos = build_prototypes(embeddings(predictor), predictor.labels)
    assert protos.shape == (26, 128)
    np.testing.assert_allclose(np.linalg.norm(protos, axis=1), 1, atol=1e-6)
    a = predictor.predict(CASES["A"]["raw"]).embedding
    assert int(np.argmax(protos @ a)) == predictor.labels.index("A")


def test_every_letter_needs_enough_samples(predictor):
    emb = embeddings(predictor)
    emb["K"] = emb["K"][:2]
    del emb["Z"]
    with pytest.raises(ValueError, match="K, Z"):
        build_prototypes(emb, predictor.labels)


def test_blend_extremes(predictor):
    protos = build_prototypes(embeddings(predictor), predictor.labels)
    pred = predictor.predict(CASES["B"]["raw"])
    np.testing.assert_allclose(Calibration(protos, alpha=0).apply(pred.probs, pred.embedding),
                               pred.probs)
    blended = Calibration(protos).apply(pred.probs, pred.embedding)
    assert blended.sum() == pytest.approx(1, abs=1e-6)
    assert predictor.labels[int(np.argmax(blended))] == "B"


def test_calibration_changes_what_the_stream_recognizes(predictor):
    # "calibrate" with A and R swapped: the user's R now looks like their A
    protos = build_prototypes(embeddings(predictor, swap=[("A", "R")]), predictor.labels)
    plain = RecognitionSession(predictor).step(CASES["R"]["raw"])
    session = RecognitionSession(predictor, calibration=Calibration(protos))
    calibrated = session.step(CASES["R"]["raw"])
    assert plain.raw == "R" and not plain.calibrated
    assert calibrated.raw == "A" and calibrated.calibrated


def test_calibration_on_real_frames_still_lets_letters_commit(predictor):
    """Regression: with a soft temperature, calibrated predictions were right but never
    reached the commit threshold, so a calibrated user could not add a single letter. Uses
    real recorded frames (not identical copies, which hide the problem): calibrate on the
    first 5 frames of each letter in day2, test on the rest of day2."""
    import pandas as pd

    from gestureflow.core.temporal import CONF_COMMIT
    from gestureflow.paths import LANDMARKS_CSV

    cols = [f"{a}{i}" for i in range(21) for a in "xy"]
    day2 = pd.read_csv(LANDMARKS_CSV).query("session == 'day2'").sort_values("timestamp")
    calib_rows = day2.groupby("label").head(5).index
    calib, test = day2.loc[calib_rows], day2.drop(calib_rows).groupby("label").head(10)

    def frames(df):
        return df[cols].to_numpy(np.float32).reshape(-1, 21, 2)

    emb = {label: [] for label in predictor.labels}
    for label, raw in zip(calib.label, frames(calib), strict=True):
        emb[label].append(predictor.predict(raw).embedding)
    calibration = Calibration(build_prototypes(emb, predictor.labels))

    plain = calibrated = 0
    for raw in frames(test):
        pred = predictor.predict(raw)
        plain += pred.probs.max() > CONF_COMMIT
        calibrated += calibration.apply(pred.probs, pred.embedding).max() > CONF_COMMIT
    assert calibrated >= plain, \
        f"calibration made fewer frames committable ({calibrated} < {plain})"
