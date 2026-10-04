"""
Export the trained CNN to ONNX for in-browser inference (Edge mode).

    python -m gestureflow.export_onnx

Writes models/gesture_cnn.onnx and models/gesture_cnn.json (labels, preprocessing settings,
and which checkpoint it came from). Re-run after every retrain; tests/test_onnx.py fails if
the ONNX file no longer matches the checkpoint.
"""

import hashlib
import json
import sys

import torch

from gestureflow.core import preprocessing
from gestureflow.core.inference import GesturePredictor
from gestureflow.paths import MODEL_PATH, ONNX_META_PATH, ONNX_PATH


def main():
    # torch.onnx prints emoji progress marks; Windows consoles default to a codepage without them
    sys.stdout.reconfigure(encoding="utf-8")
    predictor = GesturePredictor(MODEL_PATH)
    dummy = torch.zeros(1, 3, preprocessing.CANVAS, preprocessing.CANVAS)
    torch.onnx.export(
        predictor.model, (dummy,), str(ONNX_PATH),
        input_names=["skeleton"], output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=1024)}},
        dynamo=True, external_data=False,
    )
    meta = {
        "labels": predictor.labels,
        "preprocessing": preprocessing.config(),
        "source_checkpoint_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
    }
    ONNX_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.2f} MB) and {ONNX_META_PATH}")


if __name__ == "__main__":
    main()
