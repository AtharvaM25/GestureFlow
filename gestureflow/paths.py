"""Every file location in one place. Paths are absolute, so scripts work from any folder."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

LANDMARKS_CSV = REPO / "data" / "landmarks.csv"
MODEL_PATH = REPO / "models" / "gesture_cnn.pt"
HAND_LANDMARKER = REPO / "models" / "hand_landmarker.task"   # downloaded by cvzone
EVAL_REPORT = REPO / "docs" / "evaluation_report.json"
PREVIEW_IMAGE = REPO / "docs" / "images" / "preview_mean.png"
ONNX_PATH = REPO / "models" / "gesture_cnn.onnx"            # python -m gestureflow.export_onnx
ONNX_META_PATH = REPO / "models" / "gesture_cnn.json"
EDGE_PARITY_REPORT = REPO / "docs" / "edge_parity.json"     # python -m gestureflow.edge
EDGE_FIXTURE = REPO / "frontend" / "src" / "lib" / "edge-fixture.json"
CALIBRATION_REPORT = REPO / "docs" / "calibration_report.json"   # gestureflow.calibration_eval
