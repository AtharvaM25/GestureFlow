"""The Hugging Face Space bundle (free Gradio SDK) must be complete and self-consistent."""

import importlib.util
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "build_space", REPO / "deploy" / "huggingface" / "build_space.py")
build_space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_space)


def test_bundle_has_everything_and_space_settings(tmp_path):
    out = tmp_path / "space"
    build_space.build(out)
    for rel in ["app.py", "README.md", "packages.txt", "requirements.txt", "alembic.ini",
                "backend/main.py", "gestureflow/core/inference.py", "models/gesture_cnn.pt",
                "docs/evaluation_report.json", "backend/migrations/versions"]:
        assert (out / rel).exists(), rel
    front = yaml.safe_load((out / "README.md").read_text(encoding="utf-8").split("---")[1])
    assert front["sdk"] == "gradio" and front["app_file"] == "app.py"
    assert front["python_version"] == "3.12"
    # requirements.txt is the pinned file; gradio comes from sdk_version, so it must not be pinned
    reqs = (out / "requirements.txt").read_text()
    assert "torch==" in reqs and "\ngradio==" not in reqs
    assert all(line and not line.startswith("#")
               for line in (out / "packages.txt").read_text().splitlines())
    assert not list(out.rglob("__pycache__"))
    assert not (out / "data").exists() and not (out / "frontend").exists()
