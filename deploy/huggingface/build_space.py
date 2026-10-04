"""
Assemble the folder that becomes the Hugging Face Space for the API.

    python deploy/huggingface/build_space.py out/space

The Space uses the free Gradio SDK (Docker Spaces need a paid plan). A Gradio Space installs
requirements.txt (+ apt packages from packages.txt), then runs app.py; its settings (SDK
version, Python version) come from the front matter of its README.md.
"""

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# from the repository, at the same relative paths
FILES = ["requirements.txt", "alembic.ini", "gestureflow", "backend",
         "models/gesture_cnn.pt", "docs/evaluation_report.json"]
# from this folder, to the Space's root
SPACE_FILES = ["app.py", "README.md", "packages.txt"]


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for rel in FILES:
        src, dst = REPO / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    for name in SPACE_FILES:
        shutil.copy2(HERE / name, out / name)
    print(f"built Space in {out}")


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "out/space"))
