"""
Entry point for the Hugging Face Space (Gradio SDK, free CPU hardware).

Docker Spaces need a paid plan, but Gradio Spaces are free and simply run `python app.py`.
This runs the same FastAPI app as everywhere else, with a small Gradio status page mounted
at /demo, on port 7860 (the port Spaces expect). Database tables are created or updated
first, as the Docker image does with `alembic upgrade head`.
"""

import os
from pathlib import Path

import gradio as gr
import uvicorn
from alembic import command
from alembic.config import Config

HERE = Path(__file__).resolve().parent

command.upgrade(Config(str(HERE / "alembic.ini")), "head")

from backend.main import create_app  # noqa: E402  (after migrations, so the schema is ready)

api = create_app()

with gr.Blocks(title="GestureFlow API") as status_page:
    gr.Markdown(
        "# GestureFlow API\n"
        "This Space serves the backend of "
        "[GestureFlow](https://github.com/AtharvaM25/Hand_Recognition). "
        "Interactive API docs: [/docs](/docs). Health: [/api/v1/health](/api/v1/health).")

app = gr.mount_gradio_app(api, status_page, path="/demo", ssr_mode=False)

if __name__ == "__main__":
    # one worker: the rate limits are kept in memory (docs/API.md)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")),
                proxy_headers=True, forwarded_allow_ips="*")
