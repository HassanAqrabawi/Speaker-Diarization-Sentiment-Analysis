"""
TOPAZ application entry point.

Canonical launcher for the Gradio UI. Keep UI construction in `ui/app.py`.
"""

import os

# Avoid Transformers trying to use TensorFlow/Keras in this project.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from config import Config
from ui.app import create_app


def main() -> None:
    Config.create_directories()
    demo = create_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()

