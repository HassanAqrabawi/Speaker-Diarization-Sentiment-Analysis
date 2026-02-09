"""
TOPAZ Gradio UI (presentation layer only).

Simple, readable, and Gradio-native layout.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from ui.handlers import (
    download_json_report,
    process_audio_interface,
    process_youtube_interface,
)

APP_TITLE = "TOPAZ - Speaker Diarization & Sentiment Analysis"
MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v3"]

AUDIO_PLACEHOLDER_MD = (
    "### Ready to analyze\n\n"
    "Upload an audio file and click **Analyze Audio** to see:\n"
    "- Transcript with speaker labels\n"
    "- Sentiment analysis\n"
    "- Quality metrics"
)

YOUTUBE_PLACEHOLDER_MD = (
    "### Ready to analyze\n\n"
    "Paste a YouTube URL and click **Analyze YouTube Video**."
)

VIZ_PLACEHOLDER_HTML = (
    "<div style='padding: 1rem; color: #1f2937;'>"
    "Visualizations will appear here after analysis."
    "</div>"
)


def _load_css() -> str:
    css_path = Path(__file__).parent / "css" / "style.css"
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""


def _prepare_download(report: dict | None):
    """Return a visible file update so download reliably appears."""
    path = download_json_report(report)
    return gr.update(value=path, visible=True)


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        css=_load_css(),
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
        ),
    ) as demo:
        report_state = gr.State(None)

        gr.Markdown(f"# {APP_TITLE}")

        with gr.Tabs():
            with gr.Tab("Audio File"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Audio File",
                            streaming=False,
                            autoplay=False,
                        )
                        num_speakers = gr.Slider(
                            minimum=0,
                            maximum=10,
                            step=1,
                            value=0,
                            label="Number of Speakers (0 = Auto-detect)",
                        )
                        model_size = gr.Dropdown(
                            choices=MODEL_SIZES,
                            value="base",
                            label="Whisper Model Size",
                        )
                        include_timestamps = gr.Checkbox(
                            label="Include Timestamps",
                            value=True,
                        )
                        analyze_btn = gr.Button("Analyze Audio", variant="primary")

                    with gr.Column(scale=2):
                        output_md = gr.Markdown(
                            value=AUDIO_PLACEHOLDER_MD,
                            label="Transcript and Metrics",
                        )
                        output_html = gr.HTML(
                            value=VIZ_PLACEHOLDER_HTML,
                            label="Visualizations",
                        )
                        with gr.Row():
                            download_audio_btn = gr.Button("Download JSON Report")
                            download_audio_file = gr.File(
                                label="JSON Report",
                                visible=True,
                            )

            with gr.Tab("YouTube Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        youtube_url = gr.Textbox(
                            lines=1,
                            label="YouTube URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                        )
                        youtube_timestamps = gr.Checkbox(
                            label="Include Timestamps",
                            value=True,
                        )
                        youtube_btn = gr.Button(
                            "Analyze YouTube Video",
                            variant="primary",
                        )

                    with gr.Column(scale=2):
                        youtube_md = gr.Markdown(
                            value=YOUTUBE_PLACEHOLDER_MD,
                            label="Transcript and Metrics",
                        )
                        youtube_html = gr.HTML(
                            value=VIZ_PLACEHOLDER_HTML,
                            label="Visualizations",
                        )
                        with gr.Row():
                            download_youtube_btn = gr.Button("Download JSON Report")
                            download_youtube_file = gr.File(
                                label="JSON Report",
                                visible=True,
                            )

        analyze_btn.click(
            fn=process_audio_interface,
            inputs=[audio_input, include_timestamps, num_speakers, model_size],
            outputs=[output_md, output_html, report_state],
            show_progress="full",
        )

        youtube_btn.click(
            fn=process_youtube_interface,
            inputs=[youtube_url, youtube_timestamps],
            outputs=[youtube_md, youtube_html, report_state],
            show_progress="full",
        )

        download_audio_btn.click(
            fn=_prepare_download,
            inputs=[report_state],
            outputs=[download_audio_file],
            show_progress="hidden",
        )

        download_youtube_btn.click(
            fn=_prepare_download,
            inputs=[report_state],
            outputs=[download_youtube_file],
            show_progress="hidden",
        )

    return demo
