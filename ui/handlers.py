"""
UI handlers (Gradio callbacks).

These functions adapt UI inputs to backend pipeline calls and format outputs
as markdown/HTML for display. Includes restart, theme toggle, and JSON download.
"""

from __future__ import annotations

import os
import json
import tempfile
from typing import Tuple, Optional, Any
from datetime import datetime

import gradio as gr

from config import Config
from core_logic.pipeline import AdvancedTOPAZ


# ============================================
# SINGLETON PIPELINE (reused across requests)
# ============================================

_pipeline_singleton: AdvancedTOPAZ | None = None


def _get_pipeline() -> AdvancedTOPAZ:
    """Get or create the pipeline singleton."""
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = AdvancedTOPAZ()
    return _pipeline_singleton


# ============================================
# FORMATTING UTILITIES
# ============================================

def _format_transcript_markdown(report: dict) -> str:
    """Convert backend transcript data to formatted Markdown."""
    lines = report.get("transcript_lines", []) or []
    md = "## 📝 Conversation Transcript\n\n"
    
    if not lines:
        return md + "*No transcript available*\n"
    
    for line in lines:
        speaker = line.get("speaker", "Unknown")
        text = line.get("text", "")
        start_label = line.get("start_label")
        
        if start_label:
            md += f"**[{start_label}] {speaker}:** {text}\n\n"
        else:
            md += f"**{speaker}:** {text}\n\n"
    
    return md


def _create_quality_summary_markdown(quality_metrics: dict) -> str:
    """Convert quality metrics to formatted Markdown."""
    quality_score = quality_metrics.get("quality_score", 0)
    
    summary = "## ⭐ Conversation Quality Analysis\n\n"
    summary += f"**Overall Quality Score: {quality_score:.1f}/100**\n\n"
    
    flow = quality_metrics.get("conversation_flow", {}) or {}
    if flow:
        summary += "### 🔄 Conversation Flow\n"
        summary += f"- **Speaker Turns:** {flow.get('speaker_turns', 0)}\n"
        summary += f"- **Turns per Minute:** {flow.get('turns_per_minute', 0):.1f}\n"
        num_speakers = flow.get('num_speakers', 0)
        if num_speakers:
            summary += f"- **Speakers Detected:** {num_speakers}\n"
        total_dur = flow.get('total_duration', 0)
        if total_dur:
            mins = int(total_dur // 60)
            secs = int(total_dur % 60)
            summary += f"- **Duration:** {mins}m {secs}s\n"
        summary += "\n"
    
    recommendations = quality_metrics.get("recommendations", []) or []
    if recommendations:
        summary += "### 💡 Recommendations\n"
        for rec in recommendations:
            summary += f"- {rec}\n"
    
    return summary


def _format_visualizations_html(report: dict) -> str:
    """Convert visualizations to HTML for display."""
    viz = report.get("visualizations", {}) or {}
    
    viz_html_parts = [
        f'<div class="card fade-in" style="margin: 20px 0;">{v}</div>'
        for v in viz.values()
        if isinstance(v, str) and v.strip()
    ]
    
    if viz_html_parts:
        return "\n".join(viz_html_parts)
    
    return (
        '<div class="card" style="text-align: center; padding: 3rem; color: var(--text-muted);">'
        '📊 Visualizations will appear here after analysis'
        '</div>'
    )


# ============================================
# MAIN PROCESSING HANDLERS
# ============================================

def process_audio_interface(
    audio_file: Optional[str],
    include_timestamps: bool,
    num_speakers: int,
    model_size: str,
) -> Tuple[str, str, dict]:
    """
    Gradio interface for audio processing.
    
    Returns:
        (markdown_output, html_output, report_dict)
    """
    if audio_file is None:
        raise gr.Error(Config.ERROR_MESSAGES["no_audio"])
    
    try:
        # Call backend engine
        report = _get_pipeline().process_audio_file(
            audio_file,
            include_timestamps=include_timestamps,
            num_speakers=num_speakers if num_speakers > 0 else None,
            whisper_model_size=model_size,  # Pass model size from UI
        )
        
        # Add metadata
        report["processing_metadata"] = {
            "model_size": model_size,
            "timestamp": datetime.now().isoformat(),
            "file_path": audio_file,
        }
        
    except Exception as e:
        raise gr.Error(f"Processing failed: {e}") from e
    
    # Format for UI display
    transcript_md = _format_transcript_markdown(report)
    sentiment_report = report.get("sentiment_report", "")
    quality_summary = _create_quality_summary_markdown(
        report.get("quality_metrics", {}) or {}
    )
    
    output_md = f"""# 📄 Analysis Results

{transcript_md}

---

# 💭 Sentiment Analysis

{sentiment_report}

---

{quality_summary}
"""
    
    output_html = _format_visualizations_html(report)
    
    return output_md, output_html, report


def process_youtube_interface(
    youtube_url: str,
    include_timestamps: bool,
) -> Tuple[str, str, dict]:
    """
    Gradio interface for YouTube processing.
    
    Returns:
        (markdown_output, html_output, report_dict)
    """
    if not youtube_url:
        raise gr.Error("Please provide a YouTube URL")
    
    try:
        # Use the robust core_logic implementation with fallback downloader
        report = _get_pipeline().process_youtube_video(
            youtube_url=youtube_url,
            include_timestamps=include_timestamps,
            num_speakers=None,  # Auto-detect
        )
        
        # Add processing metadata
        report["processing_metadata"] = {
            "source": "youtube",
            "timestamp": datetime.now().isoformat(),
        }
        
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"YouTube processing failed: {e}") from e
    
    # Format for UI display
    transcript_md = _format_transcript_markdown(report)
    sentiment_report = report.get("sentiment_report", "")
    quality_summary = _create_quality_summary_markdown(
        report.get("quality_metrics", {}) or {}
    )
    
    video_info = report.get("video_info", {})
    video_header = ""
    if video_info:
        video_header = f"""## 📺 Video Information

- **Title:** {video_info.get('title', 'N/A')}
- **Author:** {video_info.get('author', 'N/A')}
- **Length:** {video_info.get('length', 0) // 60}m {video_info.get('length', 0) % 60}s
- **Views:** {video_info.get('views', 0):,}

---

"""
    
    output_md = f"""# 📄 Analysis Results

{video_header}

{transcript_md}

---

# 💭 Sentiment Analysis

{sentiment_report}

---

{quality_summary}
"""
    
    output_html = _format_visualizations_html(report)
    
    return output_md, output_html, report


# ============================================
# UI CONTROL HANDLERS
# ============================================

def restart_ui() -> Tuple[Any, ...]:
    """
    Reset all UI components to initial state.
    
    Returns tuple of reset values for all components:
    (audio_input, youtube_input, output_text, output_html, 
     youtube_output, youtube_html, report_state, 
     num_speakers, model_size, include_timestamps, youtube_timestamps)
    """
    placeholder_ready = (
        "📋 **Ready to analyze**\n\n"
        "Upload an audio file to see:\n"
        "- Full transcription with speaker labels\n"
        "- Sentiment analysis\n"
        "- Quality metrics"
    )
    
    placeholder_youtube = (
        "📺 **Ready to analyze**\n\n"
        "Paste a YouTube URL to extract and analyze the audio content."
    )
    
    placeholder_viz = (
        '<div class="card" style="text-align: center; padding: 3rem; color: var(--text-muted);">'
        '📊 Visualizations will appear here after analysis'
        '</div>'
    )
    
    return (
        None,  # audio_input
        "",  # youtube_input
        placeholder_ready,  # output_text
        placeholder_viz,  # output_html
        placeholder_youtube,  # youtube_output
        placeholder_viz,  # youtube_html
        None,  # report_state
        0,  # num_speakers
        "base",  # model_size
        True,  # include_timestamps
        True,  # youtube_timestamps
    )


def download_json_report(report: Optional[dict]) -> Optional[str]:
    """
    Create downloadable JSON file from report data.
    
    Args:
        report: Report dictionary from processing
    
    Returns:
        Path to temporary JSON file, or None if no report
    """
    if not report:
        raise gr.Error("No report available. Please analyze audio first.")
    
    try:
        # Create temporary JSON file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        )
        
        # Write formatted JSON
        json.dump(report, temp_file, indent=2, ensure_ascii=False)
        temp_file.close()
        
        return temp_file.name
        
    except Exception as e:
        raise gr.Error(f"Failed to create JSON report: {e}") from e
