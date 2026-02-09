"""
TOPAZ Gradio UI - Modern, Professional Interface

This module creates a clean, colorful SaaS-style UI using external CSS.
All business logic lives in core_logic; this is purely presentation.
"""

from __future__ import annotations

import os
from pathlib import Path
import gradio as gr

from ui.handlers import (
    process_audio_interface,
    process_youtube_interface,
    restart_ui,
    download_json_report,
)

# ============================================
# UI TEXT CONSTANTS (for easy localization)
# ============================================

UI_TEXT = {
    "app_name": "TOPAZ",
    "app_tagline": "Advanced Speaker Diarization & Sentiment Analysis",
    "app_description": "Transform customer service calls with AI-powered insights",
    
    # Features
    "feature_transcription_title": "🎤 Real-time Transcription",
    "feature_transcription_desc": "Advanced speaker diarization with Whisper AI for accurate multi-speaker transcription",
    "feature_sentiment_title": "😊 Sentiment Analysis",
    "feature_sentiment_desc": "Multi-model emotion detection and sentiment trend analysis across conversations",
    "feature_analytics_title": "📊 Quality Analytics",
    "feature_analytics_desc": "Comprehensive conversation quality metrics and actionable insights",
    
    # Tabs
    "tab_audio": "🎵 Audio File",
    "tab_youtube": "📺 YouTube Video",
    "tab_analytics": "📈 Analytics Dashboard",
    
    # Labels
    "label_audio_input": "Upload Audio File",
    "label_youtube_url": "YouTube URL",
    "label_num_speakers": "Number of Speakers (0 = Auto-detect)",
    "label_model_size": "Whisper Model Size",
    "label_timestamps": "Include Timestamps",
    
    # Buttons
    "btn_analyze": "🚀 Analyze Audio",
    "btn_analyze_youtube": "🎬 Analyze YouTube Video",
    "btn_download_json": "💾 Download JSON Report",
    "btn_restart": "🔄 Restart UI",
    
    # Status
    "status_ready": "Ready",
    "status_processing": "Processing...",
    "status_complete": "Complete",
    
    # Placeholders
    "placeholder_ready": "📋 **Ready to analyze**\n\nUpload an audio file to see:\n- Full transcription with speaker labels\n- Sentiment analysis\n- Quality metrics",
    "placeholder_youtube_ready": "📺 **Ready to analyze**\n\nPaste a YouTube URL to extract and analyze the audio content.",
    "placeholder_visualizations": '<div style="text-align: center; padding: 3rem; color: #718096;">📊 Visualizations will appear here after analysis</div>',
    "placeholder_analytics": """
        <div style="text-align: center; padding: 3rem;">
            <h2 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;">
                📊 Real-time Analytics Dashboard
            </h2>
            <p style="font-size: 1rem; line-height: 1.6; margin-bottom: 1rem;">
                Upload audio files to see detailed analytics and insights
            </p>
            <p style="font-size: 0.875rem; color: #718096;">
                Features: Sentiment trends • Speaker balance • Quality metrics • Response time analysis
            </p>
        </div>
    """,
}

# Model size options
MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v3"]


# ============================================
# LOAD EXTERNAL CSS
# ============================================

def load_css() -> str:
    """Load external CSS file."""
    css_path = Path(__file__).parent / "css" / "style.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


# ============================================
# UI COMPONENT BUILDERS
# ============================================

def create_navbar() -> gr.HTML:
    """Create professional navbar with theme toggle."""
    return gr.HTML(
        f"""
        <div class="topaz-navbar">
            <div class="topaz-navbar-brand">
                <div>
                    <h1>{UI_TEXT['app_name']}</h1>
                    <div class="subtitle">{UI_TEXT['app_tagline']}</div>
                </div>
            </div>
            <div class="topaz-navbar-controls">
                <div class="topaz-status-badge" id="topaz-status-badge">
                    {UI_TEXT['status_ready']}
                </div>
            </div>
        </div>
        """
    )


def create_hero() -> gr.HTML:
    """Create hero section with gradient background."""
    return gr.HTML(
        f"""
        <div class="topaz-hero fade-in">
            <h1>{UI_TEXT['app_name']}</h1>
            <p>{UI_TEXT['app_tagline']}</p>
            <p style="font-size: 1rem; opacity: 0.9; margin-top: 0.5rem;">
                {UI_TEXT['app_description']}
            </p>
        </div>
        """
    )


def create_feature_cards() -> gr.HTML:
    """Create feature highlight cards."""
    return gr.HTML(
        f"""
        <div class="three-column-grid">
            <div class="feature-card fade-in">
                <div class="feature-card-icon">{UI_TEXT['feature_transcription_title'].split()[0]}</div>
                <h3 class="feature-card-title">{UI_TEXT['feature_transcription_title'][2:]}</h3>
                <p class="feature-card-text">{UI_TEXT['feature_transcription_desc']}</p>
            </div>
            <div class="feature-card fade-in" style="animation-delay: 100ms;">
                <div class="feature-card-icon">{UI_TEXT['feature_sentiment_title'].split()[0]}</div>
                <h3 class="feature-card-title">{UI_TEXT['feature_sentiment_title'][2:]}</h3>
                <p class="feature-card-text">{UI_TEXT['feature_sentiment_desc']}</p>
            </div>
            <div class="feature-card fade-in" style="animation-delay: 200ms;">
                <div class="feature-card-icon">{UI_TEXT['feature_analytics_title'].split()[0]}</div>
                <h3 class="feature-card-title">{UI_TEXT['feature_analytics_title'][2:]}</h3>
                <p class="feature-card-text">{UI_TEXT['feature_analytics_desc']}</p>
            </div>
        </div>
        """
    )


# ============================================
# MAIN APP BUILDER
# ============================================

def create_app() -> gr.Blocks:
    """Create the main Gradio Blocks app with external CSS."""
    
    # Load CSS
    custom_css = load_css()
    
    # Add aggressive inline overrides for Gradio
    custom_css += """
    
    /* CRITICAL OVERRIDES - FORCE HIGH CONTRAST */
    .gradio-container, .gradio-container * {
        color: #000000 !important;
    }
    
    /* Tabs must be visible */
    button[role="tab"] {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border: 2px solid #cbd5e0 !important;
    }
    
    button[role="tab"][aria-selected="true"] {
        background: #667eea !important;
        color: #ffffff !important;
    }
    
    /* All labels black */
    label, .label {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    /* Markdown output must be readable */
    .markdown-body, [data-testid="markdown"] {
        color: #000000 !important;
    }
    
    .markdown-body p, [data-testid="markdown"] p {
        color: #000000 !important;
    }
    """
    
    with gr.Blocks(
        css=custom_css,
        title=f"{UI_TEXT['app_name']} - {UI_TEXT['app_tagline']}",
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="pink",
            neutral_hue="slate",
            text_size="lg",
            font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        ).set(
            body_background_fill="#f0f4f8",
            body_text_color="#000000",
            body_text_color_subdued="#1a202c",
            block_background_fill="#ffffff",
            block_label_background_fill="#ffffff",
            block_label_text_color="#000000",
            block_title_text_color="#000000",
            button_primary_background_fill="#667eea",
            button_primary_text_color="#ffffff",
            input_background_fill="#ffffff",
            input_border_color="#667eea",
            panel_background_fill="#ffffff",
        ),
    ) as demo:
        
        # State management
        report_state = gr.State(None)
        
        # ========== NAVBAR ==========
        create_navbar()
        
        with gr.Column(elem_classes="topaz-container"):
            
            # ========== HERO SECTION ==========
            create_hero()
            
            # ========== FEATURE CARDS ==========
            create_feature_cards()
            
            # ========== CONTROL PANEL ==========
            with gr.Row():
                restart_btn = gr.Button(
                    UI_TEXT['btn_restart'],
                    elem_classes="btn-danger",
                    size="sm",
                )
            
            # ========== MAIN TABS ==========
            with gr.Tabs() as tabs:
                
                # ===== TAB 1: Audio File Analysis =====
                with gr.Tab(UI_TEXT['tab_audio'], id="audio_tab"):
                    with gr.Row():
                        # Left Panel: Inputs
                        with gr.Column(scale=1):
                            gr.HTML('<div class="card-title">Input Settings</div>')
                            
                            audio_input = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label=UI_TEXT['label_audio_input'],
                            )
                            
                            num_speakers = gr.Slider(
                                minimum=0,
                                maximum=10,
                                value=0,
                                step=1,
                                label=UI_TEXT['label_num_speakers'],
                            )
                            
                            model_size = gr.Dropdown(
                                choices=MODEL_SIZES,
                                value="base",
                                label=UI_TEXT['label_model_size'],
                            )
                            
                            include_timestamps = gr.Checkbox(
                                label=UI_TEXT['label_timestamps'],
                                value=True,
                            )
                            
                            analyze_btn = gr.Button(
                                UI_TEXT['btn_analyze'],
                                variant="primary",
                                elem_classes="btn-primary",
                                size="lg",
                            )
                        
                        # Right Panel: Outputs
                        with gr.Column(scale=2):
                            gr.HTML('<div class="card-title">Analysis Results</div>')
                            
                            with gr.Row():
                                with gr.Column(scale=2):
                                    output_text = gr.Markdown(
                                        value=UI_TEXT['placeholder_ready'],
                                        elem_classes="output-markdown",
                                        label="Transcript & Metrics",
                                    )
                                
                                with gr.Column(scale=1):
                                    output_html = gr.HTML(
                                        value=UI_TEXT['placeholder_visualizations'],
                                        label="Visualizations",
                                    )
                            
                            download_btn = gr.Button(
                                UI_TEXT['btn_download_json'],
                                elem_classes="btn-success",
                            )
                            download_file = gr.File(visible=False)
                
                # ===== TAB 2: YouTube Analysis =====
                with gr.Tab(UI_TEXT['tab_youtube'], id="youtube_tab"):
                    with gr.Row():
                        # Left Panel: Inputs
                        with gr.Column(scale=1):
                            gr.HTML('<div class="card-title">YouTube Input</div>')
                            
                            youtube_input = gr.Textbox(
                                lines=1,
                                placeholder="https://www.youtube.com/watch?v=...",
                                label=UI_TEXT['label_youtube_url'],
                            )
                            
                            youtube_timestamps = gr.Checkbox(
                                label=UI_TEXT['label_timestamps'],
                                value=True,
                            )
                            
                            youtube_btn = gr.Button(
                                UI_TEXT['btn_analyze_youtube'],
                                variant="primary",
                                elem_classes="btn-primary",
                                size="lg",
                            )
                        
                        # Right Panel: Outputs
                        with gr.Column(scale=2):
                            gr.HTML('<div class="card-title">Analysis Results</div>')
                            
                            with gr.Row():
                                with gr.Column(scale=2):
                                    youtube_output = gr.Markdown(
                                        value=UI_TEXT['placeholder_youtube_ready'],
                                        elem_classes="output-markdown",
                                        label="Transcript & Metrics",
                                    )
                                
                                with gr.Column(scale=1):
                                    youtube_html = gr.HTML(
                                        value=UI_TEXT['placeholder_visualizations'],
                                        label="Visualizations",
                                    )
                            
                            youtube_download_btn = gr.Button(
                                UI_TEXT['btn_download_json'],
                                elem_classes="btn-success",
                            )
                            youtube_download_file = gr.File(visible=False)
                
                # ===== TAB 3: Analytics Dashboard =====
                with gr.Tab(UI_TEXT['tab_analytics'], id="analytics_tab"):
                    gr.HTML(UI_TEXT['placeholder_analytics'])
            
            # ========== FOOTER ==========
            gr.HTML(
                f"""
                <div style="text-align: center; margin-top: 3rem; padding: 1.5rem; 
                     border-top: 2px solid var(--border-color);">
                    <p style="font-size: 0.875rem; margin: 0.5rem 0;">
                        <strong style="color: var(--primary-600);">{UI_TEXT['app_name']}</strong> 
                        — Powered by Advanced AI for Customer Service Excellence
                    </p>
                    <p style="font-size: 0.75rem; color: var(--text-muted); margin: 0;">
                        Built with ❤️ for better customer experiences
                    </p>
                </div>
                """
            )
        
        # ========== EVENT HANDLERS ==========
        
        # Audio processing
        analyze_btn.click(
            fn=process_audio_interface,
            inputs=[audio_input, include_timestamps, num_speakers, model_size],
            outputs=[output_text, output_html, report_state],
        )
        
        # YouTube processing
        youtube_btn.click(
            fn=process_youtube_interface,
            inputs=[youtube_input, youtube_timestamps],
            outputs=[youtube_output, youtube_html, report_state],
        )
        
        # Download JSON
        download_btn.click(
            fn=download_json_report,
            inputs=[report_state],
            outputs=[download_file],
        )
        
        youtube_download_btn.click(
            fn=download_json_report,
            inputs=[report_state],
            outputs=[youtube_download_file],
        )
        
        # Restart UI
        restart_btn.click(
            fn=restart_ui,
            inputs=[],
            outputs=[
                audio_input,
                youtube_input,
                output_text,
                output_html,
                youtube_output,
                youtube_html,
                report_state,
                num_speakers,
                model_size,
                include_timestamps,
                youtube_timestamps,
            ],
        )
    
    return demo
