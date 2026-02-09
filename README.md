# TOPAZ – Speaker Diarization & Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**TOPAZ** is an AI-powered system for analyzing conversations: transcribe audio, separate speakers, run sentiment analysis, and compute conversation quality metrics. Use it for customer service calls, meetings, or any multi-speaker audio.

---

## Features

### Speaker diarization
- **Whisper** for transcription (local; no API key required).
- **pyannote.audio** + **SpeechBrain** for speaker embeddings and clustering.
- Configurable number of speakers; optional Whisper model size (tiny → large-v3) for speed vs accuracy.

### Sentiment analysis
- **VADER** (primary), **DistilBERT** (SST-2), and **emotion detection** (Hugging Face) on transcribed text.
- Per-segment and per-speaker sentiment; overall conversation sentiment and volatility.
- Runs locally; no external sentiment API.

### Conversation quality
- **Overall quality score** (0–100) from pacing, turn-taking, and sentiment.
- **Conversation flow**: speaker turns, turns per minute, speakers detected, duration.
- **Recommendations** (e.g., pacing, turn length, negative sentiment) when applicable.

### Inputs
- **Audio file**: WAV, MP3, M4A, FLAC, etc. (upload or microphone in the UI).
- **YouTube**: paste URL; audio is downloaded (pytube with yt-dlp fallback), then processed like an audio file.

### UI
- **Gradio** web app: upload audio or paste YouTube URL, choose options, view transcript, sentiment report, quality summary, and visualizations.
- Optional JSON report download.

---

## Architecture

```
TOPAZ/
├── main.py                 # Entry point: launches Gradio app
├── config.py               # Settings (paths, device, limits)
├── utils.py                # Audio validation, resampling, WAV conversion
├── core_logic/             # Backend engine (no UI dependencies)
│   ├── __init__.py         # Public API
│   ├── pipeline.py         # AdvancedTOPAZ – orchestration
│   ├── diarization.py      # SpeakerDiarizer (Whisper + pyannote)
│   ├── sentiment.py        # AdvancedSentimentAnalyzer
│   ├── analytics.py        # ConversationAnalytics (quality metrics)
│   └── youtube_downloader.py # YouTube audio download (pytube + yt-dlp)
├── ui/
│   ├── app.py              # Gradio layout and tabs
│   ├── handlers.py         # Event handlers, report formatting
│   └── css/style.css       # Custom styles
├── requirements.txt
└── README.md
```

The **core_logic** package is the single backend: all AI and pipeline logic lives there and returns plain dicts/objects. The Gradio UI in **ui/** only formats and displays that data.

---

## Installation

### Prerequisites
- **Python 3.8+**
- **8GB+ RAM** (16GB recommended for larger Whisper models)
- **GPU** (optional but recommended; CUDA for PyTorch)
- **Hugging Face token** (for pyannote/some models; set `HF_TOKEN`)

### Steps

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/TOPAZ.git
   cd TOPAZ
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download language data**
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

5. **Set Hugging Face token** (required for pyannote/some Hugging Face models)
   ```bash
   # Windows (PowerShell):
   $env:HF_TOKEN="your_token_here"
   # Linux/macOS:
   export HF_TOKEN=your_token_here
   ```

6. **Run the app**
   ```bash
   python main.py
   ```
   Open the URL shown (default: http://127.0.0.1:7860).

### Optional: YouTube fallback
For more reliable YouTube downloads, **yt-dlp** is recommended and listed in `requirements.txt`. If pytube fails, the pipeline will try yt-dlp automatically.

---

## Usage

### Web UI
1. Run `python main.py`.
2. **Audio File** tab: upload an audio file (or record), set number of speakers (0 = auto), choose Whisper model size, then click **Analyze Audio**.
3. **YouTube Video** tab: paste a YouTube URL, then click **Analyze YouTube Video**.
4. View transcript, sentiment report, quality analysis, and visualizations; download JSON report if needed.

### Python API (no UI)

Use the **core_logic** pipeline directly:

```python
from core_logic.pipeline import AdvancedTOPAZ

engine = AdvancedTOPAZ()

# Process an audio file
report = engine.process_audio_file(
    "path/to/audio.wav",
    include_timestamps=True,
    num_speakers=None,        # None = auto
    whisper_model_size="base"  # or "tiny", "small", "medium", "large", "large-v3"
)

# Or process a YouTube video
report = engine.process_youtube_video(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    include_timestamps=True,
    num_speakers=None,
)

# report is a dict, e.g.:
# - report["segments"]       – list of {speaker, text, start, end}
# - report["transcript"]     – full transcript string
# - report["sentiment_report"] – text summary
# - report["quality_metrics"] – quality_score, conversation_flow, recommendations
# - report["video_info"]     – (YouTube only) title, etc.
```

No Gradio or UI code is required when using `AdvancedTOPAZ` from `core_logic.pipeline`.

---

## Configuration

- **config.py**: `TARGET_SAMPLE_RATE`, `MAX_AUDIO_DURATION`, `DEFAULT_NUM_SPEAKERS`, `TEMP_DIR`, `OUTPUT_DIR`, etc.
- **Environment**: `HF_TOKEN` (Hugging Face), `CUDA_VISIBLE_DEVICES` (GPU), `TOPAZ_LOG_LEVEL` (optional).

---

## Analytics and metrics

- **Overall quality score (0–100)**  
  Based on conversation flow (pacing, turns per minute), turn-taking efficiency, and sentiment. Single-speaker or very short conversations may score low or zero.

- **Conversation flow**
  - **Speaker turns** – number of speaker changes.
  - **Turns per minute** – speaker changes per minute.
  - **Speakers detected** – number of distinct speakers.
  - **Duration** – total conversation length.

- **Sentiment**
  - Per-speaker and overall averages; positive/negative/neutral ratios; sentiment range and volatility.
  - Based on transcribed text only (no tone of voice).

Metrics that depend on overlapping speech or precise silence (e.g. interruption rate, exact pause length) are not reported, as they are not reliable with Whisper’s sequential segments.

---

## Requirements summary

- **torch**, **torchaudio**, **numpy** – inference and audio
- **openai-whisper** – transcription (local)
- **pyannote.audio**, **speechbrain**, **scikit-learn** – diarization
- **transformers**, **spacy**, **nltk** – sentiment and NLP
- **gradio** – web UI
- **pandas**, **matplotlib**, **seaborn**, **plotly** – analytics and visualizations
- **pytube**, **yt-dlp** – YouTube (pytube primary; yt-dlp fallback)

See **requirements.txt** for versions.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for transcription
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [Hugging Face](https://huggingface.co) for sentiment and emotion models
- [Gradio](https://gradio.app) for the web interface
- [spaCy](https://spacy.io) and [NLTK](https://www.nltk.org/) for NLP

---

**TOPAZ** – speaker diarization and sentiment analysis for conversations.
