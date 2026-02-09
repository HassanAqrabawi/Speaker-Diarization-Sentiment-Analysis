"""
Configuration file for TOPAZ Speaker Diarization Sentiment Analysis
"""
import os
import torch

class Config:
    """Configuration settings for the application"""
    
    # Model settings
    WHISPER_MODEL = "openai/whisper-large-v3"
    DIARIZATION_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
    SENTIMENT_MODEL = "facebook/bart-large-mnli"
    
    # Device settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Audio settings
    TARGET_SAMPLE_RATE = 16000
    MAX_AUDIO_DURATION = 3600  # 1 hour in seconds
    
    # Processing settings
    DEFAULT_NUM_SPEAKERS = 2
    MAX_NUM_SPEAKERS = 10
    
    # File paths
    TEMP_DIR = "temp_audio"
    OUTPUT_DIR = "outputs"
    
    # API settings
    HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
    
    # Error messages
    ERROR_MESSAGES = {
        "no_audio": "No audio file submitted! Please upload an audio file before submitting your request.",
        "invalid_format": "Invalid audio format. Please upload a supported audio file (mp3, wav, m4a, etc.).",
        "file_too_large": f"Audio file is too large. Maximum duration: {MAX_AUDIO_DURATION} seconds.",
        "transcription_failed": "Transcription failed. Please check your audio file and try again.",
        "diarization_failed": "Speaker diarization failed. Please try with a different audio file.",
        "model_loading_failed": "Failed to load AI models. Please check your internet connection and try again."
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
