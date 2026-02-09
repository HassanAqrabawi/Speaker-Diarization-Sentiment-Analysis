"""
Backend (UI-agnostic) processing modules for TOPAZ.

This package contains all AI/ML processing logic with no UI dependencies.
"""

from core_logic.diarization import SpeakerDiarizer
from core_logic.sentiment import AdvancedSentimentAnalyzer
from core_logic.analytics import ConversationAnalytics
from core_logic.pipeline import AdvancedTOPAZ
from core_logic.youtube_downloader import YouTubeDownloader, YouTubeDownloadError

__all__ = [
    "SpeakerDiarizer",
    "AdvancedSentimentAnalyzer",
    "ConversationAnalytics",
    "AdvancedTOPAZ",
    "YouTubeDownloader",
    "YouTubeDownloadError",
]
