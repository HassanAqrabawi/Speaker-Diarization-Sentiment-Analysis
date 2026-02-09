"""
Backend processing pipeline (UI-agnostic).

This file contains the orchestration previously located in `advanced_topaz.py`:
- model initialization
- audio processing pipeline
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

from config import Config
from utils import (
    validate_audio_file,
    get_audio_duration,
    downsample_audio,
    cleanup_temp_files,
    format_timestamp,
    extract_conversation_metrics,
)

from core_logic.diarization import SpeakerDiarizer
from core_logic.sentiment import AdvancedSentimentAnalyzer
from core_logic.analytics import ConversationAnalytics
from core_logic.youtube_downloader import YouTubeDownloader, YouTubeDownloadError

logger = logging.getLogger(__name__)


class AdvancedTOPAZ:
    """Advanced TOPAZ system with comprehensive analysis capabilities (backend)."""

    def __init__(self):
        Config.create_directories()
        self.diarizer: Optional[SpeakerDiarizer] = None
        self.sentiment_analyzer: Optional[AdvancedSentimentAnalyzer] = None
        self.analytics = ConversationAnalytics()
        self.youtube_downloader = YouTubeDownloader()
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize AI models with error handling (no UI exceptions here)."""
        try:
            logger.info("Initializing Speaker Diarizer...")
            # Default to faster Whisper model; can be tuned later.
            self.diarizer = SpeakerDiarizer(
                num_speakers=Config.DEFAULT_NUM_SPEAKERS,
                whisper_model_size="base",
            )

            logger.info("Initializing Sentiment Analyzer...")
            self.sentiment_analyzer = AdvancedSentimentAnalyzer()

            logger.info("All models initialized successfully")
        except Exception as e:
            logger.exception("Error initializing models")
            raise RuntimeError(Config.ERROR_MESSAGES["model_loading_failed"]) from e

    def process_audio_file(
        self,
        file_path: str,
        include_timestamps: bool = True,
        num_speakers: int | None = None,
        whisper_model_size: str | None = None,
    ) -> Dict:
        """
        Process an audio file with comprehensive analysis.

        Returns a result dict (data), with no Gradio/markdown formatting.
        """
        try:
            if not self.diarizer or not self.sentiment_analyzer:
                raise RuntimeError("Models not initialized")

            # Validate audio file
            is_valid, error_msg = validate_audio_file(file_path)
            if not is_valid:
                raise ValueError(f"Invalid audio file: {error_msg}")

            # Check audio duration
            duration = get_audio_duration(file_path)
            if duration > Config.MAX_AUDIO_DURATION:
                raise ValueError(Config.ERROR_MESSAGES["file_too_large"])

            # Downsample audio for better performance (only if needed)
            # Note: Gradio uploads files to temp directory - this is normal behavior
            # We process from that location, and only create additional temp files if resampling is needed
            processed_file = downsample_audio(file_path)
            
            # Log file paths for debugging
            logger.info(f"Processing audio file: {file_path}")
            logger.info(f"Processed file path: {processed_file}")

            # Perform speaker diarization
            logger.info("Starting speaker diarization...")
            
            # Reinitialize diarizer with new model size if requested
            if whisper_model_size and whisper_model_size != "base":
                logger.info(f"Reinitializing diarizer with Whisper model: {whisper_model_size}")
                from core_logic.diarization import SpeakerDiarizer
                self.diarizer = SpeakerDiarizer(
                    num_speakers=Config.DEFAULT_NUM_SPEAKERS,
                    whisper_model_size=whisper_model_size,
                )
            
            if num_speakers is not None and num_speakers > 0:
                self.diarizer.num_speakers = num_speakers
            else:
                # Auto-detect: start with 2 speakers, may collapse to 1
                self.diarizer.num_speakers = 2

            segments = self.diarizer.diarize_segments(processed_file)

            # Perform sentiment analysis
            logger.info("Starting sentiment analysis...")
            sentiment_analysis = self.sentiment_analyzer.analyze_conversation_sentiment(segments)

            # Perform quality analysis
            logger.info("Starting quality analysis...")
            quality_metrics = self.analytics.analyze_conversation_quality(segments, sentiment_analysis)

            # Generate visualizations
            visualizations = self.analytics.create_visualizations(segments, sentiment_analysis, quality_metrics)

            # Assemble report data (no markdown here)
            report = self._create_report_data(
                segments=segments,
                sentiment_analysis=sentiment_analysis,
                quality_metrics=quality_metrics,
                visualizations=visualizations,
                include_timestamps=include_timestamps,
            )

            cleanup_temp_files()
            return report
        except Exception:
            logger.exception("Error processing audio file")
            cleanup_temp_files()
            raise

    def _create_report_data(
        self,
        *,
        segments: List[Dict],
        sentiment_analysis: Dict,
        quality_metrics: Dict,
        visualizations: Dict,
        include_timestamps: bool,
    ) -> Dict:
        """Create backend report data (presentation handled in UI layer)."""
        conversation_metrics = extract_conversation_metrics(segments)

        # Keep pre-formatted transcript data for UI use, but do not emit markdown.
        transcript_lines: List[Dict] = []
        for segment in segments:
            transcript_lines.append(
                {
                    "speaker": segment.get("speaker", "Unknown"),
                    "text": segment.get("text", ""),
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "start_label": format_timestamp(segment.get("start", 0))
                    if include_timestamps
                    else None,
                }
            )

        # Sentiment analyzer still provides a human-readable report string; keep it as data.
        sentiment_report = self.sentiment_analyzer.generate_sentiment_report(sentiment_analysis)

        return {
            "transcript_lines": transcript_lines,
            "segments": segments,
            "conversation_metrics": conversation_metrics,
            "sentiment_analysis": sentiment_analysis,
            "sentiment_report": sentiment_report,
            "quality_metrics": quality_metrics,
            "visualizations": visualizations,
            "recommendations": quality_metrics.get("recommendations", []),
            "timestamp": datetime.now().isoformat(),
        }

    def process_youtube_video(
        self,
        youtube_url: str,
        include_timestamps: bool = True,
        num_speakers: int | None = None,
        whisper_model_size: str | None = None,
    ) -> Dict:
        """
        Process YouTube video with comprehensive analysis.

        Args:
            youtube_url: YouTube video URL (various formats supported).
            include_timestamps: Whether to include timestamps in output.
            num_speakers: Number of speakers to detect (None for auto-detect).

        Returns:
            Dictionary with analysis results including video metadata.

        Raises:
            YouTubeDownloadError: If download fails after all retry attempts.
            ValueError: If audio file is invalid.
            RuntimeError: If processing fails.
        """
        downloaded_path = None
        converted_path = None

        try:
            # Download YouTube audio
            logger.info(f"Starting YouTube video processing: {youtube_url}")
            downloaded_path, video_metadata = self.youtube_downloader.download_youtube_audio(youtube_url)
            logger.info(f"Downloaded: {downloaded_path}")

            # Convert to WAV if needed
            from utils import convert_to_wav_if_needed
            converted_path = convert_to_wav_if_needed(downloaded_path)
            logger.info(f"Audio prepared for processing: {converted_path}")

            # Process the audio file using existing pipeline
            result = self.process_audio_file(
                file_path=converted_path,
                include_timestamps=include_timestamps,
                num_speakers=num_speakers,
                whisper_model_size=whisper_model_size,
            )

            # Add video metadata to result
            result['video_info'] = {
                'title': video_metadata.get('title'),
                'author': video_metadata.get('author'),
                'length': video_metadata.get('length'),
                'views': video_metadata.get('views'),
                'description': video_metadata.get('description'),
                'publish_date': video_metadata.get('publish_date'),
                'url': youtube_url,
            }

            logger.info("YouTube video processing completed successfully")
            return result

        except YouTubeDownloadError as e:
            logger.error(f"YouTube download failed: {e}")
            raise
        except Exception as e:
            logger.exception("Error processing YouTube video")
            raise RuntimeError(f"YouTube video processing failed: {e}") from e
        finally:
            # Cleanup downloaded files
            try:
                if downloaded_path and os.path.exists(downloaded_path):
                    os.unlink(downloaded_path)
                    logger.info(f"Cleaned up: {downloaded_path}")
                if converted_path and converted_path != downloaded_path and os.path.exists(converted_path):
                    os.unlink(converted_path)
                    logger.info(f"Cleaned up: {converted_path}")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")

