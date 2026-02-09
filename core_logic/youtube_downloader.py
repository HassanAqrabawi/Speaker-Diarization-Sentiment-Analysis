"""
Robust YouTube video downloader with fallback strategies (UI-agnostic).

This module provides reliable YouTube audio downloading with:
- URL normalization and validation
- Primary downloader: pytube
- Fallback downloader: yt-dlp
- Retry logic with exponential backoff
- Detailed diagnostic logging
"""

import os
import re
import tempfile
import time
import subprocess
import logging
from typing import Optional, Dict, Tuple
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


class YouTubeDownloadError(Exception):
    """Custom exception for YouTube download failures."""
    pass


class YouTubeDownloader:
    """Robust YouTube downloader with multiple strategies."""

    def __init__(self):
        """Initialize YouTube downloader."""
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def normalize_youtube_url(self, url: str) -> str:
        """
        Normalize YouTube URL to canonical format.

        Args:
            url: Raw YouTube URL (various formats supported).

        Returns:
            Normalized URL in format: https://www.youtube.com/watch?v=VIDEO_ID

        Raises:
            ValueError: If URL is invalid or video ID cannot be extracted.
        """
        url = url.strip()

        # Extract video ID from various YouTube URL formats
        video_id = None

        # Pattern 1: youtube.com/watch?v=VIDEO_ID
        if 'youtube.com/watch' in url:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            video_id = query_params.get('v', [None])[0]

        # Pattern 2: youtu.be/VIDEO_ID
        elif 'youtu.be/' in url:
            parsed = urlparse(url)
            video_id = parsed.path.strip('/')
            # Remove any query parameters
            if '?' in video_id:
                video_id = video_id.split('?')[0]

        # Pattern 3: youtube.com/embed/VIDEO_ID
        elif 'youtube.com/embed/' in url:
            video_id = url.split('youtube.com/embed/')[1].split('?')[0].split('/')[0]

        # Pattern 4: youtube.com/v/VIDEO_ID
        elif 'youtube.com/v/' in url:
            video_id = url.split('youtube.com/v/')[1].split('?')[0].split('/')[0]

        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {url}")

        # Remove any remaining query parameters or fragments
        video_id = video_id.split('&')[0].split('#')[0]

        # Validate video ID format (11 characters, alphanumeric and -_)
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            raise ValueError(f"Invalid YouTube video ID format: {video_id}")

        # Return canonical URL
        canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"Normalized URL: {url} -> {canonical_url}")
        return canonical_url

    def download_with_pytube(self, url: str, output_dir: str) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Download YouTube audio using pytube.

        Args:
            url: Normalized YouTube URL.
            output_dir: Directory to save downloaded file.

        Returns:
            Tuple of (file_path, metadata_dict) if successful, (None, None) otherwise.
        """
        try:
            import pytube
            logger.info(f"Attempting download with pytube (version: {pytube.__version__})")
            logger.info(f"URL: {url}")

            # Create YouTube object
            yt = pytube.YouTube(url)
            logger.info(f"YouTube object created. Title: {yt.title}")

            # Filter for audio-only streams
            audio_streams = yt.streams.filter(only_audio=True)
            logger.info(f"Found {len(audio_streams)} audio streams")

            if not audio_streams:
                logger.error("No audio streams found")
                return None, None

            # Get the first audio stream (usually best quality)
            stream = audio_streams.first()
            logger.info(f"Selected stream: {stream}")

            # Download
            logger.info(f"Downloading to: {output_dir}")
            downloaded_path = stream.download(output_path=output_dir)

            if not os.path.exists(downloaded_path):
                logger.error(f"Download completed but file not found: {downloaded_path}")
                return None, None

            logger.info(f"Download successful: {downloaded_path}")

            # Extract metadata
            metadata = {
                'title': yt.title,
                'author': yt.author,
                'length': yt.length,
                'views': getattr(yt, 'views', None),
                'description': getattr(yt, 'description', None),
                'publish_date': str(getattr(yt, 'publish_date', None)),
            }

            return downloaded_path, metadata

        except ImportError:
            logger.warning("pytube not installed")
            return None, None
        except Exception as e:
            logger.error(f"pytube download failed: {type(e).__name__}: {e}")
            logger.error(f"Full exception details: {repr(e)}")
            # Log any response text if available
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error(f"HTTP response text: {e.response.text[:500]}")
            return None, None

    def download_with_ytdlp(self, url: str, output_dir: str) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Download YouTube audio using yt-dlp (fallback).

        Args:
            url: Normalized YouTube URL.
            output_dir: Directory to save downloaded file.

        Returns:
            Tuple of (file_path, metadata_dict) if successful, (None, None) otherwise.
        """
        try:
            import yt_dlp
            logger.info(f"Attempting download with yt-dlp (version: {yt_dlp.version.__version__})")
            logger.info(f"URL: {url}")

            output_template = os.path.join(output_dir, '%(id)s.%(ext)s')

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_template,
                'quiet': False,
                'no_warnings': False,
                'extract_flat': False,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'm4a',
                    'preferredquality': '192',
                }],
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info without download first (for diagnostics)
                info = ydl.extract_info(url, download=False)
                logger.info(f"Video info extracted. Title: {info.get('title')}")

                # Download
                info = ydl.extract_info(url, download=True)
                video_id = info.get('id')

                # Find the downloaded file
                possible_extensions = ['m4a', 'webm', 'mp4', 'mp3']
                downloaded_path = None

                for ext in possible_extensions:
                    candidate = os.path.join(output_dir, f"{video_id}.{ext}")
                    if os.path.exists(candidate):
                        downloaded_path = candidate
                        break

                if not downloaded_path:
                    logger.error(f"yt-dlp: Download completed but file not found in {output_dir}")
                    logger.error(f"Directory contents: {os.listdir(output_dir)}")
                    return None, None

                logger.info(f"yt-dlp download successful: {downloaded_path}")

                # Extract metadata
                metadata = {
                    'title': info.get('title'),
                    'author': info.get('uploader'),
                    'length': info.get('duration'),
                    'views': info.get('view_count'),
                    'description': info.get('description'),
                    'publish_date': info.get('upload_date'),
                }

                return downloaded_path, metadata

        except ImportError:
            logger.warning("yt-dlp not installed. Install with: pip install yt-dlp")
            return None, None
        except Exception as e:
            logger.error(f"yt-dlp download failed: {type(e).__name__}: {e}")
            logger.error(f"Full exception details: {repr(e)}")
            return None, None

    def download_youtube_audio(self, url: str) -> Tuple[str, Dict]:
        """
        Download YouTube audio with robust fallback strategy.

        Args:
            url: YouTube URL (will be normalized).

        Returns:
            Tuple of (file_path, metadata_dict).

        Raises:
            YouTubeDownloadError: If all download attempts fail.
        """
        # Normalize URL
        try:
            normalized_url = self.normalize_youtube_url(url)
        except ValueError as e:
            raise YouTubeDownloadError(f"Invalid YouTube URL: {e}")

        # Create temporary directory for download
        temp_dir = tempfile.mkdtemp(prefix="topaz_youtube_")
        logger.info(f"Created temp directory: {temp_dir}")

        # Strategy 1: Try pytube with retries
        for attempt in range(self.max_retries):
            if attempt > 0:
                wait_time = self.retry_delay * (2 ** (attempt - 1))
                logger.info(f"Retry attempt {attempt + 1}/{self.max_retries} after {wait_time}s")
                time.sleep(wait_time)

            logger.info(f"Attempt {attempt + 1}: Trying pytube...")
            file_path, metadata = self.download_with_pytube(normalized_url, temp_dir)

            if file_path and metadata:
                return file_path, metadata

        # Strategy 2: Try yt-dlp as fallback
        logger.info("pytube failed all attempts. Trying yt-dlp fallback...")
        file_path, metadata = self.download_with_ytdlp(normalized_url, temp_dir)

        if file_path and metadata:
            return file_path, metadata

        # All strategies failed
        logger.error(f"All download strategies failed for URL: {url}")
        logger.error(f"Temp directory contents: {os.listdir(temp_dir) if os.path.exists(temp_dir) else 'N/A'}")

        raise YouTubeDownloadError(
            f"Failed to download YouTube video after all attempts. "
            f"Original URL: {url}, Normalized URL: {normalized_url}. "
            f"Both pytube and yt-dlp failed. Check logs for details."
        )


__all__ = ["YouTubeDownloader", "YouTubeDownloadError"]
