"""
Speaker diarization module (UI-agnostic).

This module provides speaker diarization and transcription capabilities
using Whisper for transcription and speaker embeddings for diarization.
"""

import torch
import torchaudio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import subprocess
import whisper
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """Speaker diarization using Whisper transcription and speaker embeddings."""

    def __init__(self, num_speakers: int = 2, whisper_model_size: str = "base"):
        """
        Initialize the speaker diarizer.

        Args:
            num_speakers: Expected number of speakers in the audio.
            whisper_model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        """
        self.num_speakers = num_speakers
        self.model = whisper.load_model(whisper_model_size)
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    # ------------------------------------------------------------------
    # Audio helpers – use torchaudio directly (avoids pyannote.audio.Audio
    # version-compatibility issues with torchaudio return signatures).
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio(path: str):
        """Load audio with torchaudio, safely handling return values."""
        result = torchaudio.load(path)
        # torchaudio.load returns (waveform, sample_rate).
        # Older versions returned 3 values; handle both.
        if isinstance(result, (tuple, list)):
            waveform = result[0]
            sample_rate = result[1]
        else:
            raise RuntimeError(f"torchaudio.load returned unexpected type: {type(result)}")
        return waveform, sample_rate

    @staticmethod
    def _crop_waveform(waveform: torch.Tensor, sample_rate: int, start: float, end: float) -> torch.Tensor:
        """Crop waveform to [start, end] seconds."""
        total_samples = waveform.shape[-1]
        s = max(0, min(int(start * sample_rate), total_samples - 1))
        e = max(s + 1, min(int(end * sample_rate), total_samples))
        return waveform[..., s:e]

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        """Ensure waveform is mono (1, samples)."""
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            return waveform.mean(dim=0, keepdim=True)
        return waveform

    # ------------------------------------------------------------------

    def segment_embedding(self, segment: Dict, path: str, duration: float) -> np.ndarray:
        """
        Extract speaker embedding for a single segment.

        Uses torchaudio directly for loading / cropping to avoid
        pyannote Audio.crop() incompatibilities.
        """
        start = segment["start"]
        end = min(duration, segment["end"])

        waveform, sample_rate = self._load_audio(path)
        waveform = self._crop_waveform(waveform, sample_rate, start, end)
        waveform = self._to_mono(waveform)

        # embedding_model expects shape (batch, channels, samples)
        return self.embedding_model(waveform.unsqueeze(0))

    def diarize_segments(self, path: str) -> List[Dict]:
        """
        Perform speaker diarization and return structured segments.

        Args:
            path: Path to audio file.

        Returns:
            List of diarized segments with speaker labels, timestamps, and text.
        """
        # Convert to WAV if needed
        if not path.endswith('.wav'):
            subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
            path = 'audio.wav'

        # Whisper transcription
        result = self.model.transcribe(path)
        whisper_segments = result.get("segments", [])

        # Get duration
        waveform, sample_rate = self._load_audio(path)
        duration = waveform.shape[-1] / float(sample_rate)

        if not whisper_segments:
            return []

        # Extract embeddings
        embeddings = np.zeros(shape=(len(whisper_segments), 192))
        for i, seg in enumerate(whisper_segments):
            embeddings[i] = self.segment_embedding(seg, path, duration)

        embeddings = np.nan_to_num(embeddings)

        # Speaker clustering
        labels: Optional[np.ndarray] = None

        if len(embeddings) == 1:
            labels = np.array([0])
        else:
            sim = cosine_similarity(embeddings)
            avg_similarity = (sim.sum() - len(embeddings)) / (len(embeddings) * (len(embeddings) - 1))

            if avg_similarity > 0.85:
                labels = np.zeros(len(embeddings), dtype=int)
            else:
                k = max(2, int(self.num_speakers))
                k = min(k, len(embeddings))
                clustering = AgglomerativeClustering(n_clusters=k).fit(embeddings)
                labels = clustering.labels_
                if len(set(labels)) < 2:
                    labels = np.zeros(len(embeddings), dtype=int)

        # Build structured output
        diarized_segments: List[Dict] = []
        for i, seg in enumerate(whisper_segments):
            diarized_segments.append({
                "speaker": f"SPEAKER {int(labels[i]) + 1}",
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")).strip(),
            })

        return diarized_segments

    def diarize(self, path: str) -> str:
        """
        Backwards-compatible: returns a formatted transcript string.

        Prefer `diarize_segments()` for structured output.
        """
        segments = self.diarize_segments(path)
        if not segments:
            return ""

        def time(secs: float) -> datetime.timedelta:
            return datetime.timedelta(seconds=round(secs))

        transcript = ""
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                transcript += "\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n'
            transcript += (segment["text"] + " ").strip() + " "

        return transcript


__all__ = ["SpeakerDiarizer"]
