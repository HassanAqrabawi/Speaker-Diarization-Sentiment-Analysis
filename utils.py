"""
Utility functions for TOPAZ Speaker Diarization Sentiment Analysis
"""
import os
import tempfile
import logging
import torchaudio
import numpy as np
from typing import Optional, Dict, List, Tuple
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate audio file format and size
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file size (max 100MB)
    file_size = os.path.getsize(file_path)
    if file_size > 100 * 1024 * 1024:  # 100MB
        return False, "File too large (max 100MB)"
    
    # Check file extension
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in valid_extensions:
        return False, f"Unsupported file format: {file_ext}"
    
    return True, ""

def get_audio_duration(file_path: str) -> float:
    """
    Get audio duration in seconds
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Duration in seconds
    """
    try:
        # torchaudio.load() returns (waveform, sample_rate) - always 2 values
        load_result = torchaudio.load(file_path)
        if isinstance(load_result, tuple) and len(load_result) >= 2:
            waveform, sample_rate = load_result[0], load_result[1]
        else:
            raise ValueError(f"Unexpected torchaudio.load() result: {type(load_result)}")
        
        duration = waveform.shape[1] / sample_rate
        return duration
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        return 0.0

def downsample_audio(file_path: str, target_rate: int = 16000) -> str:
    """
    Downsample audio to target sample rate
    
    Args:
        file_path: Path to the audio file
        target_rate: Target sample rate
    
    Returns:
        Path to the downsampled audio file (returns original path if already at target rate)
    """
    try:
        # torchaudio.load() returns (waveform, sample_rate) - always 2 values
        load_result = torchaudio.load(file_path)
        if isinstance(load_result, tuple) and len(load_result) >= 2:
            waveform, original_sample_rate = load_result[0], load_result[1]
        else:
            raise ValueError(f"Unexpected torchaudio.load() result: {type(load_result)}")
        
        # If already at target rate, return original path (no need to create temp file)
        if original_sample_rate == target_rate:
            logger.info(f"Audio already at target rate {target_rate}Hz, skipping resampling")
            return file_path
        
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate, 
            new_freq=target_rate
        )
        waveform = resampler(waveform)
        
        # Save to temporary file only if resampling was needed
        temp_path = os.path.join(Config.TEMP_DIR, f"downsampled_{os.path.basename(file_path)}")
        torchaudio.save(temp_path, waveform, target_rate)
        
        return temp_path
    except Exception as e:
        logger.error(f"Error downsampling audio: {e}")
        return file_path

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        import shutil
        if os.path.exists(Config.TEMP_DIR):
            shutil.rmtree(Config.TEMP_DIR)
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")

def format_timestamp(seconds: float) -> str:
    """
    Format timestamp in MM:SS format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def convert_to_wav_if_needed(file_path: str) -> str:
    """
    Convert an input audio/video file to WAV if it's not already WAV.
    Returns path to WAV file (may be same as input if already WAV).
    """
    try:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == '.wav':
            return file_path
        
        # Load and re-save as WAV with target sample rate
        # torchaudio.load() returns (waveform, sample_rate) - always 2 values
        load_result = torchaudio.load(file_path)
        if isinstance(load_result, tuple) and len(load_result) >= 2:
            waveform, sample_rate = load_result[0], load_result[1]
        else:
            raise ValueError(f"Unexpected torchaudio.load() result: {type(load_result)}")
        
        target_path = os.path.splitext(file_path)[0] + '.wav'
        torchaudio.save(target_path, waveform, sample_rate)
        return target_path
    except Exception as e:
        logger.error(f"Error converting file to WAV: {e}")
        # Fallback to original path so caller can report a useful error
        return file_path

def calculate_speaking_time(segments: List[Dict]) -> Dict[str, float]:
    """
    Calculate speaking time for each speaker
    
    Args:
        segments: List of speaker segments
        
    Returns:
        Dictionary with speaker speaking times
    """
    speaking_times = {}
    
    for segment in segments:
        speaker = segment.get('speaker', 'Unknown')
        duration = segment.get('end', 0) - segment.get('start', 0)
        
        if speaker in speaking_times:
            speaking_times[speaker] += duration
        else:
            speaking_times[speaker] = duration
    
    return speaking_times

def extract_conversation_metrics(segments: List[Dict]) -> Dict:
    """
    Extract conversation metrics from segments
    
    Args:
        segments: List of speaker segments
        
    Returns:
        Dictionary with conversation metrics
    """
    if not segments:
        return {}
    
    # Calculate total duration
    total_duration = max(segment.get('end', 0) for segment in segments)
    
    # Calculate speaking times
    speaking_times = calculate_speaking_time(segments)
    
    # Count segments per speaker
    speaker_counts = {}
    for segment in segments:
        speaker = segment.get('speaker', 'Unknown')
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    
    # Calculate average segment length
    avg_segment_length = sum(
        segment.get('end', 0) - segment.get('start', 0) 
        for segment in segments
    ) / len(segments)
    
    return {
        'total_duration': total_duration,
        'speaking_times': speaking_times,
        'speaker_counts': speaker_counts,
        'avg_segment_length': avg_segment_length,
        'total_segments': len(segments)
    }
