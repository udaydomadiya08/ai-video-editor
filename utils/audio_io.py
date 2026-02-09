"""
Audio I/O utilities for the AI Video Editing Style Learning System.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import librosa


@dataclass
class AudioMetadata:
    """Metadata for an audio file."""
    path: str
    sample_rate: int
    duration_seconds: float
    channels: int


def get_audio_metadata(audio_path: str) -> AudioMetadata:
    """Extract metadata from an audio file."""
    y, sr = librosa.load(audio_path, sr=None, mono=False, duration=0.1)
    duration = librosa.get_duration(path=audio_path)
    channels = 1 if y.ndim == 1 else y.shape[0]
    
    return AudioMetadata(
        path=audio_path,
        sample_rate=sr,
        duration_seconds=duration,
        channels=channels
    )


def load_audio(
    audio_path: str,
    sample_rate: Optional[int] = None,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (None to keep original)
        mono: Convert to mono
        duration: Load only this many seconds
        offset: Start loading from this time
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    y, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=mono,
        duration=duration,
        offset=offset
    )
    return y, sr


def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 22050
) -> str:
    """
    Extract audio track from a video file.
    
    Args:
        video_path: Path to video file
        output_path: Output audio path (None for temp file)
        sample_rate: Target sample rate
        
    Returns:
        Path to extracted audio file
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
    
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', '1',
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def time_to_samples(time_seconds: float, sample_rate: int) -> int:
    """Convert time in seconds to sample index."""
    return int(time_seconds * sample_rate)


def samples_to_time(samples: int, sample_rate: int) -> float:
    """Convert sample index to time in seconds."""
    return samples / sample_rate


def time_to_frames(time_seconds: float, fps: float) -> int:
    """Convert time in seconds to frame index."""
    return int(time_seconds * fps)


def frames_to_time(frames: int, fps: float) -> float:
    """Convert frame index to time in seconds."""
    return frames / fps


def get_audio_duration(audio_path: str) -> float:
    """Get the duration of an audio file in seconds."""
    return librosa.get_duration(path=audio_path)
