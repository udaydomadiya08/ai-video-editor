"""
Music Analyzer - Extracts audio features for music-driven editing.

Features extracted:
- Beat detection (onsets, tempo)
- Energy envelope
- Onset strength (attack/energy slope)
- Spectral features (centroid, rolloff)

All features are used for statistical learning, not hardcoded rules.
"""
import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
import pickle

from config import get_config
from utils.audio_io import load_audio, extract_audio_from_video, get_audio_duration


@dataclass
class MusicFeatures:
    """Extracted music features from an audio track."""
    
    # Beat information
    beat_times: np.ndarray = field(default_factory=lambda: np.array([]))  # Seconds
    tempo: float = 120.0
    
    # Time-aligned features (same time axis)
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    energy: np.ndarray = field(default_factory=lambda: np.array([]))
    onset_strength: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_centroid: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_rolloff: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Duration
    duration_seconds: float = 0.0
    sample_rate: int = 22050
    hop_length: int = 512
    
    def get_features_at_time(self, time: float) -> np.ndarray:
        """Get feature vector at a specific time."""
        idx = np.searchsorted(self.times, time)
        idx = min(max(0, idx), len(self.times) - 1)
        
        return np.array([
            self.energy[idx],
            self.onset_strength[idx],
            self.spectral_centroid[idx] / 5000.0,  # Normalize
            self.spectral_rolloff[idx] / 10000.0,  # Normalize
            self._get_beat_strength(time),
            self._get_energy_slope(idx)
        ])
    
    def _get_beat_strength(self, time: float) -> float:
        """Get beat strength at time (proximity to nearest beat)."""
        if len(self.beat_times) == 0:
            return 0.0
        
        distances = np.abs(self.beat_times - time)
        min_dist = np.min(distances)
        
        # Gaussian-like falloff from beat
        sigma = 60 / self.tempo / 4  # Quarter beat window
        return np.exp(-0.5 * (min_dist / sigma) ** 2)
    
    def _get_energy_slope(self, idx: int, window: int = 3) -> float:
        """Get energy slope (attack) at index."""
        start = max(0, idx - window)
        end = min(len(self.energy), idx + 1)
        
        if end - start < 2:
            return 0.0
        
        segment = self.energy[start:end]
        return (segment[-1] - segment[0]) / len(segment)
    
    def save(self, path: str):
        """Save features to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'MusicFeatures':
        """Load features from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class MusicAnalyzer:
    """
    Analyzes audio to extract features for music-driven editing.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    def analyze(
        self,
        audio_path: str,
        verbose: bool = True
    ) -> MusicFeatures:
        """
        Analyze an audio file and extract features.
        
        Args:
            audio_path: Path to audio file
            verbose: Show progress
            
        Returns:
            MusicFeatures object
        """
        if verbose:
            print(f"Analyzing audio: {audio_path}")
        
        # Load audio
        y, sr = load_audio(
            audio_path,
            sample_rate=self.config.audio_sample_rate,
            mono=True
        )
        
        hop_length = self.config.hop_length
        duration = len(y) / sr
        
        features = MusicFeatures(
            duration_seconds=duration,
            sample_rate=sr,
            hop_length=hop_length
        )
        
        # Beat detection
        if verbose:
            print("  Detecting beats...")
        
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length
        )
        features.tempo = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])
        features.beat_times = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=hop_length
        )
        
        if verbose:
            print(f"  Tempo: {features.tempo:.1f} BPM, {len(features.beat_times)} beats")
        
        # Time axis
        n_frames = 1 + len(y) // hop_length
        features.times = librosa.frames_to_time(
            np.arange(n_frames), sr=sr, hop_length=hop_length
        )
        
        # Energy (RMS)
        if verbose:
            print("  Computing energy...")
        
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        features.energy = self._normalize_feature(rms)
        
        # Onset strength
        if verbose:
            print("  Computing onset strength...")
        
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        features.onset_strength = self._normalize_feature(onset)
        
        # Spectral centroid
        if verbose:
            print("  Computing spectral features...")
        
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop_length
        )[0]
        features.spectral_centroid = centroid
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=hop_length
        )[0]
        features.spectral_rolloff = rolloff
        
        # Ensure all arrays have same length
        min_len = min(
            len(features.times),
            len(features.energy),
            len(features.onset_strength),
            len(features.spectral_centroid),
            len(features.spectral_rolloff)
        )
        
        features.times = features.times[:min_len]
        features.energy = features.energy[:min_len]
        features.onset_strength = features.onset_strength[:min_len]
        features.spectral_centroid = features.spectral_centroid[:min_len]
        features.spectral_rolloff = features.spectral_rolloff[:min_len]
        
        if verbose:
            print(f"  Duration: {duration:.2f}s, {min_len} feature frames")
        
        return features
    
    def analyze_video(
        self,
        video_path: str,
        verbose: bool = True
    ) -> MusicFeatures:
        """
        Extract and analyze audio from a video file.
        
        Args:
            video_path: Path to video file
            verbose: Show progress
            
        Returns:
            MusicFeatures object
        """
        if verbose:
            print(f"Extracting audio from: {video_path}")
        
        audio_path = extract_audio_from_video(
            video_path,
            sample_rate=self.config.audio_sample_rate
        )
        
        return self.analyze(audio_path, verbose)
    
    def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:
        """Normalize a feature to [0, 1] range."""
        min_val = np.min(feature)
        max_val = np.max(feature)
        
        if max_val - min_val < 1e-8:
            return np.zeros_like(feature)
        
        return (feature - min_val) / (max_val - min_val)
    
    def get_beat_aligned_times(
        self,
        features: MusicFeatures,
        offset: float = 0.0
    ) -> np.ndarray:
        """
        Get beat times with optional offset.
        
        Args:
            features: MusicFeatures object
            offset: Offset from beat in seconds (can be negative for anticipation)
            
        Returns:
            Array of adjusted beat times
        """
        return features.beat_times + offset
    
    def get_high_energy_times(
        self,
        features: MusicFeatures,
        threshold_percentile: float = 75.0
    ) -> np.ndarray:
        """
        Get times where energy exceeds threshold.
        
        Args:
            features: MusicFeatures object
            threshold_percentile: Percentile threshold for "high" energy
            
        Returns:
            Array of high-energy times
        """
        threshold = np.percentile(features.energy, threshold_percentile)
        high_indices = np.where(features.energy > threshold)[0]
        return features.times[high_indices]
