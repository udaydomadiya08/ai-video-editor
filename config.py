"""
AI Video Editing Style Learning System - Configuration
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class Config:
    """Central configuration for the video editing style learning system."""
    
    # === Input/Output Paths ===
    reference_videos_dir: str = "./reference_videos"
    raw_clips_dir: str = "./raw_clips"
    music_path: str = "./music.mp3"
    output_path: str = "./output.mp4"
    models_dir: str = "./models"
    
    # === Video Output Settings ===
    output_width: int = 1080
    output_height: int = 1920  # 9:16 vertical
    output_fps: int = 30
    output_codec: str = "libx264"
    output_audio_codec: str = "aac"
    
    # === Style Learning Parameters ===
    # Cut detection
    histogram_diff_threshold: float = 0.5  # Learned, not fixed
    optical_flow_threshold: float = 30.0   # Learned, not fixed
    min_shot_length_frames: int = 6        # ~0.2s at 30fps
    
    # Motion analysis
    optical_flow_sample_rate: int = 2      # Analyze every N frames
    motion_window_size: int = 5            # Frames for smoothing
    
    # === Transition Learning Parameters ===
    transition_window_frames: int = 15     # ±15 frames around cuts
    transition_features: int = 7           # flow_x, flow_y, scale, tx, ty, blur, brightness
    
    # Autoencoder
    autoencoder_latent_dim: int = 12
    autoencoder_hidden_dim: int = 64
    autoencoder_epochs: int = 500  # Increased for better learning
    autoencoder_batch_size: int = 32
    autoencoder_lr: float = 0.001
    
    # Clustering
    max_transition_clusters: int = 8       # Maximum distinct transition types
    min_transition_clusters: int = 2       # Minimum clusters
    
    # === Music Analysis Parameters ===
    audio_sample_rate: int = 22050
    hop_length: int = 512                  # ~23ms at 22050 Hz
    beat_lookahead_ms: int = 50            # Anticipation window
    energy_window_ms: int = 100            # Energy smoothing window
    
    # === Generation Parameters ===
    clip_motion_score_weight: float = 0.7
    clip_diversity_weight: float = 0.3
    transition_sample_temperature: float = 1.0  # Higher = more variety
    
    # === Processing Settings ===
    verbose: bool = True
    device: str = "cpu"  # 'cuda' if available
    random_seed: int = 42
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.models_dir, exist_ok=True)
    
    @property
    def aspect_ratio(self) -> float:
        return self.output_width / self.output_height
    
    @property
    def transition_window_total(self) -> int:
        """Total frames in transition window (before + after cut)."""
        return self.transition_window_frames * 2 + 1
    
    @property
    def transition_vector_size(self) -> int:
        """Size of flattened transition feature vector."""
        return self.transition_window_total * self.transition_features


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(**kwargs) -> Config:
    """Set global config with custom values."""
    global _config
    _config = Config(**kwargs)
    return _config
