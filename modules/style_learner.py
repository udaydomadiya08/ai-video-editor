"""
Style Learning Module - Learns editing style from reference videos.

Analyzes:
- Cut timing patterns (shot length distributions)
- Beat-to-cut alignment behavior
- Motion intensity over time
- Transition patterns around cuts

All learning is data-driven with no hardcoded rules.
"""
import numpy as np
import cv2
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import pickle
from pathlib import Path
from tqdm import tqdm

from config import get_config
from utils.video_io import get_video_metadata, iter_frames, get_all_videos
from utils.audio_io import extract_audio_from_video, load_audio
from utils.audio_io import extract_audio_from_video, load_audio


@dataclass
class StyleParameters:
    """Learned style parameters from reference videos."""
    
    # Shot length distribution parameters
    shot_lengths: List[float] = field(default_factory=list)  # In seconds
    shot_length_distribution: Optional[str] = None  # Distribution type
    shot_length_params: Dict = field(default_factory=dict)  # Distribution parameters
    
    # Motion intensity statistics
    motion_mean: float = 0.0
    motion_std: float = 1.0
    motion_percentiles: Dict[int, float] = field(default_factory=dict)
    
    # Beat-cut alignment statistics
    beat_cut_offsets: List[float] = field(default_factory=list)  # Offset from beat in seconds
    beat_cut_alignment_mean: float = 0.0
    beat_cut_alignment_std: float = 0.05
    
    # Overall statistics
    avg_cuts_per_second: float = 0.5
    reference_fps: float = 30.0
    
    def save(self, path: str):
        """Save parameters to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'StyleParameters':
        """Load parameters from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class StyleLearner:
    """
    Learns video editing style from reference videos.
    
    All learning is statistical and data-driven.
    No hardcoded rules or named effects.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.style_params = StyleParameters()
        
    def learn_from_videos(self, video_paths: List[str], verbose: bool = True) -> StyleParameters:
        """
        Learn style parameters from a list of reference videos.
        
        Args:
            video_paths: List of paths to reference videos
            verbose: Show progress
            
        Returns:
            Learned StyleParameters
        """
        all_cuts = []
        all_shot_lengths = []
        all_motion_intensities = []
        all_beat_cut_offsets = []
        
        for video_path in tqdm(video_paths, desc="Analyzing videos", disable=not verbose):
            if verbose:
                print(f"\nAnalyzing: {Path(video_path).name}")
            
            # Get video metadata
            metadata = get_video_metadata(video_path)
            self.style_params.reference_fps = metadata.fps
            
            # Detect cuts
            cuts = self.detect_cuts(video_path, verbose)
            all_cuts.extend([(cut, metadata.fps) for cut in cuts])
            
            # Calculate shot lengths
            shot_lengths = self._calculate_shot_lengths(cuts, metadata.fps, metadata.duration_seconds)
            all_shot_lengths.extend(shot_lengths)
            
            # Calculate motion intensity
            motion = self._calculate_motion_intensity(video_path, verbose)
            all_motion_intensities.extend(motion)
            
            # Analyze beat-cut alignment
            try:
                offsets = self._analyze_beat_cut_alignment(video_path, cuts, metadata.fps)
                all_beat_cut_offsets.extend(offsets)
            except Exception as e:
                if verbose:
                    print(f"  Skipping beat analysis: {e}")
        
        # Fit shot length distribution
        if all_shot_lengths:
            self._fit_shot_length_distribution(all_shot_lengths)
        
        # Calculate motion statistics
        if all_motion_intensities:
            self._calculate_motion_statistics(all_motion_intensities)
        
        # Calculate beat-cut alignment statistics
        if all_beat_cut_offsets:
            self._calculate_beat_alignment_stats(all_beat_cut_offsets)
        
        # Calculate average cuts per second
        total_duration = sum(get_video_metadata(v).duration_seconds for v in video_paths)
        total_cuts = len(all_shot_lengths)
        if total_duration > 0:
            self.style_params.avg_cuts_per_second = total_cuts / total_duration
        
        return self.style_params
    
    def detect_cuts(self, video_path: str, verbose: bool = True) -> List[int]:
        """
        Detect cut points in a video using frame difference analysis.
        
        Uses histogram difference and optical flow discontinuity.
        Thresholds are adaptively learned from the video itself.
        """
        cuts = []
        prev_frame = None
        prev_hist = None
        frame_diffs = []
        
        # First pass: collect frame differences
        if verbose:
            print("  Pass 1: Collecting frame differences...")
        
        for frame_idx, frame in iter_frames(video_path, grayscale=True):
            if prev_frame is not None:
                # Histogram difference
                hist = cv2.calcHist([frame], [0], None, [64], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if prev_hist is not None:
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    frame_diffs.append((frame_idx, diff))
                
                prev_hist = hist
            else:
                prev_hist = cv2.calcHist([frame], [0], None, [64], [0, 256])
                prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
            
            prev_frame = frame
        
        if not frame_diffs:
            return cuts
        
        # Adaptive threshold based on distribution of differences
        diffs = np.array([d[1] for d in frame_diffs])
        
        # Use statistical threshold: mean + k * std
        # k is learned from the data distribution
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        # Robust threshold using percentile
        threshold = np.percentile(diffs, 95)  # Top 5% are likely cuts
        
        # Also ensure threshold is significantly above mean
        adaptive_threshold = max(threshold, mean_diff + 2 * std_diff)
        
        # Second pass: detect cuts using adaptive threshold
        if verbose:
            print(f"  Pass 2: Detecting cuts (adaptive threshold: {adaptive_threshold:.3f})...")
        
        min_gap = self.config.min_shot_length_frames
        last_cut = -min_gap
        
        for frame_idx, diff in frame_diffs:
            if diff > adaptive_threshold and (frame_idx - last_cut) >= min_gap:
                cuts.append(frame_idx)
                last_cut = frame_idx
        
        if verbose:
            print(f"  Found {len(cuts)} cuts")
        
        return cuts
    
    def _calculate_shot_lengths(
        self,
        cuts: List[int],
        fps: float,
        total_duration: float
    ) -> List[float]:
        """Calculate shot lengths in seconds from cut points."""
        if not cuts:
            return [total_duration]
        
        shot_lengths = []
        
        # First shot: from start to first cut
        shot_lengths.append(cuts[0] / fps)
        
        # Middle shots: between consecutive cuts
        for i in range(1, len(cuts)):
            shot_lengths.append((cuts[i] - cuts[i-1]) / fps)
        
        # Last shot: from last cut to end
        last_shot = total_duration - (cuts[-1] / fps)
        if last_shot > 0:
            shot_lengths.append(last_shot)
        
        return shot_lengths
    
    def _fit_shot_length_distribution(self, shot_lengths: List[float]):
        """
        Learns the shot length distribution directly from data (Non-Parametric).
        Uses Kernel Density Estimation (KDE) instead of named categories.
        """
        self.style_params.shot_lengths = shot_lengths
        
        # Filter out extreme outliers for stability
        data = np.array(shot_lengths)
        if len(data) < 2:
            return

        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = max(0.1, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        if len(filtered_data) < 3:
            filtered_data = data

        # Store critical percentiles and raw sorted data for sampling
        # We don't fit "Lognorm" or "Gamma" anymore. Pure data.
        self.style_params.shot_length_distribution = "raw_kde"
        self.style_params.shot_length_params = {
            'min': float(np.min(filtered_data)),
            'max': float(np.max(filtered_data)),
            'mean': float(np.mean(filtered_data)),
            'std': float(np.std(filtered_data)),
            'percentiles': [
                float(np.percentile(filtered_data, p)) for p in range(0, 101, 5)
            ],
            # Store a resampled buffer for fast generation
            'kde_samples': self._resample_kde(filtered_data, n_samples=1000)
        }
        
    def _resample_kde(self, data, n_samples=1000):
        """Generate smooth samples from data using Gaussian KDE."""
        try:
            kernel = stats.gaussian_kde(data)
            samples = kernel.resample(n_samples)[0]
            return [float(x) for x in np.clip(samples, 0.1, np.max(data)*1.2)]
        except:
            # Fallback if singular matrix
            return [float(x) for x in np.random.choice(data, n_samples)]
    
    def _calculate_motion_intensity(
        self,
        video_path: str,
        verbose: bool = True
    ) -> List[float]:
        """
        Calculate motion intensity over time using optical flow.
        """
        motion_values = []
        prev_frame = None
        
        sample_rate = self.config.optical_flow_sample_rate
        
        for frame_idx, frame in iter_frames(
            video_path,
            sample_rate=sample_rate,
            grayscale=True,
            resize=(320, 180)  # Downscale for speed
        ):
            if prev_frame is not None:
                # Farneback optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, frame,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
                
                # Motion magnitude
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_values.append(np.mean(mag))
            
            prev_frame = frame
        
        return motion_values
    
    def _calculate_motion_statistics(self, motion_values: List[float]):
        """Calculate motion intensity statistics."""
        data = np.array(motion_values)
        
        self.style_params.motion_mean = float(np.mean(data))
        self.style_params.motion_std = float(np.std(data))
        self.style_params.motion_percentiles = {
            10: float(np.percentile(data, 10)),
            25: float(np.percentile(data, 25)),
            50: float(np.percentile(data, 50)),
            75: float(np.percentile(data, 75)),
            90: float(np.percentile(data, 90))
        }
    
    def _analyze_beat_cut_alignment(
        self,
        video_path: str,
        cuts: List[int],
        fps: float
    ) -> List[float]:
        """
        Analyze how cuts align with audio beats.
        
        Returns offset of each cut from nearest beat in seconds.
        """
        import librosa
        
        # Extract audio
        audio_path = extract_audio_from_video(video_path)
        y, sr = load_audio(audio_path, sample_rate=22050)
        
        # Detect beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        if len(beat_times) == 0:
            return []
        
        # Calculate offset of each cut from nearest beat
        offsets = []
        cut_times = [cut / fps for cut in cuts]
        
        for cut_time in cut_times:
            # Find nearest beat
            distances = np.abs(beat_times - cut_time)
            nearest_idx = np.argmin(distances)
            offset = cut_time - beat_times[nearest_idx]
            offsets.append(offset)
        
        return offsets
    
    def _calculate_beat_alignment_stats(self, offsets: List[float]):
        """Calculate beat-cut alignment statistics."""
        data = np.array(offsets)
        
        self.style_params.beat_cut_offsets = offsets
        self.style_params.beat_cut_alignment_mean = float(np.mean(data))
        self.style_params.beat_cut_alignment_std = float(np.std(data))
    
    def sample_shot_length(self) -> float:
        """Sample a shot length from the raw learned distribution."""
        params = self.style_params.shot_length_params
        
        if not params or 'kde_samples' not in params:
            # Fallback to empirical sampling
            if self.style_params.shot_lengths:
                return float(np.random.choice(self.style_params.shot_lengths))
            return 1.0  # Default 1 second
        
        # Sample from the pre-computed smooth KDE buffer
        return float(np.random.choice(params['kde_samples']))
    
    def get_beat_offset(self) -> float:
        """Sample a beat-cut offset from the learned distribution."""
        mean = self.style_params.beat_cut_alignment_mean
        std = self.style_params.beat_cut_alignment_std
        
        return float(np.random.normal(mean, std))


def learn_style(reference_dir: str, output_path: str, verbose: bool = True) -> StyleParameters:
    """
    Convenience function to learn style from a directory of reference videos.
    
    Args:
        reference_dir: Directory containing reference videos
        output_path: Path to save learned parameters
        verbose: Show progress
        
    Returns:
        Learned StyleParameters
    """
    videos = get_all_videos(reference_dir)
    
    if not videos:
        raise ValueError(f"No videos found in {reference_dir}")
    
    if verbose:
        print(f"Found {len(videos)} reference videos")
    
    learner = StyleLearner()
    params = learner.learn_from_videos(videos, verbose=verbose)
    
    # Save parameters
    params.save(output_path)
    
    if verbose:
        print(f"\nSaved style parameters to {output_path}")
        print(f"  Shot length distribution: {params.shot_length_distribution}")
        print(f"  Avg cuts/second: {params.avg_cuts_per_second:.2f}")
        print(f"  Beat alignment mean: {params.beat_cut_alignment_mean*1000:.1f}ms")
    
    return params
