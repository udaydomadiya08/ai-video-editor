"""
Transition Frame Extractor

Extracts frame windows around cuts in reference videos for neural transition learning.
This module captures the actual pixel transformations that happen during transitions,
enabling the VAE to learn ANY visual effect - not just predefined categories.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from tqdm import tqdm
import os


@dataclass
class TransitionSequence:
    """
    A sequence of frames around a cut point.
    
    Contains:
    - frames_before: N frames leading up to the cut
    - frames_after: N frames after the cut
    - difference_frames: Frame-to-frame differences (motion representation)
    - flow_vectors: Optical flow between consecutive frames
    - metadata: Source video, timing, etc.
    """
    video_path: str
    cut_time: float              # Time of cut in seconds
    window_frames: int           # N frames before/after
    fps: float
    
    # Frame data (normalized to fixed size)
    frames_before: np.ndarray    # (N, H, W, C) 
    frames_after: np.ndarray     # (N, H, W, C)
    
    # Derived representations
    difference_frames: np.ndarray = None  # (2N-1, H, W, C) frame differences
    flow_vectors: np.ndarray = None       # (2N-1, H, W, 2) optical flow
    
    # Combined sequence for training
    full_sequence: np.ndarray = None      # (2N, H, W, C) all frames
    
    def to_training_tensor(self) -> np.ndarray:
        """
        Convert to normalized tensor for VAE training.
        Returns: (T, H, W, 3)
        Channels 0,1: Optical Flow (dx, dy) normalized
        Channel 2: Intensity Difference normalized
        """
        if self.flow_vectors is not None and self.difference_frames is not None:
            # 1. Normalize Flow (-20..20 -> -1..1 -> 0..1 for VAE stability)
            # We assume max flow of 20px for normalization
            flow_norm = np.clip(self.flow_vectors / 20.0, -1.0, 1.0)
            
            # 2. Intensity (Grayscale difference)
            # diff frames are uint8 0..255 (abs diff)
            # We want signed difference? No, extracting flow captures movement.
            # Difference captures lighting changes (flash/fade).
            # Let's use the luminance of the difference frame
            if self.difference_frames.shape[-1] == 3:
                intensity = cv2.cvtColor(self.difference_frames, cv2.COLOR_BGR2GRAY)
            else:
                intensity = self.difference_frames
                
            intensity_norm = intensity.astype(np.float32) / 255.0 # 0..1
            
            # Expand diff to (T, H, W, 1)
            if intensity_norm.ndim == 3:
                intensity_norm = intensity_norm[..., np.newaxis]
                
            # Combine: Flow(2) + Intensity(1) = 3 channels
            combined = np.concatenate([flow_norm, intensity_norm], axis=-1)
            # Result: (T, H, W, 3)
            return combined
            
        elif self.difference_frames is not None:
             return self.difference_frames.astype(np.float32) / 255.0
        else:
             return self.full_sequence.astype(np.float32) / 255.0


class TransitionFrameExtractor:
    """
    Extracts frame sequences around cuts for neural transition learning.
    
    For each cut in a reference video:
    1. Extract N frames before the cut
    2. Extract N frames after the cut  
    3. Compute frame differences (captures motion/effects)
    4. Compute optical flow (captures motion direction)
    5. Normalize to fixed resolution for training
    """
    
    def __init__(
        self,
        window_frames: int = 5,       # Frames before/after cut
        output_size: Tuple[int, int] = (128, 128),  # Normalized frame size
        compute_flow: bool = True,    # Whether to compute optical flow
        compute_diff: bool = True     # Whether to compute frame differences
    ):
        self.window_frames = window_frames
        self.output_size = output_size
        self.compute_flow = compute_flow
        self.compute_diff = compute_diff
    
    def extract_random_sequences(
        self,
        video_path: str,
        num_sequences: int = 50,
        verbose: bool = True
    ) -> List[TransitionSequence]:
        """
        Extract random sequences from the ENTIRE video to learn general effects.
        "Learn everything from videos"
        """
        sequences = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count < self.window_frames * 2:
            cap.release()
            return []
            
        # Pick random start frames
        # Ensure we have enough room for 2*window_frames
        max_start = frame_count - (self.window_frames * 2) - 1
        if max_start <= 0:
            start_frames = [0]
        else:
            start_frames = np.random.randint(0, max_start, num_sequences)
            
        start_frames.sort() # Sequential access is faster
        
        current_frame_idx = 0
        
        iterator = start_frames
        if verbose:
            iterator = tqdm(start_frames, desc=f"Scanning entire video {Path(video_path).name}")
            
        for start_frame in iterator:
            # Seek if needed
            if current_frame_idx > start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current_frame_idx = start_frame
            
            while current_frame_idx < start_frame:
                ret = cap.grab()
                if not ret: break
                current_frame_idx += 1
                
            # Read sequence
            window_frames = []
            needed = self.window_frames * 2
            
            for _ in range(needed):
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.resize(frame, self.output_size)
                window_frames.append(frame)
                current_frame_idx += 1
                
            if len(window_frames) == needed:
                frames_arr = np.array(window_frames)
                frames_before = frames_arr[:self.window_frames]
                frames_after = frames_arr[self.window_frames:]
                
                seq = TransitionSequence(
                    video_path=video_path,
                    cut_time=start_frame / fps if fps > 0 else 0,
                    window_frames=self.window_frames,
                    fps=fps,
                    frames_before=frames_before,
                    frames_after=frames_after,
                    full_sequence=frames_arr
                )
                
                # Compute derived
                if self.compute_diff:
                    seq.difference_frames = self._compute_differences(frames_arr)
                if self.compute_flow:
                    seq.flow_vectors = self._compute_flow(frames_arr)
                    
                sequences.append(seq)
                
        cap.release()
        return sequences

    def extract_transitions(
        self,
        video_path: str,
        cut_times: List[float],
        verbose: bool = True
    ) -> List[TransitionSequence]:
        """
        Extract transition sequences using robust sequential reading.
        Avoids cv2.set() frame seeking which is unreliable on H.264.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Sort cuts to enable sequential processing
        # Ensure cut_times are integers (frame indices)
        cut_frames = sorted([int(c) for c in cut_times])
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        sequences = []
        current_frame_idx = 0
        
        # Process cuts sequentially
        iterator = cut_frames
        if verbose:
            iterator = tqdm(cut_frames, desc=f"Extracting transitions from {Path(video_path).name}")
        
        for cut_frame in iterator:
            # Define window
            start_frame = cut_frame - self.window_frames
            end_frame = cut_frame + self.window_frames
            
            if start_frame < 0 or end_frame >= frame_count:
                continue
            
            # Skip to start_frame
            # If we are ahead of start_frame (due to overlap or disorder), we must re-seek
            if current_frame_idx > start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current_frame_idx = start_frame
            
            # Skip frames until start_frame
            while current_frame_idx < start_frame:
                ret = cap.grab() # Faster than read()
                if not ret:
                    break
                current_frame_idx += 1
            
            if current_frame_idx != start_frame:
                continue # Failed to reach start
            
            # Read window frames
            window_frames = []
            needed = self.window_frames * 2
            
            for _ in range(needed):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize immediately to save memory
                frame = cv2.resize(frame, self.output_size)
                window_frames.append(frame)
                current_frame_idx += 1
            
            if len(window_frames) == needed:
                # Create sequence
                frames_arr = np.array(window_frames)
                frames_before = frames_arr[:self.window_frames]
                frames_after = frames_arr[self.window_frames:]
                
                seq = TransitionSequence(
                    video_path=video_path,
                    cut_time=cut_frame / fps if fps > 0 else 0,
                    window_frames=self.window_frames,
                    fps=fps,
                    frames_before=frames_before,
                    frames_after=frames_after,
                    full_sequence=frames_arr
                )
                
                # Compute derived
                if self.compute_diff:
                    seq.difference_frames = self._compute_differences(frames_arr)
                if self.compute_flow:
                    seq.flow_vectors = self._compute_flow(frames_arr)
                
                sequences.append(seq)
        
        cap.release()
        
        if verbose:
            print(f"  Extracted {len(sequences)} valid transitions")
        
        return sequences
    
    def _compute_differences(self, frames: np.ndarray) -> np.ndarray:
        """
        Compute frame-to-frame differences.
        
        This captures the motion/changes between frames,
        which is what we want the VAE to learn.
        """
        n_frames = len(frames)
        diffs = []
        
        for i in range(n_frames - 1):
            # Signed difference to capture direction of change
            diff = frames[i + 1].astype(np.float32) - frames[i].astype(np.float32)
            # Normalize to 0-255 range (128 = no change)
            diff = np.clip(diff + 128, 0, 255).astype(np.uint8)
            diffs.append(diff)
        
        return np.array(diffs)
    
    def _compute_flow(self, frames: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between consecutive frames.
        
        Flow vectors capture the motion direction and magnitude.
        """
        n_frames = len(frames)
        flows = []
        
        for i in range(n_frames - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            flows.append(flow)
        
        return np.array(flows)
    
    def sequences_to_training_data(
        self,
        sequences: List[TransitionSequence],
        use_differences: bool = True
    ) -> np.ndarray:
        """
        Convert list of sequences to training data array.
        
        Args:
            sequences: List of TransitionSequence objects
            use_differences: Use frame differences (True) or raw frames (False)
            
        Returns:
            Array of shape (N, seq_len, H, W, C) normalized to 0-1
        """
        data = []
        
        for seq in sequences:
            if use_differences and seq.difference_frames is not None:
                tensor = seq.difference_frames.astype(np.float32) / 255.0
            else:
                tensor = seq.full_sequence.astype(np.float32) / 255.0
            
            data.append(tensor)
        
        return np.array(data)


def extract_all_transitions(
    video_paths: List[str],
    cut_times_per_video: Dict[str, List[float]],
    window_frames: int = 5,
    output_size: Tuple[int, int] = (128, 128),
    verbose: bool = True
) -> List[TransitionSequence]:
    """
    Convenience function to extract transitions from multiple videos.
    
    Args:
        video_paths: List of video file paths
        cut_times_per_video: Dict mapping video path to list of cut times
        window_frames: Frames before/after each cut
        output_size: Normalized frame size
        verbose: Show progress
        
    Returns:
        Combined list of all TransitionSequence objects
    """
    extractor = TransitionFrameExtractor(
        window_frames=window_frames,
        output_size=output_size
    )
    
    all_sequences = []
    
    for video_path in video_paths:
        if video_path not in cut_times_per_video:
            continue
        
        cut_times = cut_times_per_video[video_path]
        
        if not cut_times:
            continue
        
        try:
            sequences = extractor.extract_transitions(
                video_path, cut_times, verbose=verbose
            )
            all_sequences.extend(sequences)
        except Exception as e:
            if verbose:
                print(f"  Error processing {video_path}: {e}")
    
    if verbose:
        print(f"\nTotal transitions extracted: {len(all_sequences)}")
    
    return all_sequences
