"""
Motion Flow Learner

Extracts "Semantic Motion" from video using dense optical flow on a coarse grid.
Instead of hardcoded parameters (zoom, pan), it learns the actual FLOW FIELD over time.

Output: (T, 2, H_grid, W_grid) tensor representing motion vectors.
Typical Grid Size: 16x9 (captures global motion style without local object noise).
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
import os
import random

@dataclass
class MotionGrid:
    """A sequence of motion flow grids."""
    data: np.ndarray          # (T, 2, H, W) float32 tensor
    source_video: str
    grid_size: Tuple[int, int] # (W, H)

class MotionFlowLearner:
    """
    Extracts coarse optical flow grids to represent global motion style.
    """
    
    def __init__(
        self,
        grid_width: int = 16,
        grid_height: int = 9,
        sample_rate: int = 1  # 1 = every frame
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.sample_rate = sample_rate
        
    def extract_motion(
        self,
        video_path: str,
        max_duration: float = 0, # 0 = full video
        verbose: bool = True
    ) -> Optional[MotionGrid]:
        """
        Extract optical flow grid sequence from video.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # We process on small frames for speed and global motion capture
        # Resize to slightly larger than grid to get good flow, then downsample?
        # Or calculate flow on small image directly.
        # Flow on small image (e.g. 64x36) is robust to noise.
        process_w = 64
        process_h = 36
        
        prev_gray = None
        flows = []
        
        iterator = range(0, total_frames, self.sample_rate)
        if verbose:
            iterator = tqdm(iterator, desc=f"Extracting flow: {os.path.basename(video_path)}")
            
        frame_idx = 0
        for _ in iterator:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize
            small = cv2.resize(frame, (process_w, process_h))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Calculate Dense Optical Flow (Farneback)
                # Parameters optimized for smooth global motion detection
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                # flow is (H, W, 2)
                # Resize to target grid size (16x9)
                # We want average flow in each grid cell
                flow_grid = cv2.resize(flow, (self.grid_width, self.grid_height), interpolation=cv2.INTER_AREA)
                
                # Normalize flow values relative to image size?
                # Flow is in pixels. If we resize image, flow magnitude changes?
                # Farneback flow is in pixels of the input image (64x36).
                # We want "Relative Motion" style.
                # Let's keep it relative to the 64x36 grid for now (or normalize to 0-1 range).
                # Pixels logic is better for reconstruction.
                
                # Store as (2, H, W)
                flow_grid = flow_grid.transpose(2, 0, 1) # (2, H, W)
                flows.append(flow_grid)
            
            prev_gray = gray
            frame_idx += 1
            
            if max_duration > 0 and frame_idx > max_duration * 25:
                break
                
        cap.release()
        
        if not flows:
            return None
            
        # Stack -> (T, 2, H, W)
        data = np.stack(flows).astype(np.float32)
        
        return MotionGrid(
            data=data,
            source_video=video_path,
            grid_size=(self.grid_width, self.grid_height)
        )

    def prepare_training_data(self, grids: List[MotionGrid], window_size: int = 64) -> np.ndarray:
        """
        Slice grids into windows for VAE training.
        Returns: (N, 2, T, H, W)
        """
        windows = []
        stride = window_size // 2
        
        for g in grids:
            data = g.data # (T, 2, H, W)
            T = data.shape[0]
            
            if T < window_size:
                continue
                
            for i in range(0, T - window_size, stride):
                window = data[i:i+window_size] # (Window, 2, H, W)
                # Transpose to (2, Window, H, W) for 3D Conv
                window = window.transpose(1, 0, 2, 3) 
                windows.append(window)
                
        if not windows:
            return np.array([])
            
        return np.stack(windows)
