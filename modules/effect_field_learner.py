"""
Effect Field Learner

Extracts "Universal Dynamic Fields" from video using dense optical flow + intensity changes.
This captures:
1. Motion (Flow X, Y) -> Zooms, Pans, Shakes
2. Intensity (Delta Brightness) -> Flashes, Strobes, Fades, Lighting changes

Output: (T, 3, H_grid, W_grid) tensor.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
import os
import random

@dataclass
class EffectGrid:
    """A sequence of effect fields."""
    data: np.ndarray          # (T, 3, H, W) float32 tensor
    source_video: str
    grid_size: Tuple[int, int] # (W, H)

class EffectFieldLearner:
    """
    Extracts coarse flow + intensity grids to represent global visual dynamics.
    """
    
    def __init__(
        self,
        grid_width: int = 48, # M1 Safe "God Mode"
        grid_height: int = 27, # M1 Safe "God Mode"
        sample_rate: int = 1
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.sample_rate = sample_rate
        
    def extract_field(
        self,
        video_path: str,
        max_duration: float = 0, # 0 = full video
        verbose: bool = True
    ) -> Optional[EffectGrid]:
        """
        Extract effect field sequence from video.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        processed_w = 64
        processed_h = 36
        
        prev_gray = None
        fields = []
        
        iterator = range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.sample_rate)
        if verbose:
            iterator = tqdm(iterator, desc=f"Extracting effects: {os.path.basename(video_path)}")
            
        frame_idx = 0
        for _ in iterator:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for processing
            small = cv2.resize(frame, (processed_w, processed_h))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray_float = gray.astype(np.float32) / 255.0
            
            if prev_gray is not None:
                # 1. Optical Flow (Channels 0, 1)
                # Note: Farneback expects uint8 usually, but works on float? 
                # Better pass uint8 for Farneback
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                # 2. Intensity Change (Channel 2)
                # diff = current - prev
                # range roughly [-1, 1]
                prev_gray_float = prev_gray.astype(np.float32) / 255.0
                diff = gray_float - prev_gray_float
                
                # Stack: (H, W, 3) -> FlowX, FlowY, Diff
                # flow is (H, W, 2)
                combined = np.dstack([flow, diff]) # (H, W, 3)
                
                # Resize to target grid size (16x9)
                # Area interpolation averages the values (good for global style)
                grid = cv2.resize(combined, (self.grid_width, self.grid_height), interpolation=cv2.INTER_AREA)
                
                # Store as (3, H, W)
                grid = grid.transpose(2, 0, 1) # (3, H, W)
                fields.append(grid)
            
            prev_gray = gray
            frame_idx += 1
            
            if max_duration > 0 and frame_idx > max_duration * 25:
                break
                
        cap.release()
        
        if not fields:
            return None
            
        # Stack -> (T, 3, H, W)
        data = np.stack(fields).astype(np.float32)
        
        return EffectGrid(
            data=data,
            source_video=video_path,
            grid_size=(self.grid_width, self.grid_height)
        )

    def prepare_training_data(self, grids: List[EffectGrid], window_size: int = 64) -> np.ndarray:
        """
        Slice grids into windows for VAE training.
        Returns: (N, 3, T, H, W)
        """
        windows = []
        stride = window_size // 2
        
        for g in grids:
            data = g.data # (T, 3, H, W)
            T = data.shape[0]
            
            if T < window_size:
                continue
                
            for i in range(0, T - window_size, stride):
                window = data[i:i+window_size] # (Window, 3, H, W)
                # Transpose to (3, Window, H, W) for 3D Conv
                window = window.transpose(1, 0, 2, 3) 
                windows.append(window)
                
        if not windows:
            return np.array([])
            
        return np.stack(windows)
