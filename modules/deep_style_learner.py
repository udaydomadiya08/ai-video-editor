"""
Deep Style Learner

Extracts random 3D spatio-temporal patches from video for unsupervised learning.
These patches capture the full "texture" of the video:
- Color grading
- Film grain / noise patterns
- Motion blur / camera shake patterns
- Compression artifacts or other stylistic elements

Used to train the NeuralStyleVAE.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import os
import random


@dataclass
class StylePatch:
    """A 3D patch of video (Time x Height x Width x Channels)."""
    data: np.ndarray          # Normalized (0-1) float32 tensor
    source_video: str
    start_time: float


class DeepStyleLearner:
    """
    Extracts 3D patches for deep style learning.
    """
    
    def __init__(
        self,
        patch_size: int = 64,      # Spatial size (HxW)
        frames_per_patch: int = 16, # Temporal size (T)
        patches_per_video: int = 100
    ):
        self.patch_size = patch_size
        self.frames_per_patch = frames_per_patch
        self.patches_per_video = patches_per_video
    
    def extract_patches(
        self,
        video_path: str,
        verbose: bool = True
    ) -> List[StylePatch]:
        """
        Extract random 3D patches from video.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_count < self.frames_per_patch:
            # Handle short video: Loop it or skip
            # For now return empty or try to get what we can
            # Let's try to extract at least one sequence if possible
            if frame_count < 4: 
                return []
            # We can't easily extract full sequence, so skip
            return []
        
        patches = []
        
        # Determine random start points
        # Keep some distance from edges
        max_start_frame = frame_count - self.frames_per_patch
        if max_start_frame <= 0:
            max_start_frame = 1 # Take from 0
            
        # Ensure we don't sample more than possible
        num_patches = min(self.patches_per_video, max_start_frame + 1)
        
        if num_patches <= 0:
             return []

        start_frames = sorted(random.sample(
            range(max_start_frame + 1), 
            num_patches
        ))
        
        if verbose:
            print(f"  Extracting {len(start_frames)} style patches from {os.path.basename(video_path)}...")
            iter_frames = tqdm(start_frames, leave=False)
        else:
            iter_frames = start_frames
            
        for start_frame in iter_frames:
            # Extract sequence
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            valid_sequence = True
            for _ in range(self.frames_per_patch):
                ret, frame = cap.read()
                if not ret:
                    valid_sequence = False
                    break
                frames.append(frame)
            
            if not valid_sequence or len(frames) != self.frames_per_patch:
                continue
                
            # Random spatial crop
            # We crop the SAME spatial region for all frames in the sequence
            # (to capture temporal evolution of that region)
            h, w = frames[0].shape[:2]
            
            if h < self.patch_size or w < self.patch_size:
                # Resize if too small
                scale = max(self.patch_size / h, self.patch_size / w)
                frames = [cv2.resize(f, None, fx=scale, fy=scale) for f in frames]
                h, w = frames[0].shape[:2]
            
            y = random.randint(0, h - self.patch_size)
            x = random.randint(0, w - self.patch_size)
            
            crop_sequence = []
            for frame in frames:
                crop = frame[y:y+self.patch_size, x:x+self.patch_size]
                crop_sequence.append(crop)
            
            # Convert to numpy array (T, H, W, C)
            patch_data = np.array(crop_sequence)
            
            # Normalize to 0-1 float32
            patch_data = patch_data.astype(np.float32) / 255.0
            
            patches.append(StylePatch(
                data=patch_data,
                source_video=video_path,
                start_time=start_frame / fps if fps > 0 else 0
            ))
            
        cap.release()
        return patches

    def prepare_training_data(self, patches: List[StylePatch]) -> np.ndarray:
        """
        Convert list of patches to training tensor.
        Returns: (N, C, T, H, W) for PyTorch 3D Conv
        """
        if not patches:
            return np.array([])
            
        # Stack patches -> (N, T, H, W, C)
        # Note: patches[0].data is (T, H, W, C)
        data = np.stack([p.data for p in patches])
        
        # Transpose to (N, C, T, H, W)
        data = np.transpose(data, (0, 4, 1, 2, 3))
        
        return data
