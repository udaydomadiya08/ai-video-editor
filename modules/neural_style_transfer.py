"""
Neural Style Transfer (Fast Video Style)

Applies learned global video styles to new clips.

Features:
- Deep VAE Style Transfer (Texture, Grain, Color)

Optimized for speed to work on video frames in real-time.
"""

import cv2
import numpy as np
import torch
import os
from typing import Optional, Tuple, List
from modules.neural_style_vae import VideoStyleModel

class StyleTransfer:
    """
    Applies learned visual style to video frames.
    Supports deep neural transfer ONLY.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.deep_style_model: Optional[VideoStyleModel] = None
        self.style_patches: Optional[np.ndarray] = None
        self.device = config.device if config else 'cpu'
    
    def load_deep_style(self, model_path: str, reference_patches: Optional[np.ndarray] = None):
        """Load the Deep Style VAE model."""
        if os.path.exists(model_path):
            try:
                self.deep_style_model = VideoStyleModel(self.config)
                self.deep_style_model.load(model_path)
                print(f"  ✓ Loaded Deep Style VAE from {os.path.basename(model_path)}")
                
                # If no reference patches provided, we need them to define the style
                # In a real app, we'd save representative patches with the model
                # For now, we might rely on the user to provide them or use the model's internal state if we added that
                self.style_patches = reference_patches
            except Exception as e:
                print(f"  ⚠️ Could not load Deep Style VAE: {e}")
    
    def apply_deep_style_to_clip(self, clip_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply deep style transfer to a sequence of frames.
        Properly handles temporal dimension.
        """
        if self.deep_style_model is None or self.style_patches is None:
            # Fallback to original if no model
            return clip_frames
            
        # Process in chunks of 16 frames (or whatever model trained on)
        chunk_size = 16
        stylized_frames = []
        
        # Convert entire clip to numpy
        clip_np = np.array(clip_frames) # (T, H, W, C)
        
        for i in range(0, len(clip_np), chunk_size):
            chunk = clip_np[i:i+chunk_size]
            if len(chunk) < 4: # Too small for 3D conv usually
                # Just append as is
                stylized_frames.extend([f for f in chunk])
                continue
                
            # Pad if needed to reach chunk_size? Or just process variable size if model allows
            # Our VAE handles variable time dim but downsamples T by 4 (2x2 pooling)
            # So T must be divisible by 4.
            
            # Pad to multiple of 4
            T = len(chunk)
            pad_t = (4 - T % 4) % 4
            if pad_t > 0:
                # Repeat last frame
                padding = np.repeat(chunk[-1:], pad_t, axis=0)
                chunk_padded = np.concatenate([chunk, padding], axis=0)
            else:
                chunk_padded = chunk
                
            # Apply transfer
            try:
                out_chunk = self.deep_style_model.transfer_style(
                    chunk_padded, # (T, H, W, C)
                    self.style_patches,
                    alpha=0.8 # heavy style
                )
                
                # Remove padding
                if pad_t > 0:
                    out_chunk = out_chunk[:-pad_t]
                    
                stylized_frames.extend([f for f in out_chunk])
            except Exception as e:
                print(f"Style transfer failed: {e}")
                stylized_frames.extend([f for f in chunk])
            
        return stylized_frames

    def apply_frame_style(self, frame: np.ndarray) -> np.ndarray:
        """Single frame wrapper for deep style."""
        # Fake a clip of 4 frames (minimum for pooling)
        # Note: Deep model expects 3D chunks (T frames). 
        clip = [frame] * 4
        out_clip = self.apply_deep_style_to_clip(clip)
        return out_clip[0]
