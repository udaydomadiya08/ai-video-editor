"""
Neural Effect Generator

Applies learned transitions at generation time by:
1. Decoding latent embeddings to transition templates
2. Applying the learned motion/effects to actual source/target frames
3. Generating smooth intermediate frames between clips

This module bridges the learned VAE latent space with actual video rendering.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from config import get_config


@dataclass
class TransitionConfig:
    """Configuration for a transition."""
    n_frames: int = 5          # Number of transition frames
    blend_mode: str = 'lerp'   # 'lerp', 'neural', 'flow'
    intensity: float = 1.0     # Effect intensity (0-1)


class NeuralEffectGenerator:
    """
    Generates transitions using learned neural embeddings.
    
    Takes learned transition embeddings from the VAE and applies
    them to actual source/target frames during video generation.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.neural_vae = None
        self.output_size = (128, 128)  # Size used during training
    
    def set_vae(self, neural_vae):
        """Set the trained Neural Transition VAE."""
        self.neural_vae = neural_vae
        if neural_vae is not None:
            self.output_size = neural_vae.output_size
    
    def generate_transition_frames(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        embedding: np.ndarray,
        n_frames: int = 5,
        intensity: float = 1.0
    ) -> List[np.ndarray]:
        """
        Generate transition frames between two clips using learned embedding.
        
        Args:
            frame_before: Last frame of previous clip (H, W, C)
            frame_after: First frame of next clip (H, W, C)
            embedding: Latent embedding from VAE (D,)
            n_frames: Number of transition frames to generate
            intensity: Effect intensity (0=linear blend, 1=full neural effect)
            
        Returns:
            List of n_frames intermediate frames
        """
        if self.neural_vae is None:
            # Fallback to simple linear interpolation
            return self._linear_blend(frame_before, frame_after, n_frames)
        
        # Get the target size (original frame size)
        target_h, target_w = frame_before.shape[:2]
        
        # Decode embedding to get learned motion template
        motion_template = self._decode_embedding(embedding)
        
        # Apply learned motion to actual frames
        transition_frames = self._apply_motion_template(
            frame_before, frame_after,
            motion_template, n_frames, intensity
        )
        
        # Resize back to original size
        transition_frames = [
            cv2.resize(f, (target_w, target_h)) for f in transition_frames
        ]
        
        return transition_frames
    
    def _decode_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Decode embedding to motion template.
        
        Returns: Array (frames, H, W, C) representing the learned motion pattern
        """
        if self.neural_vae is None:
            return None
        
        # Decode to get sequence of frame differences
        decoded = self.neural_vae.decode(embedding.reshape(1, -1))
        
        # decoded shape: (1, frames, H, W, C)
        motion_template = decoded[0]  # (frames, H, W, C)
        
        return motion_template
    
    def _apply_motion_template(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        motion_template: np.ndarray,
        n_frames: int,
        intensity: float
    ) -> List[np.ndarray]:
        """
        Apply learned motion template to source/target frames.
        
        The motion template encodes frame-to-frame differences learned
        from reference videos. We apply these differences to blend
        the actual frames.
        """
        transition_frames = []
        
        # Resize source frames to match template size
        template_h, template_w = motion_template.shape[1:3]
        frame_before_small = cv2.resize(frame_before, (template_w, template_h))
        frame_after_small = cv2.resize(frame_after, (template_w, template_h))
        
        # Normalize frames
        frame_before_norm = frame_before_small.astype(np.float32) / 255.0
        frame_after_norm = frame_after_small.astype(np.float32) / 255.0
        
        # Get template frames (normalized)
        n_template_frames = len(motion_template)
        
        for i in range(n_frames):
            # Blend factor (0 = before, 1 = after)
            t = (i + 1) / (n_frames + 1)
            
            # Base: linear interpolation
            base_frame = (1 - t) * frame_before_norm + t * frame_after_norm
            
            # Get corresponding template frame
            template_idx = int(t * (n_template_frames - 1))
            template_idx = min(template_idx, n_template_frames - 1)
            template_frame = motion_template[template_idx]  # Already in 0-1 range
            
            # Motion from template: the template encodes differences
            # We use the template to modulate the blend
            if intensity > 0:
                # The template represents motion/effects
                # Center the template around 0.5 (was stored with 128 offset in training)
                motion_mod = (template_frame - 0.5) * 2  # Range -1 to 1
                
                # Apply motion modulation to base frame
                modulated = base_frame + motion_mod * intensity * 0.2
                modulated = np.clip(modulated, 0, 1)
                
                # Blend with linear for stability
                result = (1 - intensity * 0.5) * base_frame + intensity * 0.5 * modulated
            else:
                result = base_frame
            
            # Convert back to uint8
            result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
            transition_frames.append(result_uint8)
        
        return transition_frames
    
    def _linear_blend(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        n_frames: int
    ) -> List[np.ndarray]:
        """Simple linear interpolation fallback."""
        frames = []
        
        frame_before_f = frame_before.astype(np.float32)
        frame_after_f = frame_after.astype(np.float32)
        
        for i in range(n_frames):
            t = (i + 1) / (n_frames + 1)
            blended = (1 - t) * frame_before_f + t * frame_after_f
            frames.append(blended.astype(np.uint8))
        
        return frames
    
    def generate_effect_overlay(
        self,
        frame: np.ndarray,
        embedding: np.ndarray,
        frame_idx: int,
        total_frames: int,
        intensity: float = 1.0
    ) -> np.ndarray:
        """
        Apply learned effect to a single frame (for on-beat effects).
        
        This is used when we want to apply a learned effect (like flash,
        zoom, shake) without actually transitioning between clips.
        """
        if self.neural_vae is None:
            return frame
        
        # Decode embedding
        motion_template = self._decode_embedding(embedding)
        
        if motion_template is None:
            return frame
        
        h, w = frame.shape[:2]
        template_h, template_w = motion_template.shape[1:3]
        
        # Resize frame
        frame_small = cv2.resize(frame, (template_w, template_h))
        frame_norm = frame_small.astype(np.float32) / 255.0
        
        # Get effect frame from template
        n_template = len(motion_template)
        progress = frame_idx / max(1, total_frames - 1)
        template_idx = int(progress * (n_template - 1))
        template_idx = min(template_idx, n_template - 1)
        
        # Apply effect
        motion_mod = (motion_template[template_idx] - 0.5) * 2
        modulated = frame_norm + motion_mod * intensity * 0.3
        modulated = np.clip(modulated, 0, 1)
        
        # Blend with original
        result = (1 - intensity * 0.3) * frame_norm + intensity * 0.3 * modulated
        result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        
        # Resize back
        return cv2.resize(result_uint8, (w, h))


class TransitionBlender:
    """
    Higher-level interface for blending clips with learned transitions.
    
    Takes care of:
    - Extracting relevant frames from clips
    - Applying neural transitions
    - Managing transition timing
    """
    
    def __init__(
        self,
        neural_vae=None,
        default_transition_frames: int = 5
    ):
        self.generator = NeuralEffectGenerator()
        if neural_vae is not None:
            self.generator.set_vae(neural_vae)
        self.default_transition_frames = default_transition_frames
    
    def blend_clips(
        self,
        clip1_frames: List[np.ndarray],
        clip2_frames: List[np.ndarray],
        embedding: Optional[np.ndarray] = None,
        transition_frames: int = None
    ) -> List[np.ndarray]:
        """
        Blend two clips with learned transition.
        
        Args:
            clip1_frames: Frames from first clip
            clip2_frames: Frames from second clip
            embedding: Transition embedding (use random if None)
            transition_frames: Number of transition frames
            
        Returns:
            Combined frame list with transition
        """
        n_trans = transition_frames or self.default_transition_frames
        
        if len(clip1_frames) == 0 or len(clip2_frames) == 0:
            return clip1_frames + clip2_frames
        
        # Get boundary frames
        frame_before = clip1_frames[-1]
        frame_after = clip2_frames[0]
        
        # Generate or sample embedding
        if embedding is None:
            if self.generator.neural_vae is not None:
                # Sample random transition
                embedding = np.random.randn(self.generator.neural_vae.latent_dim)
            else:
                embedding = None
        
        # Generate transition
        if embedding is not None:
            transition = self.generator.generate_transition_frames(
                frame_before, frame_after, embedding, n_trans
            )
        else:
            transition = self.generator._linear_blend(
                frame_before, frame_after, n_trans
            )
        
        # Combine: all of clip1 (except overlap) + transition + all of clip2 (except overlap)
        # For now, just insert transition frames
        overlap = min(n_trans // 2, len(clip1_frames) - 1, len(clip2_frames) - 1)
        
        result = clip1_frames[:-overlap] if overlap > 0 else clip1_frames
        result.extend(transition)
        result.extend(clip2_frames[overlap:] if overlap > 0 else clip2_frames)
        
        return result


def sample_transition_for_music(
    neural_vae,
    music_features: np.ndarray
) -> np.ndarray:
    """
    Sample a transition embedding based on music features.
    
    Args:
        neural_vae: Trained NeuralTransitionVAE
        music_features: Features at the transition point
        
    Returns:
        Latent embedding for transition generation
    """
    # Sample from prior, modulated by music energy
    energy = np.mean(music_features[:3]) if len(music_features) >= 3 else 0.5
    
    # Higher energy = higher variance samples
    std = 0.5 + energy * 0.5
    embedding = np.random.randn(neural_vae.latent_dim) * std
    
    return embedding
