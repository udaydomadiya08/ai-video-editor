"""
Auto Editor - Generates final video using learned models.

This is the main video generation engine that:
1. Analyzes new music to extract beats/energy
2. Samples cut timing from learned distribution
3. Selects and orders clips from raw footage
4. Samples transitions based on music context
5. Applies learned transitions to create final video

All editing decisions are driven by learned statistics.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import tempfile
import subprocess
import os

from config import get_config
from utils.video_io import (
    get_video_metadata, iter_frames, extract_frame_range,
    get_all_videos, VideoWriter, crop_to_aspect
)
from utils.audio_io import get_audio_duration
from modules.style_learner import StyleParameters
from modules.music_analyzer import MusicAnalyzer, MusicFeatures
from modules.scene_scorer import SceneScorer, Scene, ScoredScenes

# Neural transition learning (learns ANY effect)
try:
    from modules.neural_transition_vae import NeuralTransitionVAE
    from modules.neural_effect_generator import NeuralEffectGenerator
    NEURAL_TRANSITIONS_AVAILABLE = True
except ImportError:
    NEURAL_TRANSITIONS_AVAILABLE = False

# Deep style transfer
try:
    from modules.neural_style_transfer import StyleTransfer
    STYLE_TRANSFER_AVAILABLE = True
except ImportError:
    STYLE_TRANSFER_AVAILABLE = False


@dataclass
class ClipInfo:
    """Information about a raw clip."""
    path: str
    start_frame: int      # For backwards compat
    end_frame: int        # For backwards compat
    motion_score: float
    fps: float
    duration: float
    start_time: float = 0.0   # Time in seconds (used by edit2-style rendering)
    end_time: float = 0.0     # Time in seconds (used by edit2-style rendering)


@dataclass
class EditSegment:
    """A segment in the final edit."""
    clip: ClipInfo
    start_time: float  # In output timeline
    duration: float
    transition_before: Optional[str] = None # "neural" or None
    transition_after: Optional[str] = None


class AutoEditor:
    """
    Generates edited videos using learned style models.
    
    All editing decisions are based on:
    - Learned shot length distribution
    - Learned beat-cut alignment
    - Learned transition embeddings
    - Music-transition mapping
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # Learned components
        # Learned components
        self.style_params: Optional[StyleParameters] = None
        
        # Neural components
        self.neural_vae = None
        self.neural_generator = None
        
        # Deep style transfer
        self.style_transfer = None
        if STYLE_TRANSFER_AVAILABLE:
            self.style_transfer = StyleTransfer(self.config)
        
        # Analysis components
        self.music_analyzer = MusicAnalyzer(self.config)
    
    def load_models(
        self,
        style_path: str,
        neural_vae_path: Optional[str] = None,
        deep_style_path: Optional[str] = None,
        neural_effect_path: Optional[str] = None
    ):
        """Load learned neural models."""
        # Load Style Parameters (Shot lengths, cuts/sec)
        # This is purely statistical (no hardcoded visual effects)
        if os.path.exists(style_path):
             self.style_params = StyleParameters.load(style_path)
        
        
        # Load deep style VAE if available
        if deep_style_path is None:
            import os
            models_dir = os.path.dirname(style_path)
            deep_style_path = os.path.join(models_dir, 'deep_style_vae.pt')
            
        if self.style_transfer and os.path.exists(deep_style_path):
            self.style_transfer.load_deep_style(deep_style_path)
            # In a real system we'd load reference patches here too
            # For now we assume they are handled or we use generic style
        
        # Load neural transition VAE if available
        
        # Load neural transition VAE if available
        if neural_vae_path is None:
            # Try default path
            import os
            models_dir = os.path.dirname(style_path)
            neural_vae_path = os.path.join(models_dir, 'neural_transition_vae.pt')
        
        if NEURAL_TRANSITIONS_AVAILABLE and os.path.exists(neural_vae_path):
            try:
                self.neural_vae = NeuralTransitionVAE(self.config)
                self.neural_vae.load(neural_vae_path)
                self.neural_generator = NeuralEffectGenerator(self.config)
                self.neural_generator.set_vae(self.neural_vae)
                print(f"  ✓ Loaded neural transition VAE (learns ANY effect!)")
            except Exception as e:
                print(f"  ⚠️ Could not load neural VAE: {e}")
                self.neural_vae = None
                self.neural_generator = None
        
        # Load Universal Effect Model (Flow + Intensity)
        self.effect_model = None
        self.effect_refs = []
        if style_path:
             models_dir = os.path.dirname(style_path)
             effect_model_path = os.path.join(models_dir, 'neural_effect_vae.pt')
             effect_refs_path = os.path.join(models_dir, 'effect_refs.pkl')
             
             if os.path.exists(effect_model_path) and os.path.exists(effect_refs_path):
                 try:
                     from modules.neural_effect_vae import UniversalEffectModel
                     self.effect_model = UniversalEffectModel(self.config)
                     self.effect_model.load(effect_model_path)
                     
                     import pickle
                     with open(effect_refs_path, 'rb') as f:
                         self.effect_refs = pickle.load(f)
                     print(f"  ✓ Loaded Universal Effect Model and {len(self.effect_refs)} refs")
                 except Exception as e:
                     print(f"  ⚠️ Could not load Universal Effect Model: {e}")
    
    def generate(
        self,
        raw_clips_dir: str,
        music_path: str,
        output_path: str,
        verbose: bool = True
    ) -> str:
        """
        Generate an edited video.
        
        Args:
            raw_clips_dir: Directory containing raw clips
            music_path: Path to music file
            output_path: Path for output video
            verbose: Show progress
            
        Returns:
            Path to generated video
        """
        if self.style_params is None:
            raise RuntimeError("Models not loaded. Call load_models first.")
        
        # Step 1: Analyze music
        if verbose:
            print("=== Step 1: Analyzing music ===")
        
        music_features = self.music_analyzer.analyze(music_path, verbose=verbose)
        music_duration = music_features.duration_seconds
        
        # Step 2: Load and score raw clips
        if verbose:
            print("\n=== Step 2: Analyzing raw clips ===")
        
        clips = self._load_and_score_clips(raw_clips_dir, verbose)
        
        if not clips:
            raise ValueError(f"No valid clips found in {raw_clips_dir}")
        
        if verbose:
            print(f"Found {len(clips)} clips")
        
        # Step 3: Plan the edit timeline
        if verbose:
            print("\n=== Step 3: Planning edit timeline ===")
        
        segments = self._plan_timeline(clips, music_features, verbose)
        
        if verbose:
            print(f"Planned {len(segments)} segments")
        
        # Step 4: Sample transitions for each cut
        if verbose:
            print("\n=== Step 4: Sampling transitions ===")
        
        segments = self._assign_transitions(segments, music_features, verbose)
        
        # Step 5: Render the video
        if verbose:
            print("\n=== Step 5: Rendering video ===")
        
        temp_video = self._render_video(segments, music_features, verbose)
        
        # Step 6: Add audio
        if verbose:
            print("\n=== Step 6: Adding audio ===")
        
        self._add_audio(temp_video, music_path, output_path, music_duration)
        
        # Cleanup
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        if verbose:
            print(f"\n=== Done! Output: {output_path} ===")
        
        return output_path
    
    def generate_from_video(
        self,
        input_video: str,
        music_path: str,
        output_path: str,
        verbose: bool = True
    ) -> str:
        """
        Generate an edited video from a SINGLE input video.
        
        Extracts best scenes from the input video and assembles them
        with beat-synced cuts and learned transitions.
        
        Args:
            input_video: Path to single source video
            music_path: Path to music file
            output_path: Path for output video
            verbose: Show progress
            
        Returns:
            Path to generated video
        """
        if self.style_params is None:
            raise RuntimeError("Models not loaded. Call load_models first.")
        
        # Step 1: Analyze music
        if verbose:
            print("=== Step 1: Analyzing music ===")
        
        music_features = self.music_analyzer.analyze(music_path, verbose=verbose)
        music_duration = music_features.duration_seconds
        
        # Step 2: Extract and score scenes from input video
        if verbose:
            print("\n=== Step 2: Extracting scenes from video ===")
        
        scene_scorer = SceneScorer(self.config)
        scored_scenes = scene_scorer.extract_and_score(input_video, verbose=verbose)
        
        if not scored_scenes.scenes:
            raise ValueError(f"No valid scenes found in {input_video}")
        
        if verbose:
            print(f"Found {len(scored_scenes.scenes)} scenes")
        
        # Get ALL scenes sorted by score (best to worst) - SAME as edit2_multi.py
        # edit2_multi builds a global pool of ALL scenes, sorts by score, then picks from it
        # We don't limit the pool - we use ALL available scenes
        all_scenes_sorted = sorted(
            scored_scenes.scenes,
            key=lambda s: s.overall_score,
            reverse=True  # Best first
        )
        selected_scenes = all_scenes_sorted  # Use ALL scenes as the pool
        
        if verbose:
            print(f"Selected {len(selected_scenes)} scenes for {music_duration:.1f}s track")
        
        # Convert Scene objects to ClipInfo for compatibility
        # Uses TIME-BASED start/end like edit2_multi.py
        clips = []
        for scene in selected_scenes:
            clips.append(ClipInfo(
                path=input_video,  # All from same video
                start_frame=scene.start_frame,
                end_frame=scene.end_frame,
                motion_score=scene.overall_score,
                fps=scored_scenes.fps,
                duration=scene.duration,
                start_time=scene.start_time,  # Time in seconds (edit2 style)
                end_time=scene.end_time        # Time in seconds (edit2 style)
            ))
        
        # Step 3: Plan the edit timeline
        if verbose:
            print("\n=== Step 3: Planning edit timeline ===")
        
        segments = self._plan_timeline(clips, music_features, verbose)
        
        if verbose:
            print(f"Planned {len(segments)} segments")
        
        # Step 4: Sample transitions for each cut
        if verbose:
            print("\n=== Step 4: Sampling transitions ===")
        
        segments = self._assign_transitions(segments, music_features, verbose)
        
        # Step 5: Render the video
        if verbose:
            print("\n=== Step 5: Rendering video ===")
        
        temp_video = self._render_video(segments, music_features, verbose)
        
        # Step 6: Add audio
        if verbose:
            print("\n=== Step 6: Adding audio ===")
        
        self._add_audio(temp_video, music_path, output_path, music_duration)
        
        # Cleanup
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        if verbose:
            print(f"\n=== Done! Output: {output_path} ===")
        
        return output_path
    
    def _load_and_score_clips(
        self,
        clips_dir: str,
        verbose: bool = True
    ) -> List[ClipInfo]:
        """Load clips and calculate motion scores."""
        video_paths = get_all_videos(clips_dir)
        clips = []
        
        for path in tqdm(video_paths, desc="Scoring clips", disable=not verbose):
            try:
                metadata = get_video_metadata(path)
                
                # Calculate motion score
                motion_score = self._calculate_motion_score(path)
                
                clips.append(ClipInfo(
                    path=path,
                    start_frame=0,
                    end_frame=metadata.frame_count,
                    motion_score=motion_score,
                    fps=metadata.fps,
                    duration=metadata.duration_seconds
                ))
            except Exception as e:
                if verbose:
                    print(f"  Skipping {path}: {e}")
        
        # Sort by motion score (best first)
        clips.sort(key=lambda c: c.motion_score, reverse=True)
        
        return clips
    
    def _calculate_motion_score(self, video_path: str) -> float:
        """Calculate motion intensity score for a clip."""
        motion_values = []
        prev_frame = None
        
        for frame_idx, frame in iter_frames(
            video_path,
            sample_rate=5,  # Sample every 5 frames
            grayscale=True,
            resize=(160, 90)
        ):
            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, frame, None,
                    pyr_scale=0.5, levels=2, winsize=15,
                    iterations=2, poly_n=5, poly_sigma=1.2, flags=0
                )
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_values.append(np.mean(mag))
            
            prev_frame = frame
            
            # Limit analysis time
            if len(motion_values) > 50:
                break
        
        return float(np.mean(motion_values)) if motion_values else 0.0
    
    def _plan_timeline(
        self,
        clips: List[ClipInfo],
        music: MusicFeatures,
        verbose: bool = True
    ) -> List[EditSegment]:
        """Plan the edit timeline based on learned style."""
        segments = []
        current_time = 0.0
        music_duration = music.duration_seconds
        clip_idx = 0
        
        # Get beat-aligned cut times
        beat_offset = self.style_params.beat_cut_alignment_mean
        cut_times = music.beat_times + beat_offset
        
        # Filter to valid range
        cut_times = cut_times[(cut_times > 0) & (cut_times < music_duration)]
        
        # Sample shot lengths and align to beats
        while current_time < music_duration:
            # Sample shot length from learned distribution
            dist_type = self.style_params.shot_length_distribution
            
            if dist_type == "raw_kde" and 'kde_samples' in self.style_params.shot_length_params:
                # Raw Learning: Sample from pre-computed KDE buffer
                samples = self.style_params.shot_length_params['kde_samples']
                duration = float(np.random.choice(samples))
            elif dist_type and dist_type != "raw_kde":
                # Fallback for old models (Legacy)
                try:
                    from scipy import stats
                    dist = getattr(stats, dist_type)
                    params = self.style_params.shot_length_params['params']
                    duration = float(dist.rvs(*params))
                    duration = np.clip(
                        duration,
                        self.style_params.shot_length_params['min'] * 0.5,
                        self.style_params.shot_length_params['max'] * 1.5
                    )
                except:
                     duration = 1.0 / max(0.1, self.style_params.avg_cuts_per_second)
            else:
                # Fallback to mean
                duration = 1.0 / max(0.1, self.style_params.avg_cuts_per_second)
            
            # Snap to nearest beat
            end_time = current_time + duration
            if len(cut_times) > 0:
                distances = np.abs(cut_times - end_time)
                nearest_beat_idx = np.argmin(distances)
                
                # Only snap if close enough
                if distances[nearest_beat_idx] < duration * 0.3:
                    end_time = cut_times[nearest_beat_idx]
            
            # Ensure we don't exceed music duration
            end_time = min(end_time, music_duration)
            actual_duration = end_time - current_time
            
            if actual_duration < 0.1:
                break
            
            # Select a clip - use each unique scene before recycling (SAME as edit2_multi.py)
            if clip_idx < len(clips):
                # Normal: Pick next best unique scene
                clip = clips[clip_idx]
            else:
                # RECYCLE: Start over from best (only when pool exhausted)
                # Log once when recycling starts
                if clip_idx == len(clips):
                    pass  # Could log "Recycling..." here
                clip = clips[clip_idx % len(clips)]
            clip_idx += 1
            
            segments.append(EditSegment(
                clip=clip,
                start_time=current_time,
                duration=actual_duration
            ))
            
            current_time = end_time
        
        return segments
    
    def _assign_transitions(
        self,
        segments: List[EditSegment],
        music: MusicFeatures,
        verbose: bool = True
    ) -> List[EditSegment]:
        """
        Assign transitions to each segment.
        If Neural VAE is available, we simply mark them for neural generation.
        """
        if self.neural_vae:
            # Neural VAE handles everything at render time
            # We just need to mark which segments need transitions
            for i, segment in enumerate(segments):
                if i > 0:
                    # Mark for transition (store dummy object or flag)
                    # We'll generate the actual pixel transition in _render_video
                    segment.transition_before = "neural" 
            return segments
            
        return segments
    
    def _render_video(
        self,
        segments: List[EditSegment],
        music: MusicFeatures,
        verbose: bool = True
    ) -> str:
        """
        Render all segments to a video file.
        
        Uses MoviePy's subclip() and concatenate_videoclips() 
        (SAME approach as edit2_multi.py to avoid black frames)
        
        (SAME approach as edit2_multi.py to avoid black frames)
        """
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        
        # Create temp output
        temp_video = tempfile.mktemp(suffix='.mp4')
        
        # Cache video clips to avoid reloading (same as edit2_multi.py)
        video_clips_cache = {}
        
        def get_cached_clip(path):
            if path not in video_clips_cache:
                video_clips_cache[path] = VideoFileClip(path).without_audio()
            return video_clips_cache[path]
        
        # Build clip list using subclip (SAME as edit2_multi.py)
        clips = []
        clip_timings = []  # Track (clip_start_in_timeline, clip_duration)
        current_timeline_pos = 0.0
        
        for segment in tqdm(segments, desc="Rendering", disable=not verbose):
            try:
                # Get source video
                src_clip = get_cached_clip(segment.clip.path)
                
                # Apply Neural Style Transfer if enable (and model loaded)
                # Note: Applying to MoviePy clip is tricky because we need numpy arrays
                # We can use fl_image but deep style works best on batches
                # For now let's apply per-frame if deep style is active
                
                # Use time-based start/end directly (SAME as edit2_multi.py)
                start_time = segment.clip.start_time
                end_time = segment.clip.end_time
                
                # Fallback to frame-based calculation if time fields not set
                if start_time == 0.0 and end_time == 0.0:
                    start_time = segment.clip.start_frame / segment.clip.fps
                    end_time = segment.clip.end_frame / segment.clip.fps
                
                # Clamp to valid range
                start_time = max(0, start_time)
                end_time = min(src_clip.duration, end_time)
                
                # Duration we need for this segment
                required_duration = segment.duration
                available_duration = end_time - start_time
                
                # Take subclip (SAME as edit2_multi.py)
                seg_len = min(required_duration, available_duration)
                if seg_len > 0.1:  # Skip tiny clips
                    subclip = src_clip.subclip(start_time, start_time + seg_len)
                    
                    # Apply Style Transfer
                    if self.style_transfer and self.style_transfer.deep_style_model:
                        # Use deep style transfer (frame by frame for now)
                        # To optimize, we should process in batches, but fl_image is per-frame
                        # We use the single-frame wrapper we added to StyleTransfer
                        def apply_style_filter(frame):
                            return self.style_transfer.apply_frame_style(frame)
                        subclip = subclip.fl_image(apply_style_filter)
                    
                    # Apply Universal Visual Effects (Flow + Intensity)
                    if self.effect_model and len(self.effect_refs) > 0:
                        import random
                        # Pick a random effect style (Field Grid sequence)
                        style_grid = random.choice(self.effect_refs) # (3, 64, 9, 16)
                        
                        # Generate target field
                        # Content = 0 (Static, Constant Brightness)
                        T_style = style_grid.shape[1] # 64 typically
                        content_grid = np.zeros_like(style_grid)
                        
                        # Run transfer to get "Pure Style Dynamics"
                        # FORCE strength to 1.0 or even 1.2 for visible effect
                        generated_field = self.effect_model.transfer_effect(content_grid, style_grid, strength=1.2)
                        # generated_field: (3, T, H, W)
                        
                        duration = seg_len
                        
                        def apply_effect(get_frame, t):
                            frame = get_frame(t)
                            h_img, w_img = frame.shape[:2]
                            
                            # Map t to index
                            norm_t = np.clip(t / duration, 0, 1)
                            idx = int(norm_t * (generated_field.shape[1] - 1))
                            
                            # Get field grid for this frame
                            # shape (3, H_grid, W_grid)
                            field_grid_frame = generated_field[:, idx] 
                            
                            # Scan to full res
                            # (3, 9, 16) -> (9, 16, 3)
                            field_low = field_grid_frame.transpose(1, 2, 0)
                            
                            # Upsample
                            field_full = cv2.resize(field_low, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                            
                            # 1. Apply Flow (Channels 0, 1)
                            # Resize magnitudes
                            scale_x = w_img / 64.0
                            scale_y = h_img / 36.0
                            
                            field_full[..., 0] *= scale_x
                            field_full[..., 1] *= scale_y
                            
                            grid_y, grid_x = np.mgrid[0:h_img, 0:w_img]
                            map_x = (grid_x - field_full[..., 0]).astype(np.float32)
                            map_y = (grid_y - field_full[..., 1]).astype(np.float32)
                            
                            warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                            
                            # 2. Apply Intensity (Channel 2)
                            # field_full[..., 2] is delta brightness [-1, 1]
                            intensity_diff = field_full[..., 2]
                            
                            # Convert to float
                            warped_f = warped.astype(np.float32) / 255.0
                            
                            # Add diff (replicate to 3 channels if RGB)
                            intensity_diff_3c = np.repeat(intensity_diff[:, :, np.newaxis], 3, axis=2)
                            
                            # Apply
                            final_f = np.clip(warped_f + intensity_diff_3c, 0, 1)
                            
                            return (final_f * 255).astype(np.uint8)
                            
                        subclip = subclip.fl(apply_effect)

                    # Apply Detail / Sharpness Matching (High Frequency Transfer)
                    # "Consume it like anything... video quality 4k"
                    subclip = subclip.fl_image(lambda frame: self._apply_detail_match(frame))
                        
                    clips.append(subclip)
                    clip_timings.append((current_timeline_pos, seg_len))
                    current_timeline_pos += seg_len
                    
            except Exception as e:
                if verbose:
                    print(f"⚠️ Failed to load segment, skipping: {e}")
        
        if not clips:
            raise ValueError("No valid clips to render!")
        
        # Concatenate all clips (SAME as edit2_multi.py)
        final = concatenate_videoclips(clips, method="compose")
        
        # Detect source FPS (SAME as edit2_multi.py)
        source_fps = self.config.output_fps
        if segments:
            source_fps = segments[0].clip.fps or source_fps
        
        # Apply LEARNED visual effects if available
        # Apply LEARNED visual effects (Neural VAE)
        # We already applied the Neural Effect Model above (lines 643+).
        # We do NOT want to apply hardcoded effects separately.
        # The user requested fully learned dynamics.
        
        
        
        # Write output
        final.write_videofile(
            temp_video,
            codec="libx264",
            audio_codec="aac",
            fps=source_fps,
            preset="fast",
            ffmpeg_params=["-crf", "18"],
            threads=4,
            logger=None
        )
        
        # Close cache
        for c in video_clips_cache.values():
            try:
                c.close()
            except:
                pass
        
        try:
            final.close()
        except:
            pass
        
        return temp_video
    
    def _apply_learned_field(self, frame: np.ndarray, field: np.ndarray) -> np.ndarray:
        """
        Apply a raw learned field (Flow + Intensity) to a frame.
        field shape: (3, H_grid, W_grid)
        """
        h_img, w_img = frame.shape[:2]
        
        # 1. Upsample field to video resolution
        # field is (3, 128, 128) typically
        field_low = field.transpose(1, 2, 0) # (H, W, 3)
        field_full = cv2.resize(field_low, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        
        # 2. Extract Flow (Ch 0, 1) and Intensity (Ch 2)
        # Flow was normalized to -1..1 representing -20..20 px range
        # We need to scale it up
        flow_scale = 20.0
        # Also scale by resolution ratio if we trained on 128x128 but rendering 1080p?
        # The flow is relative to the *image* structure.
        # If we trained on 128px, 20px is ~15% of width.
        # On 1920px, 15% is ~300px.
        # So we should scale by (width / 128.0)
        res_scale_x = w_img / 128.0
        res_scale_y = h_img / 128.0
        
        flow_x = field_full[..., 0] * 20.0 * res_scale_x
        flow_y = field_full[..., 1] * 20.0 * res_scale_y
        
        intensity = field_full[..., 2] # 0..1 (diff luminance)
        
        # 3. Apply Warp (Optical Flow)
        grid_y, grid_x = np.mgrid[0:h_img, 0:w_img]
        map_x = (grid_x - flow_x).astype(np.float32)
        map_y = (grid_y - flow_y).astype(np.float32)
        
        warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 4. Apply Intensity (Add difference)
        # The VAE learned absolute difference 0..1
        # We assume it represents a flash/fade *addition* or *subtraction*?
        # Extractor used: intensity = grayscale(diff). Diff is abs(a-b).
        # So it's always positive 0..1.
        # We can add it as a "Flash" component
        intensity_3c = np.repeat(intensity[:, :, np.newaxis], 3, axis=2)
        final = warped.astype(np.float32) / 255.0 + (intensity_3c * 1.5) # Boost slightly
        
        return np.clip(final * 255, 0, 255).astype(np.uint8)

    def _render_segment(
        self,
        writer: VideoWriter,
        segment: EditSegment,
        width: int,
        height: int,
        fps: float,
        target_aspect: float
    ):
        """Render a single segment to the video writer."""
        clip = segment.clip
        n_output_frames = int(segment.duration * fps)
        
        # Calculate source frame range
        clip_total_frames = clip.end_frame - clip.start_frame
        clip_duration = clip_total_frames / clip.fps
        
        # If clip is shorter than needed, loop it
        if clip_duration < segment.duration:
            # Loop: map output frames to clip frames with modulo
            frame_map = lambda f: clip.start_frame + int((f / fps * clip.fps) % clip_total_frames)
        else:
            # Use portion of clip
            start_offset = np.random.uniform(0, max(0, clip_duration - segment.duration))
            frame_map = lambda f: clip.start_frame + int((start_offset + f / fps) * clip.fps)
        
        if segment.transition_before == "neural":
            trans_frames = min(20, n_output_frames // 3) # Default 20 frames for neural
            
        # Pre-generate Neural Transition Sequence
        neural_fields = None
        if segment.transition_before == "neural" and self.neural_vae:
            try:
                # Generate random latent
                import torch
                with torch.no_grad():
                    # "God Mode" Latent Dimension = 128
                    z = torch.randn(1, 128).to(self.config.device)
                    # Decode -> (1, 3, T, H, W)
                    decoded = self.neural_vae.decode(z).cpu().numpy()
                    neural_fields = decoded[0] # (3, T, H, W)
                    trans_frames = neural_fields.shape[1] # T (usually 10)
            except Exception as e:
                print(f"VAE Error: {e}")
                pass
        
        # Extract and render frames
        cap = cv2.VideoCapture(clip.path)
        last_good_frame = None  # Keep last good frame to avoid black frames
        
        for out_frame in range(n_output_frames):
            src_frame = frame_map(out_frame)
            src_frame = min(src_frame, clip.end_frame - 1)
            src_frame = max(0, src_frame)  # Ensure non-negative
            
            # Try frame-based seeking first
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
            ret, frame = cap.read()
            
            # Fallback: try time-based seeking
            if not ret:
                time_sec = src_frame / clip.fps
                cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
                ret, frame = cap.read()
            
            # Fallback: reopen video and seek
            if not ret:
                cap.release()
                cap = cv2.VideoCapture(clip.path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
                ret, frame = cap.read()
            
            if ret and frame is not None:
                # Crop to target aspect ratio
                frame = crop_to_aspect(frame, target_aspect)
                frame = cv2.resize(frame, (width, height))
                last_good_frame = frame.copy()
                
                # Apply transition at start of segment
                if segment.transition_before:
                    if segment.transition_before == "neural" and neural_fields is not None:
                         # Apply RAW Learned Field
                         if out_frame < neural_fields.shape[1]:
                             # Get (3, H, W) field for this frame
                             field = neural_fields[:, out_frame]
                             frame = self._apply_learned_field(frame, field)
                             

            else:
                # Use last good frame instead of black
                if last_good_frame is not None:
                    frame = last_good_frame.copy()
                else:
                    # Only use black as absolute last resort
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            writer.write(frame)
        
        cap.release()
    

    
    def _apply_detail_match(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply detail enhancement AND slight saturation boost.
        Matches professional 4K look.
        """
        # 1. Unsharp Mask for Detail (High Frequency)
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        unsharp_image = cv2.addWeighted(frame, 1.4, gaussian, -0.4, 0)
        
        # 2. Slight Vibrance Boost (Saturation)
        hsv = cv2.cvtColor(unsharp_image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.15  # Boost saturation by 15%
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        final = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return final

    def _add_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        duration: float
    ):
        """Add audio to the video using ffmpeg."""
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-t', str(duration),
            '-shortest',
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
