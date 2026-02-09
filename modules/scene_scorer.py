"""
Scene Scorer - Extracts and scores scenes from a single input video.

Uses the same scene detection functions as edit2_multi.py and pipeline_interactive.py:
- detect_scenes(): Proxy-based scene detection with fallbacks
- get_motion_score(): Motion scoring for ranking scenes

This is the SAME CODE as edit2_multi.py, embedded directly to avoid import issues.
"""
import cv2
import numpy as np
import subprocess
import uuid
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

# Import scenedetect (same as edit2_multi.py)
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

from config import get_config
from utils.video_io import get_video_metadata


# ============================================================
# EXACT SAME CODE AS edit2_multi.py - detect_scenes function
# ============================================================
def _cleanup_proxy(proxy_path):
    """Cleanup proxy file."""
    if os.path.exists(proxy_path):
        try:
            os.remove(proxy_path)
        except:
            pass

def detect_scenes_edit2(video_path: str) -> List[Tuple[float, float]]:
    """
    Detects scenes using a temp low-res proxy (SAME as edit2_multi.py).
    """
    if not SCENEDETECT_AVAILABLE:
        return _fallback_grid_scenes(video_path)
    
    # Unique proxy name to avoid collisions
    uid = str(uuid.uuid4())[:8]
    proxy_path = f"/tmp/proxy_{uid}.mp4"

    try:
        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", "scale=-2:480",  # smaller for speed
            "-r", "10",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            proxy_path
        ]

        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        video = open_video(proxy_path, backend="opencv", framerate=10)
        
        # Initial Attempt
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
        scene_manager.detect_scenes(video)
        scenes = scene_manager.get_scene_list()

        if not scenes:
            # Fallback 1: Lower threshold
            print(f"   ⚠️ Standard detection failed. Retrying with sensitive threshold...")
            video = open_video(proxy_path, backend="opencv", framerate=10)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=10.0))
            scene_manager.detect_scenes(video)
            scenes = scene_manager.get_scene_list()

        if not scenes:
            # Fallback 2: Fixed Grid (Force Output)
            print(f"   ⚠️ Detection failed. Applying FIXED GRID segmentation as fallback.")
            return _fallback_grid_scenes(video_path)
        
        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
        
    except Exception as e:
        print(f"⚠️ Scene detection failed: {e}")
        return _fallback_grid_scenes(video_path)
        
    finally:
        _cleanup_proxy(proxy_path)


def _fallback_grid_scenes(video_path: str) -> List[Tuple[float, float]]:
    """Fixed grid fallback when detection fails."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frames / fps if fps > 0 else 10.0
        cap.release()
    except:
        duration = 10.0
    
    fixed_scenes = []
    curr = 0
    while curr < duration:
        fixed_scenes.append((curr, min(curr + 3.0, duration)))
        curr += 3.0
    return fixed_scenes


# ============================================================
# EXACT SAME CODE AS edit2_multi.py - get_motion_score function
# ============================================================
def get_motion_score_edit2(video_path: str, start: float, end: float) -> float:
    """
    Calculate motion score for a scene (SAME as edit2_multi.py).
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    prev = None
    total_diff = 0
    frames = 0
    max_frames = 100  # sample first 100 frames max per scene for speed

    while cap.get(cv2.CAP_PROP_POS_MSEC) < end * 1000 and frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (320, 180))  # downscale for speed
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        if prev is not None:
            total_diff += np.mean(cv2.absdiff(prev, gray))
        
        prev = gray
        frames += 1

    cap.release()
    return total_diff / frames if frames > 0 else 0

from config import get_config
from utils.video_io import get_video_metadata


@dataclass
class Scene:
    """A scene segment from the input video."""
    start_time: float       # Start time in seconds
    end_time: float         # End time in seconds
    duration: float         # Duration in seconds
    video_path: str         # Source video path
    
    # Scene scores (higher is better)
    motion_score: float = 0.0       # Motion intensity
    overall_score: float = 0.0      # Combined score
    
    @property
    def start_frame(self) -> int:
        """Approximate start frame (assuming 30fps)."""
        return int(self.start_time * 30)
    
    @property
    def end_frame(self) -> int:
        """Approximate end frame (assuming 30fps)."""
        return int(self.end_time * 30)


@dataclass
class ScoredScenes:
    """Collection of scored scenes from a video."""
    video_path: str
    scenes: List[Scene] = field(default_factory=list)
    fps: float = 30.0
    duration: float = 0.0
    
    def get_top_scenes(self, n: int = 10) -> List[Scene]:
        """Get top N scenes by overall score."""
        sorted_scenes = sorted(self.scenes, key=lambda s: s.overall_score, reverse=True)
        return sorted_scenes[:n]
    
    def get_scenes_for_duration(
        self,
        target_duration: float,
        min_scene_duration: float = 0.5
    ) -> List[Scene]:
        """
        Get enough top scenes to fill target duration.
        Recycles best scenes if pool exhausted (same as edit2_multi).
        """
        # Filter by minimum duration
        valid_scenes = [s for s in self.scenes if s.duration >= min_scene_duration]
        
        # Sort by score
        sorted_scenes = sorted(valid_scenes, key=lambda s: s.overall_score, reverse=True)
        
        # Accumulate until target duration
        selected = []
        total = 0.0
        idx = 0
        
        while total < target_duration:
            if not sorted_scenes:
                break
            
            # Use each unique scene first before recycling (SAME as edit2_multi.py)
            if idx < len(sorted_scenes):
                # Normal: Pick next best unique scene
                scene = sorted_scenes[idx]
            else:
                # RECYCLE: Only start recycling after pool exhausted
                scene = sorted_scenes[idx % len(sorted_scenes)]
            
            selected.append(scene)
            total += scene.duration
            idx += 1
            
            # Prevent infinite loop
            if idx > len(sorted_scenes) * 10:
                break
        
        return selected


class SceneScorer:
    """
    Extracts and scores scenes from a single input video.
    
    Uses the EXACT same functions as edit2_multi.py / pipeline_interactive.py:
    - detect_scenes_edit2(): Proxy-based scene detection with fallbacks
    - get_motion_score_edit2(): Motion-based scene scoring
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    def extract_and_score(
        self,
        video_path: str,
        min_scene_duration: float = 0.5,
        verbose: bool = True
    ) -> ScoredScenes:
        """
        Extract and score all scenes from a video.
        
        Uses detect_scenes_edit2() and get_motion_score_edit2()
        (SAME code as edit2_multi.py / pipeline_interactive.py)
        """
        metadata = get_video_metadata(video_path)
        fps = metadata.fps
        
        if verbose:
            print(f"Analyzing video: {video_path}")
            print(f"  Duration: {metadata.duration_seconds:.1f}s, {metadata.frame_count} frames")
        
        result = ScoredScenes(
            video_path=video_path,
            fps=fps,
            duration=metadata.duration_seconds
        )
        
        # Use detect_scenes (SAME as edit2_multi.py / pipeline_interactive.py)
        if verbose:
            print("  Detecting scene boundaries...")
        
        scene_times = detect_scenes_edit2(video_path)
        
        if verbose:
            print(f"  Found {len(scene_times)} potential scenes")
        
        # Score each scene using get_motion_score (SAME as edit2_multi.py)
        if verbose:
            print("  Scoring scenes...")
        
        valid_count = 0
        for start, end in scene_times:
            duration = end - start
            
            # Skip tiny scenes
            if duration < min_scene_duration:
                continue
            
            # Use motion scoring (SAME as edit2_multi.py)
            motion = get_motion_score_edit2(video_path, start, end)
            
            scene = Scene(
                start_time=start,
                end_time=end,
                duration=duration,
                video_path=video_path,
                motion_score=motion,
                overall_score=motion
            )
            
            result.scenes.append(scene)
            valid_count += 1
        
        if verbose:
            print(f"  Scored {valid_count} valid scenes")
            if result.scenes:
                top_scene = max(result.scenes, key=lambda s: s.overall_score)
                print(f"  Best scene: {top_scene.start_time:.1f}s-{top_scene.end_time:.1f}s (score: {top_scene.overall_score:.3f})")
        
        return result
        
        return result
    
    def _fallback_detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """Fallback scene detection if edit2_multi not available."""
        metadata = get_video_metadata(video_path)
        duration = metadata.duration_seconds or 10.0
        
        scenes = []
        curr = 0.0
        while curr < duration:
            scenes.append((curr, min(curr + 3.0, duration)))
            curr += 3.0
        
        return scenes
    
    def _fallback_motion_score(self, video_path: str, start: float, end: float) -> float:
        """Fallback motion scoring if edit2_multi not available."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        
        prev = None
        total_diff = 0
        frames = 0
        max_frames = 100
        
        while cap.get(cv2.CAP_PROP_POS_MSEC) < end * 1000 and frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            small = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            
            if prev is not None:
                total_diff += np.mean(cv2.absdiff(prev, gray))
            
            prev = gray
            frames += 1
        
        cap.release()
        return total_diff / frames if frames > 0 else 0


def extract_best_scenes(
    video_path: str,
    n_scenes: int = 10,
    verbose: bool = True
) -> List[Scene]:
    """
    Convenience function to extract the best N scenes from a video.
    Uses edit2_multi's detect_scenes and get_motion_score.
    """
    scorer = SceneScorer()
    result = scorer.extract_and_score(video_path, verbose=verbose)
    return result.get_top_scenes(n_scenes)


def get_scenes_for_duration(
    video_path: str,
    target_duration: float,
    min_scene_duration: float = 0.5,
    verbose: bool = True
) -> List[Scene]:
    """
    Get enough best scenes to fill target duration.
    Same recycling logic as edit2_multi.py.
    """
    scorer = SceneScorer()
    result = scorer.extract_and_score(video_path, min_scene_duration=min_scene_duration, verbose=verbose)
    return result.get_scenes_for_duration(target_duration, min_scene_duration)
