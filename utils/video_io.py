"""
Video I/O utilities for the AI Video Editing Style Learning System.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, List
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class VideoMetadata:
    """Metadata for a video file."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Extract metadata from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return VideoMetadata(
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration
    )


def iter_frames(
    video_path: str,
    sample_rate: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    grayscale: bool = False,
    resize: Optional[Tuple[int, int]] = None,
    show_progress: bool = False
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Iterate over video frames with optional sampling and preprocessing.
    
    Args:
        video_path: Path to video file
        sample_rate: Yield every N frames
        start_frame: Starting frame index
        end_frame: Ending frame index (exclusive), None for all
        grayscale: Convert to grayscale
        resize: Resize to (width, height)
        show_progress: Show progress bar
        
    Yields:
        Tuple of (frame_index, frame_array)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = end_frame or total_frames
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_range = range(start_frame, min(end_frame, total_frames))
    if show_progress:
        frame_range = tqdm(frame_range, desc="Reading frames")
    
    for frame_idx in frame_range:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate != 0:
            continue
        
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if resize:
            frame = cv2.resize(frame, resize)
        
        yield frame_idx, frame
    
    cap.release()


def extract_frame_at(video_path: str, frame_idx: int) -> np.ndarray:
    """Extract a single frame from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")
    
    return frame


def extract_frame_range(
    video_path: str,
    start_frame: int,
    end_frame: int,
    grayscale: bool = False
) -> List[np.ndarray]:
    """Extract a range of frames from a video."""
    frames = []
    for _, frame in iter_frames(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        grayscale=grayscale
    ):
        frames.append(frame)
    return frames


def get_all_videos(directory: str, extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv')) -> List[str]:
    """Get all video files in a directory."""
    directory = Path(directory)
    videos = []
    for ext in extensions:
        videos.extend(directory.glob(f"*{ext}"))
        videos.extend(directory.glob(f"*{ext.upper()}"))
    return sorted([str(v) for v in videos])


class VideoWriter:
    """Context manager for writing video files."""
    
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = 'mp4v'
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.writer = None
    
    def __enter__(self):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.release()
    
    def write(self, frame: np.ndarray):
        """Write a frame to the video."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)


def crop_to_aspect(
    frame: np.ndarray,
    target_aspect: float,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None
) -> np.ndarray:
    """
    Crop a frame to a target aspect ratio.
    
    Args:
        frame: Input frame
        target_aspect: Target aspect ratio (width/height)
        center_x: Crop center x (0-1), None for center
        center_y: Crop center y (0-1), None for center
        
    Returns:
        Cropped frame
    """
    h, w = frame.shape[:2]
    current_aspect = w / h
    
    if current_aspect > target_aspect:
        # Too wide, crop width
        new_w = int(h * target_aspect)
        x_center = int(w * (center_x if center_x else 0.5))
        x_start = max(0, min(w - new_w, x_center - new_w // 2))
        return frame[:, x_start:x_start + new_w]
    else:
        # Too tall, crop height
        new_h = int(w / target_aspect)
        y_center = int(h * (center_y if center_y else 0.5))
        y_start = max(0, min(h - new_h, y_center - new_h // 2))
        return frame[y_start:y_start + new_h, :]
