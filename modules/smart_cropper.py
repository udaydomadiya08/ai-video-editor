"""
Smart Cropper Module - AI-powered 9:16 video cropping with subject tracking.

Uses YOLOv8 for person detection and MediaPipe for face detection,
with a spring-damped camera system for smooth, cinematic tracking.
"""
import cv2
import numpy as np
import sys
from typing import List, Tuple, Optional
from moviepy.editor import VideoFileClip, ImageSequenceClip

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Person detection disabled.")

try:
    from mediapipe.python.solutions import face_detection as mp_face
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not installed. Face detection disabled.")

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

# ===============================
# OUTPUT SETTINGS (VERTICAL)
# ===============================
OUT_W = 1080
OUT_H = 1920

# ===============================
# DETECTION TUNING
# ===============================
MIN_FACE_AREA = 0.001
MIN_PERSON_AREA = 0.04
CONF_THRESH = 0.35


class CinematicTracker:
    """
    AI-powered subject tracker with spring-damped camera physics.
    
    Prioritizes faces, falls back to person detection.
    Uses physics-based smoothing to avoid jitter.
    """
    
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h

        # Camera center (float for precision)
        self.cx = float(w // 2)
        self.cy = float(h // 2)

        # Velocity (for damping)
        self.vx = 0.0
        self.vy = 0.0

        # Physics tuning (KEY for smoothness)
        self.spring = 0.015      # lower = smoother
        self.damping = 0.80      # higher = less jitter
        self.max_speed = 12.0    # px per frame (hard cap)

        # Dead-zone (ignore micro movement)
        self.deadzone_x = int(w * 0.04)
        self.deadzone_y = int(h * 0.04)

        # Detection hold
        self.hold_frames = 0
        self.target_x = self.cx
        self.target_y = self.cy

        # Initialize detectors
        if MEDIAPIPE_AVAILABLE:
            self.face = mp_face.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.6
            )
        else:
            self.face = None
            
        if YOLO_AVAILABLE:
            self.yolo = YOLO("yolov8n.pt")
        else:
            self.yolo = None

    def reset(self):
        """Reset tracker to center."""
        self.cx = float(self.w // 2)
        self.cy = float(self.h // 2)
        self.vx = self.vy = 0.0
        self.target_x = self.cx
        self.target_y = self.cy
        self.hold_frames = 0

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect largest face, return center coordinates."""
        if self.face is None:
            return None
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face.process(rgb)
        if not res.detections:
            return None

        best = None
        best_area = 0
        for d in res.detections:
            r = d.location_data.relative_bounding_box
            area = r.width * r.height
            if area > best_area:
                cx = int((r.xmin + r.width / 2) * self.w)
                cy = int((r.ymin + r.height / 2) * self.h)
                best = (cx, cy)
                best_area = area

        return best

    def detect_person(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect largest person, return center coordinates."""
        if self.yolo is None:
            return None
            
        y = self.yolo(frame, verbose=False)[0]
        best = None
        best_score = 0

        for b in y.boxes:
            if self.yolo.names[int(b.cls[0])] != "person":
                continue
            conf = float(b.conf[0])
            if conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            score = conf * area
            if score > best_score:
                best_score = score
                best = ((x1 + x2) // 2, (y1 + y2) // 2)

        return best

    def update(self, frame: np.ndarray) -> Tuple[int, int]:
        """
        Update camera position based on detected subject.
        Returns (cx, cy) center position for cropping.
        """
        # Face first, person fallback
        detected = self.detect_face(frame)
        if not detected:
            detected = self.detect_person(frame)

        if detected:
            tx, ty = detected

            dx = tx - self.target_x
            dy = ty - self.target_y

            # --- SOFT DEADZONE ---
            dzx = self.deadzone_x
            dzy = self.deadzone_y

            move_x = abs(dx) > dzx
            move_y = abs(dy) > dzy

            # --- INTENT CONFIRMATION ---
            if move_x or move_y:
                self.hold_frames += 1
            else:
                self.hold_frames = max(0, self.hold_frames - 1)

            # Only accept new target if movement is REAL
            if self.hold_frames >= 5:
                # Partial update (prevents snapping)
                self.target_x = self.target_x * 0.85 + tx * 0.15
                self.target_y = self.target_y * 0.85 + ty * 0.15
                self.hold_frames = 0
        else:
            self.hold_frames = max(0, self.hold_frames - 1)

        # -------- CAMERA PHYSICS --------
        dx = self.target_x - self.cx
        dy = self.target_y - self.cy

        # SPRING (smooth but free)
        self.vx += dx * self.spring
        self.vy += dy * self.spring

        # DAMPING (kills jitter)
        self.vx *= self.damping
        self.vy *= self.damping

        # SPEED LIMIT (no sudden jumps)
        self.vx = np.clip(self.vx, -self.max_speed, self.max_speed)
        self.vy = np.clip(self.vy, -self.max_speed, self.max_speed)

        self.cx += self.vx
        self.cy += self.vy

        return int(self.cx), int(self.cy)


def detect_scenes(path: str) -> List[Tuple[int, int]]:
    """
    Detect scene boundaries in video.
    Returns list of (start_frame, end_frame) tuples.
    """
    if not SCENEDETECT_AVAILABLE:
        return []
        
    try:
        video = open_video(path)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=15))
        sm.detect_scenes(video)
        scenes = sm.get_scene_list()
        return [(s[0].get_frames(), s[1].get_frames()) for s in scenes]
    except Exception:
        return []


def smart_crop_video(
    input_path: str,
    output_path: str,
    target_width: int = OUT_W,
    target_height: int = OUT_H,
    verbose: bool = True
) -> str:
    """
    Smart crop a video to vertical 9:16 with AI subject tracking.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        target_width: Output width (default 1080)
        target_height: Output height (default 1920)
        verbose: Show progress
        
    Returns:
        Path to output video
    """
    # Detect scene boundaries for tracker reset
    scenes = detect_scenes(input_path)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate crop dimensions
    ratio = target_width / target_height
    crop_h = h
    crop_w = int(h * ratio)
    
    # Handle case where video is narrower than target aspect
    if crop_w > w:
        crop_w = w
        crop_h = int(w / ratio)

    tracker = CinematicTracker(w, h)

    frames = []
    idx = 0
    scene_idx = 0

    if verbose:
        print(f"   Smart cropping: {w}x{h} → {target_width}x{target_height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Scene cut → hard reset tracker
        if scene_idx < len(scenes) and idx >= scenes[scene_idx][1]:
            tracker.reset()
            scene_idx += 1

        cx, cy = tracker.update(frame)

        # Calculate crop bounds
        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2

        x1 = max(0, min(w - crop_w, x1))
        y1 = max(0, min(h - crop_h, y1))

        crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        out = cv2.resize(crop, (target_width, target_height))

        frames.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        idx += 1
        if verbose and idx % 100 == 0:
            sys.stdout.write(f"\r   Processing frame {idx}...")
            sys.stdout.flush()

    cap.release()
    
    if verbose:
        print(f"\r   Processed {idx} frames")

    # Create output video
    clip = ImageSequenceClip(frames, fps=fps)
    src = VideoFileClip(input_path)
    if src.audio:
        clip = clip.set_audio(src.audio)

    clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        threads=4,
        logger=None
    )
    
    src.close()
    clip.close()
    
    return output_path


def center_crop_video(
    input_path: str,
    output_path: str,
    target_width: int = OUT_W,
    target_height: int = OUT_H
) -> str:
    """
    Simple center crop without AI tracking.
    Faster fallback when AI tracking not needed.
    """
    clip = VideoFileClip(input_path)
    w, h = clip.size
    
    target_ratio = target_width / target_height
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        # Too wide: crop width
        new_w = int(h * target_ratio)
        crop_clip = clip.crop(x1=(w - new_w)//2, y1=0, width=new_w, height=h)
    else:
        # Too tall: crop height
        new_h = int(w / target_ratio)
        crop_clip = clip.crop(x1=0, y1=(h - new_h)//2, width=w, height=new_h)
    
    final = crop_clip.resize((target_width, target_height))
    
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=clip.fps,
        preset="medium",
        logger=None
    )
    
    clip.close()
    final.close()
    
    return output_path
