# import cv2
# import numpy as np
# import math
# import argparse
# import sys
# from moviepy.editor import VideoFileClip, ImageSequenceClip
# from ultralytics import YOLO
# from scenedetect import open_video, SceneManager
# from scenedetect.detectors import ContentDetector
# from mediapipe.python.solutions import face_detection as mp_face

# # ===============================
# # OUTPUT SETTINGS (VERTICAL)
# # ===============================
# OUT_W = 1080
# OUT_H = 1920

# # ===============================
# # TRACKING / DETECTION TUNING
# # ===============================
# FLOW_WIN = (21, 21)
# FLOW_MAX_LEVEL = 3
# MIN_FACE_AREA = 0.001
# MIN_PERSON_AREA = 0.045
# CONF_THRESH = 0.35

# STATE_SEARCHING = 0
# STATE_LOCKED = 1

# # ===============================
# # TRACKER
# # ===============================
# class CinematicTracker:
#     def __init__(self, w, h):
#         self.w = w
#         self.h = h
#         self.center = (w // 2, h // 2)
#         self.state = STATE_SEARCHING
#         self.lock_score = 0
#         self.prev_gray = None

#         self.face = mp_face.FaceDetection(1, 0.5)
#         self.yolo = YOLO("yolov8n.pt")

#     def reset(self):
#         self.center = (self.w // 2, self.h // 2)
#         self.state = STATE_SEARCHING
#         self.lock_score = 0
#         self.prev_gray = None

#     def get_detections(self, frame):
#         dets = []
#         total = self.w * self.h

#         # Face detection
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = self.face.process(rgb)
#         if res.detections:
#             for d in res.detections:
#                 r = d.location_data.relative_bounding_box
#                 area = r.width * r.height
#                 if area < MIN_FACE_AREA:
#                     continue
#                 cx = int((r.xmin + r.width / 2) * self.w)
#                 cy = int((r.ymin + r.height / 2) * self.h)
#                 dets.append({
#                     "type": "face",
#                     "center": (cx, cy),
#                     "score": 1000 + area * 5000
#                 })

#         # Person detection
#         y = self.yolo(frame, verbose=False)[0]
#         for b in y.boxes:
#             if self.yolo.names[int(b.cls[0])] != "person":
#                 continue
#             conf = float(b.conf[0])
#             if conf < CONF_THRESH:
#                 continue
#             x1, y1, x2, y2 = map(int, b.xyxy[0])
#             area = ((x2 - x1) * (y2 - y1)) / total
#             if area < MIN_PERSON_AREA:
#                 continue
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             dets.append({
#                 "type": "person",
#                 "center": (cx, cy),
#                 "score": conf * 100 + area * 500
#             })

#         return dets

#     def update(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         dets = self.get_detections(frame)

#         if self.state == STATE_SEARCHING:
#             if dets:
#                 best = max(dets, key=lambda d: d["score"])
#                 self.center = best["center"]
#                 self.state = STATE_LOCKED
#                 self.lock_score = 100

#         else:
#             best = None
#             min_dist = 1e9
#             for d in dets:
#                 dy = abs(d["center"][1] - self.center[1])
#                 if dy < min_dist:
#                     min_dist = dy
#                     best = d

#             if best:
#                 cy = best["center"][1]
#                 if abs(cy - self.center[1]) > 2:
#                     self.center = (
#                         self.center[0],
#                         int(self.center[1] * 0.95 + cy * 0.05)
#                     )
#                 self.lock_score = 100
#             else:
#                 self.lock_score -= 2
#                 if self.lock_score <= 0:
#                     self.reset()

#         self.prev_gray = gray
#         return self.center

# # ===============================
# # SCENE DETECTION
# # ===============================
# def detect_scenes(path):
#     video = open_video(path)
#     sm = SceneManager()
#     sm.add_detector(ContentDetector(threshold=15))
#     sm.detect_scenes(video)
#     scenes = sm.get_scene_list()
#     return [(s[0].get_frames(), s[1].get_frames()) for s in scenes]

# # ===============================
# # MAIN PIPELINE
# # ===============================
# def smart_crop_video(input_path, output_path):
#     scenes = []
#     try:
#         scenes = detect_scenes(input_path)
#     except:
#         pass

#     cap = cv2.VideoCapture(input_path)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     ratio = OUT_W / OUT_H

#     # Vertical crop
#     crop_h = h
#     crop_w = int(h * ratio)

#     tracker = CinematicTracker(w, h)
#     frames = []
#     idx = 0
#     scene_idx = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if scene_idx < len(scenes) and idx >= scenes[scene_idx][1]:
#             tracker.reset()
#             scene_idx += 1

#         cx, cy = tracker.update(frame)

#         # ===== VERTICAL LOGIC (NO RULE OF THIRDS) =====
#         x1 = (w - crop_w) // 2      # LOCK X
#         y1 = cy - crop_h // 2       # TRACK Y

#         x1 = max(0, min(w - crop_w, x1))
#         y1 = max(0, min(h - crop_h, y1))

#         crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
#         out = cv2.resize(crop, (OUT_W, OUT_H))
#         frames.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

#         idx += 1
#         if idx % 50 == 0:
#             sys.stdout.write(f"\r⏳ Frame {idx}")
#             sys.stdout.flush()

#     cap.release()

#     clip = ImageSequenceClip(frames, fps=fps)
#     src = VideoFileClip(input_path)
#     if src.audio:
#         clip = clip.set_audio(src.audio)

#     clip.write_videofile(
#         output_path,
#         codec="libx264",
#         audio_codec="aac",
#         ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
#         threads=4,
#         logger=None
#     )

# # ===============================
# # ENTRY
# # ===============================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--i", required=True)
#     parser.add_argument("--o", required=True)
#     args = parser.parse_args()
#     smart_crop_video(args.i, args.o)


import cv2
import numpy as np
import argparse
import sys
from moviepy.editor import VideoFileClip, ImageSequenceClip
from ultralytics import YOLO
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from mediapipe.python.solutions import face_detection as mp_face

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

# ===============================
# TRACKER (FIXED)
# ===============================
class CinematicTracker:
    def __init__(self, w, h):
        self.w = w
        self.h = h

        # Camera center (float for precision)
        self.cx = float(w // 2)
        self.cy = float(h // 2)

        # Velocity (for damping)
        self.vx = 0.0
        self.vy = 0.0

        # Physics tuning (KEY)
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

        self.face = mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.6
        )
        self.yolo = YOLO("yolov8n.pt")

    def reset(self):
        self.cx = float(self.w // 2)
        self.cy = float(self.h // 2)
        self.vx = self.vy = 0.0
        self.target_x = self.cx
        self.target_y = self.cy
        self.hold_frames = 0

    # ---------------------------
    # FACE FIRST
    # ---------------------------
    def detect_face(self, frame):
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

    # ---------------------------
    # PERSON FALLBACK
    # ---------------------------
    def detect_person(self, frame):
        y = self.yolo(frame, verbose=False)[0]
        best = None
        best_score = 0

        for b in y.boxes:
            if self.yolo.names[int(b.cls[0])] != "person":
                continue
            conf = float(b.conf[0])
            if conf < 0.35:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            score = conf * area
            if score > best_score:
                best_score = score
                best = ((x1 + x2) // 2, (y1 + y2) // 2)

        return best

    # ---------------------------
    # UPDATE (JITTER-FREE)
    # ---------------------------
    def update(self, frame):
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




# ===============================
# SCENE DETECTION
# ===============================
def detect_scenes(path):
    video = open_video(path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=15))
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()
    return [(s[0].get_frames(), s[1].get_frames()) for s in scenes]


# ===============================
# MAIN PIPELINE
# ===============================
def smart_crop_video(input_path, output_path):
    try:
        scenes = detect_scenes(input_path)
    except:
        scenes = []

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ratio = OUT_W / OUT_H
    crop_h = h
    crop_w = int(h * ratio)

    tracker = CinematicTracker(w, h)

    frames = []
    idx = 0
    scene_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Scene cut → hard reset
        if scene_idx < len(scenes) and idx >= scenes[scene_idx][1]:
            tracker.reset()
            scene_idx += 1

        cx, cy = tracker.update(frame)

        # ---------------------------
        # TRUE CINEMATIC VERTICAL CROP
        # ---------------------------
        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2

        x1 = max(0, min(w - crop_w, x1))
        y1 = max(0, min(h - crop_h, y1))

        crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        out = cv2.resize(crop, (OUT_W, OUT_H))

        frames.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        idx += 1
        if idx % 50 == 0:
            sys.stdout.write(f"\r⏳ Frame {idx}")
            sys.stdout.flush()

    cap.release()

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


# ===============================
# ENTRY
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", required=True)
    parser.add_argument("--o", required=True)
    args = parser.parse_args()
    smart_crop_video(args.i, args.o)


# import cv2
# import numpy as np
# import argparse
# from moviepy.editor import VideoFileClip, ImageSequenceClip
# from ultralytics import YOLO
# from scenedetect import open_video, SceneManager
# from scenedetect.detectors import ContentDetector
# from mediapipe.python.solutions import face_detection as mp_face

# # ===============================
# # OUTPUT SETTINGS
# # ===============================
# OUT_W = 1080
# OUT_H = 1920

# # ===============================
# # SCENE DETECTION
# # ===============================
# def detect_scenes(path):
#     video = open_video(path)
#     sm = SceneManager()
#     sm.add_detector(ContentDetector(threshold=15))
#     sm.detect_scenes(video)
#     scenes = sm.get_scene_list()
#     return [(s[0].get_frames(), s[1].get_frames()) for s in scenes]

# # ===============================
# # OPTICAL FLOW TRACKER (LOCKED)
# # ===============================
# class LockedFlowTracker:
#     def __init__(self, w, h):
#         self.w = w
#         self.h = h
#         self.center = (w // 2, h // 2)
#         self.box = None
#         self.prev_gray = None
#         self.locked = False

#     def reset(self):
#         self.center = (self.w // 2, self.h // 2)
#         self.box = None
#         self.prev_gray = None
#         self.locked = False

#     def initialize(self, frame, box):
#         self.box = box
#         x1, y1, x2, y2 = box
#         self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
#         self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         self.locked = True

#     def update(self, frame):
#         if not self.locked or self.box is None:
#             return self.center

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         x1, y1, x2, y2 = self.box

#         pts = []
#         step = 20
#         for y in range(y1, y2, step):
#             for x in range(x1, x2, step):
#                 pts.append([x, y])

#         if len(pts) < 10:
#             return self.center

#         p0 = np.float32(pts).reshape(-1, 1, 2)
#         p1, st, _ = cv2.calcOpticalFlowPyrLK(
#             self.prev_gray, gray, p0, None,
#             winSize=(21, 21),
#             maxLevel=3,
#             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
#         )

#         good_new = p1[st == 1]
#         good_old = p0[st == 1]

#         if len(good_new) < 8:
#             return self.center

#         shift = np.median(good_new - good_old, axis=0)
#         dx, dy = int(shift[0]), int(shift[1])

#         # clamp motion (NO JITTER)
#         dx = np.clip(dx, -10, 10)
#         dy = np.clip(dy, -10, 10)

#         self.center = (self.center[0] + dx, self.center[1] + dy)
#         self.box = (
#             self.box[0] + dx,
#             self.box[1] + dy,
#             self.box[2] + dx,
#             self.box[3] + dy
#         )

#         self.prev_gray = gray
#         return self.center

# # ===============================
# # SUBJECT DETECTOR (ONCE PER SCENE)
# # ===============================
# class SubjectDetector:
#     def __init__(self):
#         self.face = mp_face.FaceDetection(1, 0.5)
#         self.yolo = YOLO("yolov8n.pt")

#     def detect(self, frame, w, h):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Face priority
#         res = self.face.process(rgb)
#         if res.detections:
#             d = res.detections[0]
#             r = d.location_data.relative_bounding_box
#             x1 = int(r.xmin * w)
#             y1 = int(r.ymin * h)
#             x2 = int((r.xmin + r.width) * w)
#             y2 = int((r.xmin + r.height) * h)
#             return (x1, y1, x2, y2)

#         # Person fallback
#         y = self.yolo(frame, verbose=False)[0]
#         for b in y.boxes:
#             if self.yolo.names[int(b.cls[0])] == "person":
#                 return tuple(map(int, b.xyxy[0]))

#         return None

# # ===============================
# # MAIN PIPELINE
# # ===============================
# def smart_crop_video(input_path, output_path):
#     scenes = detect_scenes(input_path)

#     cap = cv2.VideoCapture(input_path)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     crop_h = h
#     crop_w = int(h * (OUT_W / OUT_H))

#     detector = SubjectDetector()
#     tracker = LockedFlowTracker(w, h)

#     frames = []
#     frame_idx = 0
#     scene_idx = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # ---- NEW SCENE: HARD RESET + RELOCK ----
#         if scene_idx < len(scenes) and frame_idx == scenes[scene_idx][0]:
#             tracker.reset()
#             box = detector.detect(frame, w, h)
#             if box:
#                 tracker.initialize(frame, box)
#             scene_idx += 1

#         cx, cy = tracker.update(frame)

#         x1 = cx - crop_w // 2
#         y1 = cy - crop_h // 2

#         x1 = max(0, min(w - crop_w, x1))
#         y1 = max(0, min(h - crop_h, y1))

#         crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
#         out = cv2.resize(crop, (OUT_W, OUT_H))
#         frames.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

#         frame_idx += 1

#     cap.release()

#     clip = ImageSequenceClip(frames, fps=fps)
#     src = VideoFileClip(input_path)
#     if src.audio:
#         clip = clip.set_audio(src.audio)

#     clip.write_videofile(
#         output_path,
#         codec="libx264",
#         audio_codec="aac",
#         ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
#         logger=None
#     )

# # ===============================
# # ENTRY
# # ===============================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--i", required=True)
#     parser.add_argument("--o", required=True)
#     args = parser.parse_args()
#     smart_crop_video(args.i, args.o)
