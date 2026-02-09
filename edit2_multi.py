# fast cuts MULTI-SOURCE MONTAGE (Best of Folder)


import librosa
import numpy as np
import cv2
import os
import argparse
import librosa.util.utils # Explicit import for patching

# ---------------------------------------------------------
# PATCH: Fix for librosa 0.10.0 crash with numba/numpy
# ---------------------------------------------------------
def _abs2_patch(x):
    return x.real**2 + x.imag**2
librosa.util.utils.abs2 = _abs2_patch
# ---------------------------------------------------------

from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from smtcro import smart_crop_video
from clean import perform_cleanup

# -------------------------------------------------
# STEP 1: STRONG BEAT DETECTION
# -------------------------------------------------
def detect_cut_points(audio_path):
    y, sr = librosa.load(audio_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        trim=True
    )
    return librosa.frames_to_time(beats, sr=sr)

# -------------------------------------------------
# STEP 2: SCENE DETECTION (ROBUST)
# -------------------------------------------------
def cleanup_proxy(proxy_path):
    if os.path.exists(proxy_path):
        try:
            os.remove(proxy_path)
        except:
            pass

def detect_scenes(video_path):
    """
    Detects scenes using a temp low-res proxy to be safe/fast.
    """
    import subprocess
    import os
    
    # Unique proxy name to avoid collisions
    import uuid
    uid = str(uuid.uuid4())[:8]
    proxy_path = f"proxy_{uid}.mp4"

    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", "scale=-2:480", # even smaller for speed
        "-r", "10",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        proxy_path
    ]

    try:
        subprocess.run(command, check=True, stderr=subprocess.DEVNULL)
        
        video = open_video(proxy_path, backend="opencv", framerate=10)
        
        # Initial Attempt
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
        scene_manager.detect_scenes(video)
        scenes = scene_manager.get_scene_list()

        if not scenes:
            # Fallback 1: Lower threshold
            print(f"   ⚠️ Standard detection failed. Retrying with sensitive threshold...")
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=10.0))
            scene_manager.detect_scenes(video)
            scenes = scene_manager.get_scene_list()

        if not scenes:
            # Fallback 2: Fixed Grid (Force Output)
            print(f"   ⚠️ Detection failed. Applying FIXED GRID segmentation as fallback.")
            duration = video.duration.get_seconds()
            if duration is None: duration = 10.0 # safety
            
            fixed_scenes = []
            curr = 0
            while curr < duration:
                fixed_scenes.append((curr, min(curr + 3.0, duration)))
                curr += 3.0
            return fixed_scenes
        
        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
        
    except Exception as e:
        print(f"⚠️ Scene detection failed for {os.path.basename(video_path)}: {e}")
        return []
        
    finally:
        cleanup_proxy(proxy_path)


# -------------------------------------------------
# STEP 3: MOTION SCORING
# -------------------------------------------------
def get_motion_score(video_path, start, end):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    prev = None
    total_diff = 0
    frames = 0
    max_frames = 100 # sample first 100 frames max per scene for speed

    while cap.get(cv2.CAP_PROP_POS_MSEC) < end * 1000 and frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (320, 180)) # downscale for speed
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        if prev is not None:
            total_diff += np.mean(cv2.absdiff(prev, gray))
        
        prev = gray
        frames += 1

    cap.release()
    return total_diff / frames if frames > 0 else 0


# -------------------------------------------------
# STEP 4: GLOBAL MONTAGE ASSEMBLY
# -------------------------------------------------
def create_montage(input_dir, music_path, output_path):
    # 1. Gather Inputs
    valid_exts = ('.mp4', '.mov', '.avi', '.mkv')
    video_files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith(valid_exts)
    ]

    if not video_files:
        print("❌ No videos found!")
        return

    print(f"🎞️ Found {len(video_files)} source videos.")

    # 2. Analyze Music
    print(f"🎵 Analyzing Beat: {music_path}...")
    beats = detect_cut_points(music_path)
    audio = AudioFileClip(music_path)
    music_duration = audio.duration
    
    # 3. Build GLOBAL Scene Pool
    print("\n🌍 BUILDING SCENE POOL (Scanning all videos)...")
    
    # Format: (score, video_path, start, end)
    all_scenes = [] 
    
    for i, vid in enumerate(video_files):
        print(f"   [{i+1}/{len(video_files)}] Scanning: {os.path.basename(vid)}...")
        scenes = detect_scenes(vid)
        
        for s, e in scenes:
            if e - s < 0.5: continue # ignore tiny scenes
            score = get_motion_score(vid, s, e)
            all_scenes.append({
                "score": score,
                "path": vid,
                "start": s,
                "end": e,
                "duration": e - s
            })

    # Sort globally by energy
    all_scenes.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n✨ Total Pool Size: {len(all_scenes)} scenes available.")

    # 4. Assemble Montage
    print("\n✂️ Assembling Montage from Best Scenes...")
    
    clips = []
    current_time = 0
    scene_idx = 0
    
    # Create caches to minimize reloading
    video_clips_cache = {} # path -> VideoFileClip

    def get_clip_segment(scene_data, required_duration):
        path = scene_data["path"]
        if path not in video_clips_cache:
            video_clips_cache[path] = VideoFileClip(path).without_audio()
        
        src_clip = video_clips_cache[path]
        s = scene_data["start"]
        e = scene_data["end"]
        
        # Take subclip
        seg_len = min(required_duration, e - s)
        return src_clip.subclip(s, s + seg_len)

    # --- BEAT LOOP ---
    beat_idx = 0
    while current_time < music_duration:
        # Determine cut length
        if beat_idx < len(beats) - 1:
            cut_duration = beats[beat_idx + 1] - beats[beat_idx]
        else:
            cut_duration = 2.0 # default fallback

        # Pick Scene
        if scene_idx < len(all_scenes):
            # Normal: Pick next best unique scene
            scene = all_scenes[scene_idx]
            scene_idx += 1
        else:
            # RECYCLE: Start over from best
            print("🔄 Pool Exhausted! Recycling best scenes...")
            scene_idx = 0
            scene = all_scenes[scene_idx]
            scene_idx += 1

        # Use Scene
        try:
            clip = get_clip_segment(scene, cut_duration)
            clips.append(clip)
            current_time += clip.duration
            beat_idx += 1
        except Exception as e:
            print(f"⚠️ Failed to load clip, skipping: {e}")

    # 5. Concatenate
    print("\nRendering Draft...")
    final = concatenate_videoclips(clips, method="compose")
    final = final.subclip(0, music_duration)
    final = final.set_audio(audio)

    # 6. Export HQ Intermediate
    temp_hq = "temp_montage_hq.mp4"
    
    # Detect Source FPS
    source_fps = 30
    if video_files:
        try:
            with VideoFileClip(video_files[0]) as probe:
                source_fps = probe.fps
        except: pass

    final.write_videofile(
        temp_hq,
        codec="libx264",
        audio_codec="aac",
        fps=source_fps,
        preset="slow",
        ffmpeg_params=["-crf", "18"],
        threads=4
    )
    
    # Close cache
    for c in video_clips_cache.values():
        c.close()

    # 7. Orientation Check & Smart Crop
    print("\n📐 Checking Orientation...")
    with VideoFileClip(temp_hq) as probe:
        w, h = probe.size
        
    if h >= w:
        print("✅ Already Vertical. Renaming...")
        if os.path.exists(output_path): os.remove(output_path)
        os.rename(temp_hq, output_path)
    else:
        print("📱 Horizontal detected. Applying Smart Crop...")
        smart_crop_video(temp_hq, output_path)
        os.remove(temp_hq)

    print(f"\n🎉 MONTAGE COMPLETE: {output_path}")
    
    perform_cleanup()


if __name__ == "__main__":
    INPUT_DIR = "/Users/uday/Downloads/edmmusic/test"
    MUSIC_PATH = "output.mp3"
    OUTPUT_FILE = "output_videos/FINAL_MONTAGE.mp4" 

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    try:
        create_montage(INPUT_DIR, MUSIC_PATH, OUTPUT_FILE)
    except KeyboardInterrupt:
        print("\n⛔ Cancelled.")
