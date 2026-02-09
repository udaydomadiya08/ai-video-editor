
import os
import sys
import time
import argparse
import subprocess
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip
from smart_crop_standalone import smart_crop_video
from edit2_multi import detect_cut_points, detect_scenes, get_motion_score
from clean import perform_cleanup
# Import processors from existing scripts
# We need to dynamically import or replicate the logic since they are scripts, not modules.
# However, for robustness, I will implement the core processing logic within this script 
# by importing the functional parts and wrapping them.

from edit_interactive2_normal import process_job_normal
from edit_interactive2_hd import process_job_hd

# =================================================
# ⚙️ CONFIG UTILS
# =================================================
VIDEO_DIR = "/Users/uday/Downloads/edmmusic/input_vid"
MUSIC_DIR = "/Users/uday/Downloads/edmmusic/music"
OUTPUT_DIR = "output_pipeline"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_files(directory, extensions):
    return sorted([
        f for f in os.listdir(directory) 
        if f.lower().endswith(extensions)
    ])

def print_menu(title, items, base_dir=None, is_video=False):
    print(f"\n📁 {title}:")
    for i, item in enumerate(items):
        meta = ""
        if base_dir and is_video:
            try:
                full_path = os.path.join(base_dir, item)
                cap = cv2.VideoCapture(full_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0:
                        dur = frames / fps
                        meta = f" ({int(dur//60)}:{int(dur%60):02d})"
                cap.release()
            except: pass
        print(f"  [{i+1}] {item}{meta}")

def get_single_index(prompt, max_idx):
    while True:
        try:
            print(prompt, end="")
            choice = input().strip()
            idx = int(choice) - 1
            if 0 <= idx < max_idx: return idx
        except ValueError:
            pass
        print("❌ Invalid number.")

def get_bool(prompt):
    while True:
        print(prompt, end="")
        choice = input().strip()
        if choice in ['1', 'y', 'Y']: return True
        if choice in ['2', 'n', 'N', '0']: return False
        print("❌ Enter 1 (Yes) or 2 (No).")

def get_float(prompt, default=0.0):
    while True:
        print(prompt, end="")
        choice = input().strip()
        if not choice: return default
        try:
            return float(choice)
        except ValueError:
            print("❌ Invalid number.")

# =================================================
# 🛠️ PROCESSING HELPERS
# =================================================

def slice_music(music_path, start, duration, temp_path):
    """Slices the global music for a specific edit segment."""
    try:
        with AudioFileClip(music_path) as audio:
            # Loop audio if needed
            while audio.duration < start + duration:
                audio = concatenate_videoclips([audio, audio]) # Audio concatenation via clips logic
                # Actually for AudioFileClip, we might need manual looping or composition
                # Better way:
                # We will handle looping in final mix. 
                # For generating the *edit*, we need a concrete file.
                pass
            
            # Safe slice
            end = min(start + duration, audio.duration)
            chunk = audio.subclip(start, end)
            chunk.write_audiofile(temp_path, logger=None)
            return True
    except Exception as e:
        print(f"❌ Music slice error: {e}")
        return False

def center_crop_to_vertical(input_path, output_path):
    """Crops a video to 1080x1920 (9:16) by zooming/centering."""
    try:
        clip = VideoFileClip(input_path)
        w, h = clip.size
        
        target_ratio = 1080 / 1920
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # Too wide: scale by height, crop width
            # Actually, to FILL, we need to match height to 1920 (if scaling up) 
            # or just crop width.
            # MoviePy resize logic:
            # We want to enable 1080x1920 output.
            # Simplest: crop centered to 9:16 aspect, then resize to 1080x1920.
            
            new_w = int(h * target_ratio)
            crop_clip = clip.crop(x1=(w - new_w)//2, y1=0, width=new_w, height=h)
            final = crop_clip.resize((1080, 1920))
            
        else:
            # Too tall (rare for horizontal inputs): scale by width, crop height
            new_h = int(w / target_ratio)
            crop_clip = clip.crop(x1=0, y1=(h - new_h)//2, width=w, height=new_h)
            final = crop_clip.resize((1080, 1920))

        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30, # Enforce standard FPS
            preset="medium",
            logger=None
        )
        clip.close()
        final.close()
        return True
    except Exception as e:
        print(f"❌ Center crop failed: {e}")
        return False

def apply_deepfilternet(video_path, output_dir, file_prefix):
    """
    Extracts audio, runs DeepFilterNet to isolate vocals/remove noise.
    Returns path to enhanced audio file.
    """
    try:
        # 1. Extract raw audio
        temp_wav = os.path.join(output_dir, f"{file_prefix}_raw.wav")
        # -map 0:a:0 picks first audio track. -ac 1 mono is usually better for DFNet but stereo works too.
        # DFNet supports 48k.
        cmd_extract = [
            "ffmpeg", "-y", "-i", video_path, 
            "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", 
            temp_wav
        ]
        subprocess.run(cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. Run DeepFilterNet
        # deepFilter {input} -o {out_dir}
        cmd_df = ["deepFilter", temp_wav, "-o", output_dir] 
        subprocess.run(cmd_df, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        enhanced_wav = os.path.join(output_dir, f"{file_prefix}_raw_DeepFilterNet3.wav")
        
        if os.path.exists(enhanced_wav):
            if os.path.exists(temp_wav): os.remove(temp_wav)
            return enhanced_wav
        else:
            print(f"❌ DeepFilterNet output not found: {enhanced_wav}")
            return None
            
    except Exception as e:
        print(f"❌ DeepFilterNet failed: {e}")
        return None

def process_segment(job_id, seg_idx, segment, global_music_path, offset_start):
    """
    Processes a single segment configuration into a video clip file.
    Returns: (output_path, actual_duration)
    """
    
    video_path = segment['video_path']
    seg_type = segment['type'] # 'normal' or 'edit'
    mode_hd = segment['hd']    # True/False
    smart_crop = segment['smart_crop'] # True/False
    enhance_audio = segment.get('enhance_audio', False) # New Flag
    zoom_type = segment.get('zoom_type', None) # Zoom in/out/None
    
    # Timing Logic
    use_full = segment.get('use_full', False)
    seg_start = segment.get('start', 0.0)
    seg_end = segment.get('end', 0.0)
    
    temp_out = os.path.join(OUTPUT_DIR, f"temp_{job_id}_{seg_idx}.mp4")
    temp_trimmed = os.path.join(OUTPUT_DIR, f"temp_source_trimmed_{job_id}_{seg_idx}.mp4")

    print(f"\n   ⚙️ Processing Segment {seg_idx+1}: {seg_type.upper()} | HD={mode_hd} | Crop={smart_crop}")

    # 1. PREPARE TRIMMED SOURCE
    # We must trim the video first to define the source material & duration
    try:
        source_clip = VideoFileClip(video_path)
        
        if use_full:
            start_t = 0
            end_t = source_clip.duration
        else:
            start_t = seg_start
            end_t = seg_end if seg_end > 0 else source_clip.duration
            
        # Bounds check
        end_t = min(end_t, source_clip.duration)
        if start_t >= end_t:
            print("   ⚠️ Invalid start/end times. Using full clip.")
            start_t = 0
            end_t = source_clip.duration
            
        duration = end_t - start_t
        print(f"      ⏱️ Time Range: {start_t}s - {end_t}s (Duration: {duration:.2f}s)")
        
        trimmed_clip = source_clip.subclip(start_t, end_t)
        
        # Write intermediate trimmed source (needed for scripts)
        # HD/Normal settings apply to FINAL encode, but for intermediate we want quality.
        # We can use 'fast' preset here to save time, quality lost is minimal if bitrate high.
        trimmed_clip.write_videofile(
            temp_trimmed,
            codec="libx264",
            preset="fast", # Speed up intermediate
            logger=None
        )
        source_clip.close()
        trimmed_clip.close()
        
    except Exception as e:
        print(f"❌ Source Trimming Failed: {e}")
        return None, 0

    # ---------------------------
    # A. EDIT GENERATION
    # ---------------------------
    if seg_type == 'edit':
        # 1. Prepare Music Slice
        temp_music = os.path.join(OUTPUT_DIR, f"temp_music_{job_id}_{seg_idx}.wav")
        
        # Use target duration if specified, else use full source length
        target_duration = segment.get('edit_duration', 0.0)
        if target_duration <= 0.0:
            target_duration = duration # Fallback to using all of the trimmed source length
            
        print(f"      🎵 Generating Edit Duration based on music slice: {target_duration:.2f}s")
        
        full_audio = AudioFileClip(global_music_path)
        audio_len = full_audio.duration
        loop_start = offset_start % audio_len
        
        chunk = full_audio.subclip(loop_start, min(loop_start+target_duration, audio_len))
        chunk.write_audiofile(temp_music, logger=None)
        full_audio.close()
        
        # 2. Run Edit Processor
        # IMPORTANT: 'process_job' functions usually handle smart crop internally or return raw.
        # If smart_crop is FALSE, they return original aspect. 
        # We must intercept this and FORCE CENTER CROP if user wants vertical fit.
        
        # Let's generate to a pre-final path from the processor
        temp_proc_out = os.path.join(OUTPUT_DIR, f"temp_proc_{job_id}_{seg_idx}.mp4")
        
        try:
            if mode_hd:
                 # HD Processor
                process_job_hd(
                     f"{job_id}_{seg_idx}", 
                     [temp_trimmed], # Use trimmed source
                     temp_music, 
                     smart_crop, # Handles smart crop internally if True
                     temp_proc_out,
                     zoom_type=zoom_type
                )
            else:
                 # Normal Processor
                process_job_normal(
                     f"{job_id}_{seg_idx}", 
                     [temp_trimmed], # Use trimmed source
                     temp_music, 
                     smart_crop, 
                     temp_proc_out,
                     zoom_type=zoom_type
                )
        except Exception as e:
            print(f"❌ Edit Generation Failed: {e}")
            if os.path.exists(temp_trimmed): os.remove(temp_trimmed)
            return None, 0

        if os.path.exists(temp_trimmed): os.remove(temp_trimmed)
        
        if not os.path.exists(temp_proc_out):
            return None, 0
            
        # 3. Post-Process:
        if smart_crop:
            # If processor didn't handle it, or we trusted it. 
            # Note: process_job_hd/normal take the smart_crop flag.
            # Assuming they work. If not, we could force it here.
            # But earlier user said smart crop had bars. 
            # The standalone script is better.
            # Ideally we run standalone script on the result if the processor failed?
            # But process_job calls 'smart_crop_video'.
            # We patched the import! So process_job might still be using OLD import inside ITSELF?
            # Ah! process_job_hd is in another file!
            # We imported it. It has its OWN imports.
            # We cannot easily patch imports inside `edit_interactive2_hd.py` without editing that file.
            # So, for EDITS, we should probably run `smart_crop_standalone` explicitly if we want the NEW logic.
            
            # Let's Move the temp_proc_out to temp_out.
            # AND if smart_crop is True, we might re-run standalone to be SAFE?
            # Or trust the patching? We can't patch the other file easily from here.
            
            # SAFE METHOD: Always run standalone smart crop here if requested.
            # But wait, process_job ALREADY cropped it?
            # If process_job used old `smtcro`, it might have bars.
            # So: Run Smart Crop Standalone on the output of process_job!
            
            print("   📱 Applying Standalone Smart Crop to Edit...")
            # Check if already vertical to save time
            try:
                chk = VideoFileClip(temp_proc_out)
                cw, ch = chk.size
                chk.close()
                if ch > cw:
                    print("      ⚡ Video is already vertical. Using fast center crop.")
                    center_crop_to_vertical(temp_proc_out, temp_out)
                else:
                    smart_crop_video(temp_proc_out, temp_out)
            except:
                smart_crop_video(temp_proc_out, temp_out)
            
            if os.path.exists(temp_proc_out): os.remove(temp_proc_out)
        else:
            # OPTION 2: "Original Video Only" (Horizontal/Passthrough)
            if os.path.exists(temp_out): os.remove(temp_out)
            os.rename(temp_proc_out, temp_out)

        return temp_out, target_duration

    # ---------------------------
    # B. RAW PREPARATION
    # ---------------------------
    else:
        # We already have temp_trimmed. 
        try:
            # Optional: AUDIO ENHANCEMENT (DEEPFILTERNET)
            if enhance_audio:
                print("      🎤 Running DeepFilterNet to isolate vocals...")
                enhanced_wav = apply_deepfilternet(temp_trimmed, OUTPUT_DIR, f"{job_id}_{seg_idx}")
                
                if enhanced_wav:
                    print("      ✅ Audio enhanced successfully. Replacing track.")
                    
                    temp_enhanced_video = os.path.join(OUTPUT_DIR, f"temp_enhanced_{job_id}_{seg_idx}.mp4")
                    
                    clip_v = VideoFileClip(temp_trimmed)
                    clip_a = AudioFileClip(enhanced_wav)
                    
                    # USER REQUEST: Boost cleaned audio by 5x
                    print("      🔊 Boosting cleaned audio by 5x...")
                    clip_a = clip_a.volumex(5.0)
                    
                    clip_v = clip_v.set_audio(clip_a)
                    clip_v.write_videofile(temp_enhanced_video, codec="libx264", audio_codec="aac", logger=None)
                    
                    clip_v.close()
                    clip_a.close()
                    
                    # Swap
                    if os.path.exists(temp_trimmed): os.remove(temp_trimmed)
                    os.rename(temp_enhanced_video, temp_trimmed)
                    
                    # Cleanup wav
                    if os.path.exists(enhanced_wav): os.remove(enhanced_wav)

            if smart_crop:
                print("   📱 Applying Smart Crop...")
                # Check if already vertical to save time
                try:
                    chk = VideoFileClip(temp_trimmed)
                    cw, ch = chk.size
                    chk.close()
                    if ch > cw:
                        print("      ⚡ Video is already vertical. Using fast center crop.")
                        center_crop_to_vertical(temp_trimmed, temp_out)
                    else:
                         smart_crop_video(temp_trimmed, temp_out)
                except:
                    smart_crop_video(temp_trimmed, temp_out)
            else:
                # OPTION 2: "Original Video Only" (Horizontal/Passthrough)
                print("   ✅ Keeping Original Aspect Ratio (Horizontal)...")
                # Just rename/move
                if os.path.exists(temp_out): os.remove(temp_out)
                os.rename(temp_trimmed, temp_out)
                
            if os.path.exists(temp_trimmed): os.remove(temp_trimmed)
            
            return temp_out, duration

        except Exception as e:
            print(f"❌ Raw Processing Failed: {e}")
            if os.path.exists(temp_trimmed): os.remove(temp_trimmed)
            return None, 0

# =================================================
# 📝 CONFIG PARSER (TEXT FILE)
# =================================================
def parse_config_file(config_path, video_dir, music_dir, video_files, music_files):
    """Parses a simple text-based config file for jobs."""
    jobs = []
    current_job = None
    current_segments = []
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"): continue
        
        if line.upper() == "[JOB]":
            # Save previous
            if current_job:
                current_job['segments'] = current_segments
                jobs.append(current_job)
            
            # Start new
            current_job = {
                "id": len(jobs) + 1,
                "music": "", 
                "music_start": 0.0,
                "output": os.path.join(OUTPUT_DIR, f"FINAL_PIPELINE_{len(jobs)+1}.mp4")
            }
            current_segments = []
            
        elif line.upper() == "[SEGMENT]":
            current_segments.append({
                "video_path": "",
                "type": "normal",
                "hd": False,
                "smart_crop": False,
                "enhance_audio": False,
                "use_full": True,
                "start": 0.0,
                "end": 0.0,
                "edit_duration": 0.0,
                "zoom_type": None
            })
            
        elif "=" in line:
            key, raw_val = [x.strip() for x in line.split("=", 1)]
            val = raw_val.split("#")[0].strip()
            
            if not current_job: continue # Should be in a job
            
            # JOB LEVEL
            if key.lower() == "music":
                # Check if it's an index (digits)
                if val.isdigit():
                    idx = int(val) - 1
                    if 0 <= idx < len(music_files):
                        current_job['music'] = os.path.join(music_dir, music_files[idx])
                    else:
                        print(f"⚠️ Invalid Music Index: {val}")
                # Try full path or find in music_dir
                elif os.path.exists(val):
                    current_job['music'] = val
                else:
                    guess = os.path.join(music_dir, val)
                    if os.path.exists(guess):
                        current_job['music'] = guess
                    else:
                        print(f"⚠️ Music file not found: {val}")
                        
            elif key.lower() == "start_music_at":
                current_job['music_start'] = float(val)
                
            # SEGMENT LEVEL
            elif current_segments:
                seg = current_segments[-1]
                
                if key.lower() == "video":
                     # Check if it's an index (digits)
                    if val.isdigit():
                        idx = int(val) - 1
                        if 0 <= idx < len(video_files):
                            seg['video_path'] = os.path.join(video_dir, video_files[idx])
                        else:
                             print(f"⚠️ Invalid Video Index: {val}")
                    # Try full path or find in video_dir
                    elif os.path.exists(val):
                        seg['video_path'] = val
                    else:
                        guess = os.path.join(video_dir, val)
                        if os.path.exists(guess):
                            seg['video_path'] = guess
                        else:
                            print(f"⚠️ Video file not found: {val}")
                            
                elif key.lower() == "type":
                    seg['type'] = val.lower()
                    if seg['type'] == 'normal': seg['enhance_audio'] = True # Auto-enable for normal
                    
                elif key.lower() == "hd":
                    seg['hd'] = (val.lower() == "true")
                    
                elif key.lower() == "smart_crop":
                    seg['smart_crop'] = (val.lower() == "true")
                    
                elif key.lower() == "use_full_duration":
                    seg['use_full'] = (val.lower() == "true")
                    
                elif key.lower() == "start_time":
                    seg['start'] = float(val)
                    
                elif key.lower() == "end_time":
                    seg['end'] = float(val)
                    
                elif key.lower() == "edit_duration":
                    seg['edit_duration'] = float(val)
                    
                elif key.lower() == "zoom_effect":
                    v = val.lower()
                    if v in ['in', 'out']:
                        seg['zoom_type'] = v
                    else:
                        seg['zoom_type'] = None

    # Save last
    if current_job:
        current_job['segments'] = current_segments
        jobs.append(current_job)
        
    return jobs

# =================================================
# 📝 SCAFFOLD GENERATOR
# =================================================
def generate_config_scaffold(num_jobs, video_files, music_files, output_path="input_config.txt"):
    """Generates a config file with N job templates and file lists."""
    
    with open(output_path, 'w') as f:
        f.write("# PIPELINE INPUT CONFIGURATION\n")
        f.write("# ============================\n#\n")
        f.write("# 📂 AVAILABLE VIDEO FILES (Use Index or Filename):\n")
        for i, vf in enumerate(video_files):
            f.write(f"#  [{i+1}] {vf}\n")
            
        f.write("#\n# 🎵 AVAILABLE MUSIC FILES:\n")
        for i, mf in enumerate(music_files):
            f.write(f"#  [{i+1}] {mf}\n")
            
        f.write("#\n# ============================\n\n")
        
        for i in range(num_jobs):
            job_num = i + 1
            f.write(f"[JOB]\n")
            f.write(f"Music = 1   # Use Index [1] or Filename\n")
            f.write(f"Start_Music_At = 0.0\n\n")
            
            f.write(f"    # --- SEGMENT 1 ---\n")
            f.write(f"    [SEGMENT]\n")
            f.write(f"    Video = 1   # Use Index [1] or Filename\n")
            f.write(f"    Type = normal           # options: normal, edit\n")
            f.write(f"    HD = true               # options: true, false\n")
            f.write(f"    Smart_Crop = true       # options: true, false\n")
            f.write(f"    Use_Full_Duration = true\n")
            f.write(f"    Start_Time = 0.0\n")
            f.write(f"    End_Time = 0.0\n")
            f.write(f"    # Edit Specifics\n")
            f.write(f"    Edit_Duration = 0.0\n")
            f.write(f"    Zoom_Effect = none      # options: in, out, none\n\n")
            
    print(f"\n✅ Generated config template with {num_jobs} jobs: {output_path}")
    print(f"👉 Please open '{output_path}', edit the file names, and run this script again.")

# =================================================
# 🖥️ MAIN LOGIC
# =================================================
def main():
    clear_screen()
    print("🎬 INTERACTIVE PIPELINE BUILDER 🎬")
    print("==================================")
    
    # SYSTEM CLEANUP AT START
    perform_cleanup()
    
    video_dir = VIDEO_DIR # Use local logic
    music_dir = MUSIC_DIR
    
    if not os.path.exists(video_dir) or not os.path.exists(music_dir):
        print(f"❌ Error: Directories '{video_dir}' or '{music_dir}' not found.")
        # Try to recover if just input_vid is missing
        if not os.path.exists(video_dir):
             print(f"   (Will check current directory for videos...)")
        else:
             return

    video_files = []
    if os.path.exists(video_dir):
        video_files = get_files(video_dir, ('.mp4', '.mov', '.mkv', '.avi'))
    
    # Fallback to current directory if input_vid is empty or missing
    if not video_files:
        print(f"⚠️  No videos found in '{video_dir}'. Checking current directory...")
        current_videos = get_files(".", ('.mp4', '.mov', '.mkv', '.avi'))
        # Filter out output files to avoid confusion
        current_videos = [f for f in current_videos if not f.startswith("output_") and not f.startswith("temp_")]
        
        if current_videos:
            video_files = current_videos
            video_dir = "."
            print(f"   ✅ Found {len(video_files)} videos in current directory.")
            
    music_files = get_files(music_dir, ('.mp3', '.wav'))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CHECK FOR CONFIG FILE
    config_file = "input_config.txt"
    should_run_interactive = False
    
    if os.path.exists(config_file):
        print(f"\nFound existing config: '{config_file}'")
        print("   [1] RUN this config")
        print("   [2] REGENERATE template (Overwrite)")
        print("   [3] Ignore and use Interactive Mode")
        print("   Choice: ", end="")
        c = input().strip()
        
        if c == '1':
            print(f"\n📄 Parsing '{config_file}'...")
            jobs = parse_config_file(config_file, video_dir, music_dir, video_files, music_files)
            if not jobs:
                print("❌ No valid jobs found. Switching to Interactive.")
                should_run_interactive = True
        elif c == '2':
            # Regenerate logic falls through to below
            jobs = []
        else:
            should_run_interactive = True
            jobs = []
    else:
        # Config doesn't exist, offer to generate
        print(f"\nConfig '{config_file}' not found.")
        print("   [1] Generate Config Template")
        print("   [2] Use Interactive Mode")
        print("   Choice: ", end="")
        c = input().strip()
        if c == '2':
            should_run_interactive = True
            jobs = []
        else:
            jobs = [] # Proceed to generate

    # GENERATE SCAFFOLD IF NEEDED (No jobs, not interactive)
    if not jobs and not should_run_interactive:
        # Show files to help user decide N? Or just ask N.
        # User requested: "list all index of input vides and aslo music in temrnial"
        print_menu("Available Videos", video_files, video_dir, True)
        print_menu("Available Music", music_files)
        
        try:
            print("\nHow many blank JOB templates to generate? : ", end="")
            n_jobs = int(input())
        except: n_jobs = 1
        
        if n_jobs > 0:
            generate_config_scaffold(n_jobs, video_files, music_files, config_file)
            return # Exit so user can edit
        else:
            print("No jobs requested. Exiting.")
            return

    # INTERACTIVE MODE
    if should_run_interactive or (not jobs and should_run_interactive):
        # 1. Batch Setup
        try:
            print("\nHow many FINAL videos to generate? : ", end="")
            num_jobs = int(input())
        except: num_jobs = 1

        jobs = []

        # 2. Job Collection Loop
        for j in range(num_jobs):
            job_id = j + 1
            print(f"\n📝 CONFIGURING FINAL VIDEO {job_id}/{num_jobs}")
            print("-----------------------------------")
            
            # Select Global Music
            print_menu("Available Music", music_files)
            print(f"Select Global Music for Video {job_id}: ", end="")
            m_idx = get_single_index("", len(music_files))
            global_music = os.path.join(music_dir, music_files[m_idx])
            print(f"   Start Music at (seconds) [default: 0]: ", end="")
            music_start_time = get_float("", 0.0)
            
            segments = []
            seg_counter = 0
            
            while True:
                seg_counter += 1
                print(f"\n   ➕ Add Segment {seg_counter} (or type 'done' to finish video)")
                
                # Video Selection
                print_menu("Input Videos", video_files, video_dir, True)
                print(f"   Select Video Index (or 'done'): ", end="")
                choice = input().strip()
                
                if choice.lower() == 'done':
                    break
                    
                try:
                    v_idx = int(choice) - 1
                    if not (0 <= v_idx < len(video_files)):
                        print("   ❌ Invalid index.")
                        continue
                except:
                    print("   ❌ Invalid input.")
                    continue
                    
                selected_video = os.path.join(video_dir, video_files[v_idx])
                
                # Parameters
                print("   --- Settings ---")
                # Type
                is_edit = False
                enhance_audio = False
                
                t_choice = ""
                print("   [1] Raw Video  [2] Make Edit : ", end="")
                t_choice = input().strip()
                if t_choice == '2': 
                    is_edit = True
                else:
                    # Normal video -> Always isolate vocals now
                    enhance_audio = True
                    print("   🎤 Auto-enabling Vocal Isolation (DeepFilterNet)")
                
                # HD / Normal
                is_hd = False
                print("   [1] HD (High Quality)  [2] Normal (Fast) : ", end="")
                q_choice = input().strip()
                if q_choice == '1': is_hd = True
                
                # Crop
                is_smart = False
                print("   [1] Smart Crop (9:16)  [2] Horizontal (Original) : ", end="")
                c_choice = input().strip()
                if c_choice == '1': is_smart = True
                
                # Time Range
                print("   --- Time Range ---")
                print("   Use full video duration? (y/n/full) [default: y]: ", end="")
                tr_choice = input().strip().lower()
                
                use_full = False
                start_t = 0.0
                end_t = 0.0
                
                if not tr_choice or tr_choice in ['y', 'yes', 'full', '1']:
                    use_full = True
                else:
                    print("   Start Time (sec): ", end="")
                    start_t = get_float("", 0.0)
                    print("   End Time (sec) (0 for end): ", end="")
                    end_t = get_float("", 0.0)
                
                # Additional Duration for Edit
                edit_dur = 0.0
                if is_edit:
                    print("   --- Edit Length ---")
                    print("   Target Duration of this Edit (seconds) [0=Use full source length]: ", end="")
                    edit_dur = get_float("", 0.0)
                
                # Zoom Option (Edits Only)
                zoom_type = None
                if is_edit:
                    print("   Apply Zoom Effect to clips? (y/n) [default: n]: ", end="")
                    want_zoom = input().strip().lower()
                    if want_zoom in ['y', 'yes', '1']:
                        print("   Select Type: [1] Zoom IN  [2] Zoom OUT : ", end="")
                        z_choice = input().strip()
                        if z_choice == '1': zoom_type = 'in'
                        elif z_choice == '2': zoom_type = 'out'
                
                segments.append({
                    "video_path": selected_video,
                    "type": 'edit' if is_edit else 'normal',
                    "hd": is_hd,
                    "smart_crop": is_smart,
                    "enhance_audio": enhance_audio,
                    "use_full": use_full,
                    "start": start_t,
                    "end": end_t,
                    "edit_duration": edit_dur,
                    "zoom_type": zoom_type
                })
                
            jobs.append({
                "id": job_id,
                "music": global_music,
                "music_start": music_start_time,
                "segments": segments,
                "output": os.path.join(OUTPUT_DIR, f"FINAL_PIPELINE_{job_id}.mp4")
            })

            
        jobs.append({
            "id": job_id,
            "music": global_music,
            "music_start": music_start_time,
            "segments": segments,
            "output": os.path.join(OUTPUT_DIR, f"FINAL_PIPELINE_{job_id}.mp4")
        })

    # 3. Batch Execution
    print(f"\n🚀 STARTING BATCH PROCESSING ({len(jobs)} Jobs)...")
    
    for job in jobs:
        print(f"\n▶️  PROCESSING FINAL VIDEO {job['id']}: {os.path.basename(job['output'])}")
        
        final_clips = []
        music_offset = job.get('music_start', 0.0)
        
        # Process Segments
        for i, seg in enumerate(job['segments']):
            out_path, actual_dur = process_segment(
                job['id'], i, seg, job['music'], music_offset
            )
            
            if out_path:
                try:
                    clip = VideoFileClip(out_path)
                    
                    # Store metadata for mixing
                    clip.is_edit = (seg['type'] == 'edit')
                    final_clips.append(clip)
                    
                    music_offset += actual_dur
                except Exception as e:
                    print(f"   ❌ Failed to load processed segment: {e}")
        
        if not final_clips:
            print("   ❌ No clips generated. Skipping.")
            continue
            
        # Assembly & Audio Mixing
        print("   🎚️  Assembling & Mixing Audio...")
        
        # 1. Concatenate Video
        final_video = concatenate_videoclips(final_clips, method="compose")
        
        # 2. Global Music Background (Smart Chunking with Wrap)
        music_audio = AudioFileClip(job['music'])
        music_chunks = []
        curr_time = 0 
        # Using separate tracker for timeline relative to music start
        music_timeline_cursor = job.get('music_start', 0.0)
        
        from moviepy.editor import concatenate_audioclips
        
        for clip in final_clips:
            dur = clip.duration
            
            # Handle wrapping manually for safer subclip
            # Calculate range in music file [start, start+dur]
            # Modulo length
            m_len = music_audio.duration
            m_start = music_timeline_cursor % m_len
            m_end = m_start + dur
            
            if m_end <= m_len:
                # Simple Case: No wrap
                raw_chunk = music_audio.subclip(m_start, m_end)
            else:
                # Wrap Case: Part 1 (End of file) + Part 2 (Start of file needed)
                part1 = music_audio.subclip(m_start, m_len)
                rem_dur = dur - part1.duration
                
                # If remainder > music duration (very short song, long clip), loop multiples
                loops_needed = int(rem_dur // m_len)
                rem_final = rem_dur % m_len
                
                parts = [part1] + [music_audio]*loops_needed + [music_audio.subclip(0, rem_final)]
                raw_chunk = concatenate_audioclips(parts)

            # Ducking Logic
            if clip.is_edit:
                # Edit: Music 100%, Clip Muted
                chunk = raw_chunk.volumex(1.0)
                clip.audio = None # Mute source
            else:
                # Normal: Music 8% (0.08), Clip 100%
                chunk = raw_chunk.volumex(0.08)
                
            music_chunks.append(chunk)
            music_timeline_cursor += dur
            
        final_music_track = concatenate_audioclips(music_chunks)
        
        # Helper to mix source + music
        source_audios = []
        t = 0
        for clip in final_clips:
            if clip.audio:
                source_audios.append(clip.audio.set_start(t))
            t += clip.duration
            
        # Combine Global Music + Source Audios
        final_audio = CompositeAudioClip([final_music_track] + source_audios)
        final_video = final_video.set_audio(final_audio)
        
        # Export
        print(f"   💾 Exporting to {job['output']}...")
        final_video.write_videofile(
            job['output'],
            codec="libx264",
            audio_codec="aac",
            fps=final_video.fps,
            preset="medium",
            ffmpeg_params=["-crf", "18"],
            logger=None
        )
        
        # Cleanup clips
        music_audio.close()
        for c in final_clips: c.close()
        
        # Cleanup Temp Files for this Job
        print(f"   🧹 Cleaning up temp files for Job {job['id']}...")
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith(f"temp_") and f"_{job['id']}_" in f:
                try:
                    os.remove(os.path.join(OUTPUT_DIR, f))
                except: pass
            # Also clean up deepfilter raw files if any leftover
            if f.startswith(f"{job['id']}_") and "_raw" in f:
                try:
                    os.remove(os.path.join(OUTPUT_DIR, f))
                except: pass
        
        # Aggressive System Cleanup after each job
        perform_cleanup()
        
    print("\n✅ BATCH PROCESSING COMPLETE!")

if __name__ == "__main__":
    main()


