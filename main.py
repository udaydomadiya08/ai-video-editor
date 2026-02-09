#!/usr/bin/env python3
"""
AI Video Editing Style Learning System - Main CLI

Usage:
    # Train on reference videos
    python main.py train --refs ./reference_videos --output ./models

    # Generate edited video from single input + music
    python main.py generate --input video.mp4 --music track.mp3 --output output.mp4 --models ./models

    # Or use multiple clips
    python main.py generate --clips ./raw_clips --music track.mp3 --output output.mp4 --models ./models
"""
import argparse
import os
import datetime
from pathlib import Path

from config import get_config
from utils.video_io import get_all_videos
from modules.style_learner import StyleLearner
from modules.music_analyzer import MusicAnalyzer
from modules.auto_editor import AutoEditor


def train(args):
    """Train on reference videos to learn editing style."""
    import torch
    
    # Detect and configure device based on new flags
    device = None
    num_gpus = 1
    gpu_ids = None
    
    if args.cpu:
        # Force CPU
        device = 'cpu'
        print("💻 Using CPU as requested")
    elif args.mps:
        # Force MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            device = 'mps'
            print("🚀 Using Apple Silicon MPS GPU")
        else:
            print("⚠️ MPS requested but not available, falling back to CPU")
            device = 'cpu'
    elif args.gpu is not None:
        # GPU mode with specified number
        num_gpus = args.gpu
        
        if not torch.cuda.is_available():
            print("⚠️ CUDA GPU requested but not available, falling back to CPU")
            device = 'cpu'
            num_gpus = 1
        else:
            device = 'cuda'
            available_gpus = torch.cuda.device_count()
            
            if num_gpus == -1:
                # Use all available GPUs
                num_gpus = available_gpus
                gpu_ids = list(range(available_gpus))
                print(f"🚀 Multi-GPU mode: Using ALL {num_gpus} GPUs")
                for i in range(num_gpus):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            elif num_gpus > 1:
                if num_gpus > available_gpus:
                    print(f"⚠️ Requested {num_gpus} GPUs but only {available_gpus} available")
                    num_gpus = available_gpus
                    print(f"   Using {num_gpus} GPU(s)")
                
                gpu_ids = list(range(num_gpus))
                print(f"🚀 Multi-GPU mode: Using {num_gpus} GPUs")
                for i in range(num_gpus):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                # Single GPU
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 Using CUDA GPU: {gpu_name}")
    else:
        # Auto-detect (default)
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"🚀 CUDA GPU detected - using GPU acceleration")
            print(f"   GPU: {gpu_name} ({gpu_count} device(s) available)")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print("🚀 Apple Silicon GPU detected - using MPS acceleration")
        else:
            device = 'cpu'
            print("💻 Using CPU (no GPU detected)")
    
    # Set device in config
    from config import set_config
    config = set_config(device=device)
    
    # Store GPU configuration for models to use
    config.num_gpus = num_gpus
    config.gpu_ids = gpu_ids
    
    # Find reference videos
    ref_videos = get_all_videos(args.refs)
    if not ref_videos:
        print(f"No videos found in {args.refs}")
        return
    
    print(f"Found {len(ref_videos)} reference videos")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Step 1: Learn style patterns
    print("\n" + "="*50)
    print("STEP 1: Learning Style Patterns")
    print("="*50)
    
    style_learner = StyleLearner(config)
    style_params = style_learner.learn_from_videos(ref_videos, verbose=True)
    
    style_path = os.path.join(args.output, 'style_params.pkl')
    style_params.save(style_path)
    print(f"Saved style parameters to {style_path}")
    
    # Step 2: Extract Cuts for Neural Training
    print("\n" + "="*50)
    print("STEP 2: Extracting Cuts for Neural Training")
    print("="*50)
    
    cut_points = {}
    print("  Detecting cuts in reference videos...")
    for video_path in ref_videos:
        try:
            # Use style_learner's cut detection (based heavily on content changes)
            video_cuts = style_learner.detect_cuts(video_path, verbose=False)
            cut_points[video_path] = video_cuts
            print(f"  {Path(video_path).name}: {len(video_cuts)} cuts")
        except Exception as e:
            print(f"  Error detecting cuts in {Path(video_path).name}: {e}")
            cut_points[video_path] = []
    
    # Step 4: Learn Neural Transitions (frame-level effects)
    print("\n" + "="*50)
    print("STEP 4: Learning Neural Transitions")
    print("="*50)
    
    from modules.transition_frame_extractor import TransitionFrameExtractor
    from modules.neural_transition_vae import NeuralTransitionVAE
    
    # Extract transition frame sequences around cuts
    print("  Extracting transition frame sequences...")
    extractor = TransitionFrameExtractor(
        window_frames=10,       # 10 frames before/after = 20 frames total (High Context)
        output_size=(128, 128) # Normalized size for training
    )
    
    all_sequences = []
    for video_path in ref_videos:
        if video_path in cut_points and cut_points[video_path]:
            try:
                sequences = extractor.extract_transitions(
                    video_path, cut_points[video_path], verbose=False
                )
                all_sequences.extend(sequences)
            except Exception as e:
                print(f"  Skipping cuts in {Path(video_path).name}: {e}")
                
        # ALSO extract random sequences from whole video ("God Mode" Learning)
        try:
            # Extract 100 random sequences per video to effectively "Watch Everything"
            # 100 * 20 frames = 2000 frames per video (covers typical 1-2 min video densely)
            random_seqs = extractor.extract_random_sequences(
                video_path, num_sequences=100, verbose=False
            )
            all_sequences.extend(random_seqs)
        except Exception as e:
             print(f"  Skipping random seqs in {Path(video_path).name}: {e}")
    
    print(f"  Extracted {len(all_sequences)} transition sequences")
    
    # MEMORY OPTIMIZATION FOR M1 MAC (8GB RAM)
    # 2800 sequences * 20 frames * 128x128 * 3 * 4 bytes ~ 11GB RAM -> CRASH
    # Limit to 500 sequences (~2GB RAM) for stability
    MAX_SEQUENCES = 500
    if len(all_sequences) > MAX_SEQUENCES:
        import random
        print(f"  ⚠️ Limiting to {MAX_SEQUENCES} sequences to prevent OOM crash (Random Sample)")
        random.shuffle(all_sequences)
        all_sequences = all_sequences[:MAX_SEQUENCES]
    
    if len(all_sequences) >= 10:  # Need minimum data
        # Convert to training format
        training_data = extractor.sequences_to_training_data(
            all_sequences, use_differences=True
        )
        
        print(f"  Training data shape: {training_data.shape}")
        
        # Train Neural Transition VAE
        neural_vae = NeuralTransitionVAE(config)
        
        # Resume from existing model if requested
        neural_vae_path = os.path.join(args.output, 'neural_transition_vae.pt')
        if args.resume and os.path.exists(neural_vae_path):
            print(f"  📂 Loading existing model from {neural_vae_path}")
            neural_vae.load(neural_vae_path)
            print(f"  ✓ Resuming training (incremental learning)")
        
        neural_vae.train(
            training_data,
            epochs=20,  # Reduced for M1 CPU speed
            batch_size=4, # Reduced for M1
            verbose=True
        )
        
        # Save
        neural_vae.save(neural_vae_path)
        print(f"Saved neural transition VAE to {neural_vae_path}")
    else:
        print("  ⚠️ Not enough transitions for neural learning (need at least 10)")
        print("  Skipping neural transition training")

    # Step 5: Learn Master Visual Style (Deep VAE + AdaIN)
    print("\n" + "="*50)
    print("STEP 5: Learning Master Visual Style (AdaIN)")
    print("="*50)
    print("  Learning global video texture, color grading, and motion patterns...")
    
    from modules.deep_style_learner import DeepStyleLearner
    from modules.neural_style_vae import NeuralStyleVAE, VideoStyleModel
    
    # 1. Extract random patches
    # Optimized for HIGH QUALITY (larger spatial/temporal window)
    style_learner = DeepStyleLearner(
        patch_size=128,        # 128 is safe max for M1 8GB (192 is OOM risk)
        frames_per_patch=16,   
        patches_per_video=50   # 50 is sufficient sampling without blowing RAM
    )
    
    all_style_patches = []
    for video_path in ref_videos:
        print(f"  Sampling {Path(video_path).name}...")
        try:
            patches = style_learner.extract_patches(video_path, verbose=False)
            all_style_patches.extend(patches)
        except Exception as e:
            print(f"  Skipping {Path(video_path).name}: {e}")

    # MEMORY OPTIMIZATION FOR M1 (Step 5)
    # Limit style patches to avoid OOM
    MAX_PATCHES = 500
    if len(all_style_patches) > MAX_PATCHES:
        import random
        print(f"  ⚠️ Limiting to {MAX_PATCHES} style patches for memory safety")
        random.shuffle(all_style_patches)
        all_style_patches = all_style_patches[:MAX_PATCHES]
            
    print(f"  Collected {len(all_style_patches)} style patches")
    
    if len(all_style_patches) >= 20:
        # Prepare data
        style_data = style_learner.prepare_training_data(all_style_patches)
        print(f"  Style training data: {style_data.shape}")
        
        # Train Deep Style VAE
        style_model = VideoStyleModel(config)
        
        # Resume from existing model if requested
        deep_style_path = os.path.join(args.output, 'deep_style_vae.pt')
        if args.resume and os.path.exists(deep_style_path):
            print(f"  📂 Loading existing model from {deep_style_path}")
            style_model.load(deep_style_path)
            print(f"  ✓ Resuming training (incremental learning)")
        
        style_model.train(
            style_data,
            epochs=10,      # Reduced for M1 CPU
            batch_size=2,   # Reduced for M1 8GB
            lr=1e-4,
            verbose=True
        )
        
        # Save
        style_model.save(deep_style_path)
        print(f"Saved Deep Style VAE to {deep_style_path}")
        
        # Also save representative style patches as "Learned Style Reference"
        # We save the aggregated style features for later transfer
        # (For now just save the model, extraction happens at inference if needed)
        
    else:
        print("  ⚠️ Not enough data for deep style learning")

    # Step 6: Learn Universal Visual Effects (Flow + Intensity)
    print("\n" + "="*50)
    print("STEP 6: Learning Universal Effects (Motion + Intensity)")
    print("="*50)
    print("  Learning dynamic visual patterns (shake, strobe, fade, zoom) from effect fields...")
    
    from modules.effect_field_learner import EffectFieldLearner
    from modules.neural_effect_vae import UniversalEffectModel
    
    # Optimize for M1/8GB RAM ("Efficient God Mode")
    # 48x27 is still very high detail for motion fields (matches 16:9 aspect)
    effect_learner = EffectFieldLearner(grid_width=48, grid_height=27)
    all_windows = []
    
    for video_path in ref_videos:
        print(f"  Extracting effects from {Path(video_path).name}...")
        try:
            # Extract full grid sequence
            grid_seq = effect_learner.extract_field(video_path, verbose=False)
            if grid_seq:
                # Prepare windows for VAE (T=64)
                windows = effect_learner.prepare_training_data([grid_seq], window_size=64)
                if len(windows) > 0:
                    all_windows.append(windows)
        except Exception as e:
            print(f"  Failed: {e}")
            
    if all_windows:
        import numpy as np
        training_data = np.concatenate(all_windows, axis=0) # (N, 3, 64, 9, 16)
        print(f"  Collected {len(training_data)} effect sequences")
        
        effect_model = UniversalEffectModel(config)
        
        # Resume from existing model if requested
        effect_path = os.path.join(args.output, 'neural_effect_vae.pt')
        if args.resume and os.path.exists(effect_path):
            print(f"  📂 Loading existing model from {effect_path}")
            effect_model.load(effect_path)
            print(f"  ✓ Resuming training (incremental learning)")
        
        effect_model.train(
            training_data,
            epochs=50,
            batch_size=4 # Reduced for M1
        )
        
        effect_model.save(effect_path)
        print(f"Saved Universal Effect VAE to {effect_path}")
        
        # Save reference effect grids for style transfer
        import pickle
        ref_path = os.path.join(args.output, 'effect_refs.pkl')
        # Save a few representative windows
        indices = np.random.choice(len(training_data), min(50, len(training_data)), replace=False)
        refs = training_data[indices]
        with open(ref_path, 'wb') as f:
            pickle.dump(refs, f)
        print(f"Saved {len(refs)} reference effect patterns to {ref_path}")
    else:
        print("  ⚠️ Not enough data for effect learning")

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"\nModels saved to: {args.output}")
    print(f"  - style_params.pkl")
    neural_trans_path = os.path.join(args.output, 'neural_transition_vae.pt')
    if os.path.exists(neural_trans_path):
        print(f"  - neural_transition_vae.pt (Neural Transitions)")
        
    deep_style_path = os.path.join(args.output, 'deep_style_vae.pt')
    if os.path.exists(deep_style_path):
        print(f"  - deep_style_vae.pt (Master Visual Style)")
        
    effect_path = os.path.join(args.output, 'neural_effect_vae.pt')
    if os.path.exists(effect_path):
        print(f"  - neural_effect_vae.pt (Universal Effect VAE)")



def validate_models(args):
    """Show what the models learned from training."""
    import pickle
    
    models_dir = args.models
    
    print("\n" + "="*60)
    print("MODEL VALIDATION - What Was Learned")
    print("="*60)
    
    # 1. Style Parameters
    style_path = os.path.join(models_dir, 'style_params.pkl')
    if os.path.exists(style_path):
        with open(style_path, 'rb') as f:
            style_params = pickle.load(f)
        
        print("\n📊 STYLE PARAMETERS (from reference videos):")
        print("-"*50)
        
        if hasattr(style_params, 'mean_shot_length'):
            print(f"  Average shot length: {style_params.mean_shot_length:.2f}s")
        if hasattr(style_params, 'std_shot_length'):
            print(f"  Shot length variation: ±{style_params.std_shot_length:.2f}s")
        if hasattr(style_params, 'shot_length_distribution') and style_params.shot_length_distribution:
            lengths = style_params.shot_length_distribution
            try:
                numeric_lengths = [float(l) for l in lengths if isinstance(l, (int, float))]
                if numeric_lengths:
                    print(f"  Shot length range: {min(numeric_lengths):.2f}s - {max(numeric_lengths):.2f}s")
                    print(f"  Number of shots analyzed: {len(numeric_lengths)}")
            except:
                print(f"  Number of shots analyzed: {len(lengths)}")
        if hasattr(style_params, 'mean_motion'):
            print(f"  Average motion intensity: {style_params.mean_motion:.3f}")
        if hasattr(style_params, 'beat_alignment_ratio'):
            print(f"  Beat alignment ratio: {style_params.beat_alignment_ratio:.1%}")
        if hasattr(style_params, 'cut_histogram_threshold'):
            print(f"  Learned cut threshold: {style_params.cut_histogram_threshold:.3f}")
        if hasattr(style_params, 'reference_fps'):
            print(f"  Reference FPS: {style_params.reference_fps:.1f}")
    else:
        print("\n⚠️ Style parameters not found!")
    
    # 2. Transition Data
    trans_path = os.path.join(models_dir, 'transition_data.pkl')
    if os.path.exists(trans_path):
        with open(trans_path, 'rb') as f:
            trans_data = pickle.load(f)
        
        print("\n🎬 TRANSITION DATA (motion patterns):")
        print("-"*50)
        
        if hasattr(trans_data, 'n_samples'):
            print(f"  Transitions analyzed: {trans_data.n_samples}")
        if hasattr(trans_data, 'n_clusters'):
            print(f"  Transition types discovered: {trans_data.n_clusters}")
        if hasattr(trans_data, 'cluster_sizes') and trans_data.cluster_sizes:
            print(f"  Cluster distribution: {trans_data.cluster_sizes}")
        if hasattr(trans_data, 'transition_vectors') and len(trans_data.transition_vectors) > 0:
            print(f"  Vector dimensionality: {trans_data.transition_vectors.shape}")
            print(f"  Motion range: {trans_data.transition_vectors.min():.3f} to {trans_data.transition_vectors.max():.3f}")
    else:
        print("\n⚠️ Transition data not found!")
    
    # 3. Autoencoder
    auto_path = os.path.join(models_dir, 'transition_autoencoder.pt')
    if os.path.exists(auto_path):
        import torch
        checkpoint = torch.load(auto_path, map_location='cpu', weights_only=False)
        
        print("\n🧠 AUTOENCODER MODEL:")
        print("-"*50)
        print(f"  Input dimension: {checkpoint.get('input_dim', 'unknown')}")
        print(f"  Latent dimension: {checkpoint.get('latent_dim', 'unknown')}")
        print(f"  Hidden dimension: {checkpoint.get('hidden_dim', 'unknown')}")
        
        if 'train_losses' in checkpoint and checkpoint['train_losses']:
            losses = checkpoint['train_losses']
            print(f"  Training epochs completed: {len(losses)}")
            print(f"  Initial loss: {losses[0]:.4f}")
            print(f"  Final loss: {losses[-1]:.4f}")
            improvement = (losses[0] - losses[-1]) / losses[0] * 100
            print(f"  Improvement: {improvement:.1f}%")
            
            if improvement < 20:
                print("\n  ⚠️ LOW IMPROVEMENT - Model may need:")
                print("    - More training data (more reference videos)")
                print("    - More varied reference videos")
                print("    - Longer training (increase epochs)")
    else:
        print("\n⚠️ Autoencoder not found!")
    
    # 4. Music Mapper
    mapper_path = os.path.join(models_dir, 'music_mapper.pkl')
    if os.path.exists(mapper_path):
        with open(mapper_path, 'rb') as f:
            mapper = pickle.load(f)
        
        print("\n🎵 MUSIC-TRANSITION MAPPER:")
        print("-"*50)
        
        if hasattr(mapper, 'n_samples'):
            print(f"  Training samples: {mapper.n_samples}")
        if hasattr(mapper, 'music_features') and len(mapper.music_features) > 0:
            print(f"  Music features used: {mapper.music_features.shape[1] if len(mapper.music_features.shape) > 1 else 'N/A'}")
        if hasattr(mapper, 'trained') and mapper.trained:
            print(f"  Model trained: ✓")
        else:
            print(f"  Model trained: ✗ (may need more data)")
    else:
        print("\n⚠️ Music mapper not found!")
    
    print("\n" + "="*60)
    print("To retrain: python main.py train --refs ./data --output ./models")
    print("="*60)

def generate(args):
    """Generate edited video using learned models."""
    config = get_config()
    
    # Validate inputs
    if not args.input and not args.clips:
        print("Error: Must provide either --input (single video) or --clips (directory)")
        return
    
    if not os.path.exists(args.music):
        print(f"Error: Music file not found: {args.music}")
        return
    
    # Create auto editor
    editor = AutoEditor(config)
    
    # Load models
    print("Loading learned models...")
    editor.load_models(
        style_path=os.path.join(args.models, 'style_params.pkl'),
        transition_data_path=os.path.join(args.models, 'transition_data.pkl'),
        autoencoder_path=os.path.join(args.models, 'transition_autoencoder.pt'),
        mapper_path=os.path.join(args.models, 'music_mapper.pkl')
    )
    
    # Generate video
    if args.input:
        # Single video mode
        print(f"\nGenerating from single video: {args.input}")
        editor.generate_from_video(
            input_video=args.input,
            music_path=args.music,
            output_path=args.output,
            verbose=True
        )
    else:
        # Multiple clips mode
        print(f"\nGenerating from clips directory: {args.clips}")
        editor.generate(
            raw_clips_dir=args.clips,
            music_path=args.music,
            output_path=args.output,
            verbose=True
        )
    
    print(f"\n✓ Output saved: {args.output}")


def interactive():
    """Interactive mode - select video and music from folders."""
    config = get_config()
    
    input_vids_dir = '/Users/uday/Downloads/edmmusic/input_vid'
    music_dir = '/Users/uday/Downloads/edmmusic/music'
    models_dir = './models'
    
    # Check directories exist
    if not os.path.exists(input_vids_dir):
        os.makedirs(input_vids_dir, exist_ok=True)
        print(f"Created {input_vids_dir}/ - Please add your input videos there.")
        return
    
    if not os.path.exists(music_dir):
        os.makedirs(music_dir, exist_ok=True)
        print(f"Created {music_dir}/ - Please add your music files there.")
        return
    
    # List available videos
    videos = get_all_videos(input_vids_dir)
    if not videos:
        print(f"No videos found in {input_vids_dir}/")
        print("Add your videos there and run again.")
        return
    
    # List available music
    music_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}
    music_files = []
    for f in Path(music_dir).iterdir():
        if f.suffix.lower() in music_extensions:
            music_files.append(str(f))
    
    if not music_files:
        print(f"No music files found in {music_dir}/")
        print("Add your music files (.mp3, .wav, etc.) there and run again.")
        return
    
    # Display videos
    print("\n" + "="*50)
    print("AVAILABLE VIDEOS (input_vids/)")
    print("="*50)
    for i, video in enumerate(videos, 1):
        name = Path(video).name
        print(f"  [{i}] {name}")
    
    # Display music
    print("\n" + "="*50)
    print("AVAILABLE MUSIC (music/)")
    print("="*50)
    for i, music in enumerate(music_files, 1):
        name = Path(music).name
        print(f"  [{i}] {name}")
    
    # Get user selection
    print("\n" + "-"*50)
    try:
        vid_choice = int(input("Select video number: ")) - 1
        if vid_choice < 0 or vid_choice >= len(videos):
            print("Invalid video selection")
            return
        
        music_choice = int(input("Select music number: ")) - 1
        if music_choice < 0 or music_choice >= len(music_files):
            print("Invalid music selection")
            return
    except ValueError:
        print("Please enter a number")
        return
    except KeyboardInterrupt:
        print("\nCancelled")
        return
    
    selected_video = videos[vid_choice]
    selected_music = music_files[music_choice]
    
    # Ask for music duration
    print("\n" + "-"*50)
    print("Music timing (press Enter for full track):")
    try:
        start_input = input("  Start time (seconds, e.g. 30): ").strip()
        end_input = input("  End time (seconds, e.g. 60): ").strip()
        
        music_start = float(start_input) if start_input else None
        music_end = float(end_input) if end_input else None
    except ValueError:
        print("Invalid time format, using full track")
        music_start = None
        music_end = None
    except KeyboardInterrupt:
        print("\nCancelled")
        return
    
    # Ask for smart crop
    print("\n" + "-"*50)
    print("Output format:")
    try:
        crop_input = input("  Apply Smart Crop (AI subject tracking, 9:16)? (y/n) [n]: ").strip().lower()
        apply_smart_crop = crop_input in ['y', 'yes', '1']
    except KeyboardInterrupt:
        print("\nCancelled")
        return
    
    # Generate output name
    vid_name = Path(selected_video).stem
    music_name = Path(selected_music).stem
    
    # Add time range to output name if specified
    time_suffix = ""
    if music_start is not None or music_end is not None:
        s = int(music_start) if music_start else 0
        e = int(music_end) if music_end else "end"
        time_suffix = f"_{s}s-{e}s"
    
    crop_suffix = "_9x16" if apply_smart_crop else ""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"output_{timestamp}_{vid_name[:20]}_{music_name[:20]}{time_suffix}{crop_suffix}.mp4"
    output_path = f"./output/{output_name}"
    
    os.makedirs('./output', exist_ok=True)
    
    print(f"\n" + "="*50)
    print("GENERATING VIDEO")
    print("="*50)
    print(f"  Input:  {Path(selected_video).name}")
    print(f"  Music:  {Path(selected_music).name}")
    if music_start or music_end:
        print(f"  Range:  {music_start or 0}s - {music_end or 'end'}s")
    if apply_smart_crop:
        print(f"  Format: 9:16 (Smart Crop with AI tracking)")
    else:
        print(f"  Format: Original aspect ratio")
    print(f"  Output: {output_name}")
    print("="*50 + "\n")
    
    # Trim music if needed
    trimmed_music = selected_music
    if music_start is not None or music_end is not None:
        import tempfile
        import subprocess
        
        trimmed_music = tempfile.mktemp(suffix='.mp3')
        cmd = ['ffmpeg', '-y', '-i', selected_music]
        
        if music_start:
            cmd.extend(['-ss', str(music_start)])
        if music_end:
            duration = music_end - (music_start or 0)
            cmd.extend(['-t', str(duration)])
        
        cmd.extend(['-c:a', 'copy', trimmed_music])
        
        print("Trimming music...")
        subprocess.run(cmd, capture_output=True, check=True)
    
    # Check models exist
    if not os.path.exists(os.path.join(models_dir, 'style_params.pkl')):
        print(f"Error: Models not found in {models_dir}/")
        print("Run training first: python main.py train --refs ./data --output ./models")
        return
    
    # Create auto editor
    editor = AutoEditor(config)
    
    # Load models
    editor.load_models(
        style_path=os.path.join(models_dir, 'style_params.pkl'),
        transition_data_path=os.path.join(models_dir, 'transition_data.pkl'),
        autoencoder_path=os.path.join(models_dir, 'transition_autoencoder.pt'),
        mapper_path=os.path.join(models_dir, 'music_mapper.pkl')
    )
    
    # Generate (to temp path if smart crop enabled)
    if apply_smart_crop:
        import tempfile
        temp_output = tempfile.mktemp(suffix='.mp4')
        
        editor.generate_from_video(
            input_video=selected_video,
            music_path=trimmed_music,
            output_path=temp_output,
            verbose=True
        )
        
        # Apply smart crop
        print("\n=== Applying Smart Crop (AI Subject Tracking) ===")
        from modules.smart_cropper import smart_crop_video
        smart_crop_video(temp_output, output_path, verbose=True)
        
        # Cleanup temp
        if os.path.exists(temp_output):
            os.remove(temp_output)
    else:
        editor.generate_from_video(
            input_video=selected_video,
            music_path=trimmed_music,
            output_path=output_path,
            verbose=True
        )
    
    # Cleanup trimmed music
    if trimmed_music != selected_music and os.path.exists(trimmed_music):
        os.remove(trimmed_music)
    
    print(f"\n✓ Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Video Editing Style Learning System"
    )
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train on reference videos to learn editing style'
    )
    train_parser.add_argument(
        '--refs', required=True,
        help='Directory containing reference edited videos'
    )
    train_parser.add_argument(
        '--output', default='./models',
        help='Output directory for learned models'
    )
    train_parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from existing models (incremental learning)'
    )
    
    # Device selection (mutually exclusive)
    device_group = train_parser.add_mutually_exclusive_group()
    device_group.add_argument(
        '--cpu', action='store_true',
        help='Use CPU for training'
    )
    device_group.add_argument(
        '--mps', action='store_true',
        help='Use Apple Silicon GPU (MPS) for training'
    )
    device_group.add_argument(
        '--gpu', type=int, nargs='?', const=1, metavar='N',
        help='Use NVIDIA GPU(s) for training. N = number of GPUs (default: 1, use -1 for all available)'
    )
    
    # Generate command
    gen_parser = subparsers.add_parser(
        'generate',
        help='Generate edited video using learned models'
    )
    gen_parser.add_argument(
        '--input',
        help='Single input video to extract scenes from'
    )
    gen_parser.add_argument(
        '--clips',
        help='Directory containing raw video clips'
    )
    gen_parser.add_argument(
        '--music', required=True,
        help='Path to music file'
    )
    gen_parser.add_argument(
        '--output', required=True,
        help='Output video path'
    )
    gen_parser.add_argument(
        '--models', default='./models',
        help='Directory containing learned models'
    )
    
    # Interactive command (default when no args)
    subparsers.add_parser(
        'run',
        help='Interactive mode - select video and music from folders'
    )
    
    # Validate command - check what was learned
    validate_parser = subparsers.add_parser(
        'validate',
        help='Show what the models learned from training'
    )
    validate_parser.add_argument(
        '--models', default='./models',
        help='Directory containing trained models'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'generate':
        generate(args)
    elif args.command == 'run':
        interactive()
    elif args.command == 'validate':
        validate_models(args)
    else:
        # Default to interactive mode
        interactive()


if __name__ == '__main__':
    main()

