
import os
import sys
from modules.style_learner import StyleLearner
from modules.transition_frame_extractor import TransitionFrameExtractor
from config import get_config

def main():
    config = get_config()
    
    # Path to a test video
    video_path = "data/YTDown.com_Shorts_Mr-Perfection-Henry-Cavill-Edit-Canto-De_Media_Kv4JrG6h25U_001_1080p.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    print(f"Testing on: {video_path}")
    
    # 1. Detect Cuts
    print("\n--- Step 1: Detect Cuts ---")
    style_learner = StyleLearner(config)
    cuts = style_learner.detect_cuts(video_path, verbose=True)
    print(f"Detected {len(cuts)} cuts: {cuts}")
    
    if not cuts:
        print("No cuts found! Cannot extract.")
        return

    # 2. Extract Transitions
    print("\n--- Step 2: Extract Transitions ---")
    extractor = TransitionFrameExtractor(
        window_frames=5,
        output_size=(128, 128)
    )
    
    sequences = extractor.extract_transitions(video_path, cuts, verbose=True)
    print(f"\nExtracted {len(sequences)} valid sequences")
    
    if len(sequences) > 0:
        print("Success! The extractor works.")
        print(f"Shape of first sequence: {sequences[0].full_sequence.shape}")
    else:
        print("Failure! No sequences extracted.")

if __name__ == "__main__":
    main()
