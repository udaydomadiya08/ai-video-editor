"""
Motion Curve Extractor

Extracts time-series of editing parameters from video:
- Zoom Speed (derived from optical flow divergence)
- Camera Shake / Pan Speed
- Brightness / Contrast
- Cut detection (for segmentation)

These curves form the "Editing Pattern" of the reference video.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
import os


@dataclass
class EditingCurves:
    """Time-series of editing parameters."""
    zoom: np.ndarray       # Zoom factor per frame (1.0 = none, >1 = in, <1 = out)
    shift_x: np.ndarray    # Horizontal shift per frame (pixels)
    shift_y: np.ndarray    # Vertical shift per frame (pixels)
    brightness: np.ndarray # Mean brightness (0-255)
    contrast: np.ndarray   # Contrast (std dev)
    motion_energy: np.ndarray # Overall motion magnitude
    
    def __len__(self):
        return len(self.zoom)
        
    def to_array(self) -> np.ndarray:
        """Convert to (T, 6) array for VAE."""
        return np.stack([
            self.zoom, 
            self.shift_x, 
            self.shift_y, 
            self.brightness, 
            self.contrast,
            self.motion_energy
        ], axis=1)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'EditingCurves':
        return cls(
            zoom=arr[:, 0],
            shift_x=arr[:, 1],
            shift_y=arr[:, 2],
            brightness=arr[:, 3],
            contrast=arr[:, 4],
            motion_energy=arr[:, 5]
        )


class MotionCurveExtractor:
    """
    Extracts motion and style curves from video.
    """
    
    def __init__(self, sample_rate: int = 1):
        self.sample_rate = sample_rate
        
    def extract(self, video_path: str, verbose: bool = True) -> EditingCurves:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        zooms = []
        shift_xs = []
        shift_ys = []
        brights = []
        contrasts = []
        energies = []
        
        prev_gray = None
        prev_pts = None
        
        iterator = range(0, frame_count, self.sample_rate)
        if verbose:
            iterator = tqdm(iterator, desc=f"Extracting curves: {os.path.basename(video_path)}")
            
        for _ in iterator:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize for speed
            small = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            
            # Brightness/Contrast
            brights.append(np.mean(gray))
            contrasts.append(np.std(gray))
            
            # Motion
            zoom = 1.0
            sx = 0.0
            sy = 0.0
            energy = 0.0
            
            if prev_gray is not None:
                # Sparse Optical Flow (Lucas-Kanade) for robust affine estimation
                # Detect features
                if prev_pts is None or len(prev_pts) < 10:
                    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
                    
                if prev_pts is not None and len(prev_pts) > 0:
                    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                    
                    # Select good points
                    good_new = curr_pts[status==1]
                    good_old = prev_pts[status==1]
                    
                    if len(good_new) > 10:
                        # Estimate Partial Affine (scale + translation)
                        # [ s  0 tx ]
                        # [ 0  s ty ]
                        try:
                            M, _ = cv2.estimateAffinePartial2D(good_old, good_new)
                            if M is not None:
                                # Extract scale (zoom) and translation
                                sx = -M[0, 2] # Motion of Scene vs Camera (if scene moves right, cam pans left)
                                sy = -M[1, 2] # Actually we want Camera Motion. 
                                # If points move +10, checking how *camera* moved.
                                # If points moved +10 (right), camera moved -10 (left).
                                # But let's just stick to "Scene Motion" or "Transformation to apply".
                                # Edit pattern: We want to apply ZOOM.
                                # If ref video zooms in (points move out), scale > 1.
                                # M[0,0] is scale * cos(theta). Assuming small rotation.
                                scale = np.sqrt(M[0,0]**2 + M[0,1]**2)
                                
                                # If points expand (zoom in), scale > 1.
                                # We want to LEARN that.
                                zoom = scale
                                sx = M[0, 2]
                                sy = M[1, 2]
                                
                                energy = np.mean(np.sqrt((good_new - good_old)**2))
                        except Exception:
                            pass
                        
                    # Update points
                    prev_pts = good_new.reshape(-1, 1, 2)
            
            zooms.append(zoom)
            shift_xs.append(sx)
            shift_ys.append(sy)
            energies.append(energy)
            
            prev_gray = gray
            
        cap.release()
        
        # Pad first frame
        if len(zooms) < len(brights):
            zooms.insert(0, 1.0)
            shift_xs.insert(0, 0.0)
            shift_ys.insert(0, 0.0)
            energies.insert(0, 0.0)
            
        # Ensure equal lengths
        min_len = min(len(zooms), len(brights))
        
        return EditingCurves(
            zoom=np.array(zooms[:min_len]),
            shift_x=np.array(shift_xs[:min_len]),
            shift_y=np.array(shift_ys[:min_len]),
            brightness=np.array(brights[:min_len]),
            contrast=np.array(contrasts[:min_len]),
            motion_energy=np.array(energies[:min_len])
        )
