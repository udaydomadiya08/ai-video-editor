"""
Neural Style VAE (3D AdaIN)

A VAE that learns continuous video style (texture, color, grain, motion patterns).
Uses Adaptive Instance Normalization (AdaIN) in 3D to transfer style from a 
reference video patch to a content video patch.

Architecture:
- Content Encoder: 3D Conv -> Feature Map (F_c)
- Style Encoder: 3D Conv -> Style Mean/Var (mu_s, sigma_s)
- AdaIN: Normalize(F_c) * sigma_s + mu_s
- Decoder: 3D Conv -> Stylized Video
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm
import os

from config import get_config


class AdaIN3D(nn.Module):
    """
    Adaptive Instance Normalization for 3D features (video).
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        """
        content_feat: (B, C, T, H, W)
        style_feat: (B, C) or (B, C, 1, 1, 1) - Mean/std statistics
        """
        # Calculate content statistics
        # Mean/Std across spatial and temporal dimensions (leaving channel dimension)
        # Dimensions: 0=Batch, 1=Channel, 2=Time, 3=Height, 4=Width
        size = content_feat.size()
        N, C = size[:2]
        
        # Reshape to (N, C, -1) to compute stats
        c_flat = content_feat.view(N, C, -1)
        
        c_mean = c_flat.mean(dim=2).view(N, C, 1, 1, 1)
        c_std = c_flat.std(dim=2).view(N, C, 1, 1, 1) + 1e-5
        
        # Calculate style statistics from style features
        # Assuming style_feat is already the set of style statistics (mean, std)
        # Wait, usually style encoder outputs features, then we compute stats FROM those features.
        # But here let's assume style_feat IS the feature map of the style image/video.
        
        s_flat = style_feat.view(N, C, -1)
        s_mean = s_flat.mean(dim=2).view(N, C, 1, 1, 1)
        s_std = s_flat.std(dim=2).view(N, C, 1, 1, 1) + 1e-5
        
        # Normalize content
        normalized = (content_feat - c_mean) / c_std
        
        # Scale and shift with style
        return normalized * s_std + s_mean


class DeepStyleEncoder(nn.Module):
    """
    3D Encoder for style/content features.
    Similar to VGG but 3D for video.
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        
        self.net = nn.Sequential(
            # (B, 3, T, H, W) -> (B, 32, T, H/2, W/2)
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            # -> (B, 64, T, H/4, W/4)
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            # -> (B, 128, T, H/8, W/8)
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepStyleDecoder(nn.Module):
    """
    3D Decoder to reconstruct video from features.
    """
    
    def __init__(self, out_channels: int = 3, base_channels: int = 32):
        super().__init__()
        
        self.net = nn.Sequential(
            # (B, 128, T, H/8, W/8) -> (B, 64, T, H/4, W/4)
            nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 2, 2)),
            
            # -> (B, 32, T, H/2, W/2)
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 2, 2)),
            
            # -> (B, 3, T, H, W)
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # 0-1 output
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralStyleVAE(nn.Module):
    """
    Feature-based Autoencoder with AdaIN support.
    
    This effectively learns a "Universal Style Transfer" capability for video.
    """
    
    def __init__(self, base_channels: int = 64):
        super().__init__()
        
        self.encoder = DeepStyleEncoder(base_channels=base_channels)
        self.decoder = DeepStyleDecoder(base_channels=base_channels)
        self.adain = AdaIN3D()
        
    def forward(self, content: torch.Tensor, style: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If style is None, acts as normal autoencoder (reconstruct content).
        If style is provided, transfers style to content.
        """
        content_feat = self.encoder(content)
        
        if style is not None:
            style_feat = self.encoder(style)
            stylized_feat = self.adain(content_feat, style_feat)
        else:
            stylized_feat = content_feat
            
        out = self.decoder(stylized_feat)
        
        return out, content_feat
    
    def encode_style(self, style_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return style mean/std for storage."""
        style_feat = self.encoder(style_video)
        
        size = style_feat.size()
        N, C = size[:2]
        s_flat = style_feat.view(N, C, -1)
        s_mean = s_flat.mean(dim=2)
        s_std = s_flat.std(dim=2)
        
        return s_mean, s_std


class VideoStyleModel:
    """
    High-level interface for training and using the Deep Style VAE.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.device = self.config.device
        self.model = NeuralStyleVAE().to(self.device).float() # Using float for consistency
        self.train_losses = []
        
    def train(
        self,
        patches: np.ndarray,
        epochs: int = 50,
        batch_size: int = 4,
        lr: float = 1e-4,
        verbose: bool = True
    ):
        """
        Train the VAE on patches to learn video reconstruction.
        Patches: (N, C, T, H, W)
        """
        if verbose:
            print(f"  Training Deep Style Model on {len(patches)} patches...")
            
        dataset = TensorDataset(torch.FloatTensor(patches))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        
        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training Style VAE")
            
        for epoch in iterator:
            epoch_loss = 0.0
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                
                # Autoencoder mode: content and style are same
                recon, _ = self.model(x, style=None)
                
                loss = criterion(recon, x)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.train_losses.append(avg_loss)
            
            if verbose and (epoch+1) % 10 == 0:
                tqdm.write(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")
                
        if verbose:
            print(f"  Final Loss: {self.train_losses[-1]:.4f}")
            
    def transfer_style(
        self,
        content_frames: np.ndarray,
        style_patches: np.ndarray,
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Transfer style from patches to content frames.
        
        Args:
            content_frames: (T, H, W, C)
            style_patches: (N, C, T, H, W) - Reference patches
            alpha: 0-1 interpolation strength
        
        Returns:
            Stylized frames (T, H, W, C)
        """
        self.model.eval()
        
        # Prepare content
        # Reshape to (1, C, T, H, W)
        # Note: Content might be long, so we might need to process in chunks
        # But for now let's assume it fits or is resized
        
        h, w = content_frames.shape[1:3]
        
        # Ensure divisible by 8 (for 3 pooling layers)
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        
        # Resize content
        content_resized = []
        for f in content_frames:
            content_resized.append(cv2.resize(f, (new_w, new_h)))
        content_tensor = np.array(content_resized).transpose(3, 0, 1, 2) # (C, T, H, W)
        content_tensor = content_tensor[np.newaxis, ...].astype(np.float32) / 255.0
        content_tensor = torch.FloatTensor(content_tensor).to(self.device)
        
        # Prepare style
        # We average style features from multiple patches to get robust stats
        style_tensor = torch.FloatTensor(style_patches).to(self.device)
        
        with torch.no_grad():
            # 1. Encode content
            content_feat = self.model.encoder(content_tensor)
            
            # 2. Encode style (batch of patches)
            style_feats = self.model.encoder(style_tensor)
            
            # 3. Compute style stats
            # Average mean/std across all style patches
            N, C, T, H, W = style_feats.size()
            s_flat = style_feats.view(N, C, -1)
            s_mean = s_flat.mean(dim=2).mean(dim=0).view(1, C, 1, 1, 1) # Avg across batch
            s_std = s_flat.std(dim=2).mean(dim=0).view(1, C, 1, 1, 1)
            
            # 4. Compute content stats
            c_flat = content_feat.view(1, C, -1)
            c_mean = c_flat.mean(dim=2).view(1, C, 1, 1, 1)
            c_std = c_flat.std(dim=2).view(1, C, 1, 1, 1)
            
            # 5. AdaIN manually (since we averaged stats)
            normalized = (content_feat - c_mean) / (c_std + 1e-5)
            stylized_feat = normalized * s_std + s_mean
            
            # Interpolate if alpha < 1
            if alpha < 1.0:
                stylized_feat = (1 - alpha) * content_feat + alpha * stylized_feat
            
            # 6. Decode
            out = self.model.decoder(stylized_feat)
            
        # Post-process
        out_np = out.cpu().numpy()[0].transpose(1, 2, 3, 0) # (T, H, W, C)
        out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
        
        # Resize back to original
        final_frames = []
        for f in out_np:
            final_frames.append(cv2.resize(f, (w, h)))
            
        return np.array(final_frames)

    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'train_losses': self.train_losses
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.train_losses = checkpoint.get('train_losses', [])
