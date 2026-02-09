"""
Neural Motion VAE

A 3D Convolutional VAE that learns dense semantic motion fields.
Input: (Batch, 2, Time, H, W) optical flow grids.
Latent Space: "Motion Texture" (e.g. Handheld, Dolly Zoom, Whip Pan).

Uses AdaIN to transfer motion style from reference loops to source clips.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, List
from config import get_config

class AdaptiveInstanceNorm3d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm3d(num_features, affine=False)
        
    def forward(self, x, style_mean, style_std):
        # x: (B, C, T, H, W)
        # style: (B, C) -> expand to (B, C, 1, 1, 1)
        size = x.size()
        N, C, T, H, W = size
        
        style_mean = style_mean.view(N, C, 1, 1, 1)
        style_std = style_std.view(N, C, 1, 1, 1) + 1e-5
        
        return self.norm(x) * style_std + style_mean

class MotionEncoder(nn.Module):
    def __init__(self, in_channels=2, base_channels=32):
        super().__init__()
        # Input: (B, 2, T=64, H=9, W=16)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 2)), # T/2, H, W/2 -> (32, 9, 8)
            
            nn.Conv3d(base_channels, base_channels*2, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # T/4, H/2, W/4 -> (16, 4, 4)
            
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            # Output: (B, 128, 16, 4, 4)
        )
        
    def forward(self, x):
        return self.net(x)

class MotionDecoder(nn.Module):
    def __init__(self, out_channels=2, base_channels=32):
        super().__init__()
        # Input: (B, 128, 16, 4, 4)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=(3, 3, 3), padding=1),
            nn.Upsample(scale_factor=(2, 2, 2)), # -> (32, 8, 8) -> target H=9?
            nn.ReLU(),
            
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=(3, 3, 3), padding=1),
            nn.Upsample(scale_factor=(2, 1, 2)), # -> (64, 8, 16) -> target H=9?
            nn.ReLU(),
            
            nn.Conv3d(base_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            # Final resize to exactly (T, H, W) if upsampling was imperfect
        )
        
    def forward(self, x, target_size):
        out = self.net(x)
        # Interpolate to match target size (T, H, W) exactly
        if out.shape[2:] != target_size:
            out = nn.functional.interpolate(out, size=target_size, mode='trilinear')
        return out


class NeuralMotionVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MotionEncoder()
        self.decoder = MotionDecoder()
        self.adain = AdaptiveInstanceNorm3d(128) # latent channels
        
    def calc_stats(self, feat):
        # Calculate mean/std across spatial/temporal dimensions
        # feat: (B, C, T, H, W)
        N, C, T, H, W = feat.size()
        feat_view = feat.view(N, C, -1)
        mean = feat_view.mean(dim=2)
        std = feat_view.std(dim=2)
        return mean, std

    def forward(self, content, style=None):
        c_feat = self.encoder(content)
        
        if style is not None:
            s_feat = self.encoder(style)
            s_mean, s_std = self.calc_stats(s_feat)
            stylized = self.adain(c_feat, s_mean, s_std)
        else:
            stylized = c_feat
            
        # Target size from input
        T, H, W = content.shape[2:]
        return self.decoder(stylized, (T, H, W))


class SemanticMotionModel:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.device = self.config.device
        self.model = NeuralMotionVAE().to(self.device)
        
    def train(self, motion_grids: np.ndarray, epochs=50, batch_size=8):
        """Train on (N, 2, T, H, W) flow grids."""
        data = torch.FloatTensor(motion_grids)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        opt = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0].to(self.device)
                opt.zero_grad()
                recon = self.model(x)
                loss = criterion(recon, x)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if (epoch+1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
                
    def transfer_motion(self, content_grid: np.ndarray, style_grid: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Transfer motion style from reference to content."""
        self.model.eval()
        with torch.no_grad():
            c = torch.FloatTensor(content_grid).unsqueeze(0).to(self.device) # (1, 2, T, H, W)
            s = torch.FloatTensor(style_grid).unsqueeze(0).to(self.device)   # (1, 2, T, H, W)
            
            c_feat = self.model.encoder(c)
            s_feat = self.model.encoder(s)
            s_mean, s_std = self.model.calc_stats(s_feat)
            
            stylized = self.model.adain(c_feat, s_mean, s_std)
            
            if strength < 1.0:
                # Interpolate feat
                stylized = (1 - strength) * c_feat + strength * stylized
                
            out = self.model.decoder(stylized, c.shape[2:])
            
        return out.cpu().numpy()[0] # (2, T, H, W)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
