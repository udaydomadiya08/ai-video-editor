"""
Neural Universal Visual Effect VAE

A 3D Convolutional VAE that learns "Universal Dynamic Fields".
Input: (Batch, 3, Time, H, W) field grids.
Channels:
    0: Horizontal Flow (Motion)
    1: Vertical Flow (Motion)
    2: Intensity Change (Visual Energy / Flash / Strobe)

Latent Space: "Dynamic Texture" (e.g. Handheld Shake + Strobe, Zoom + Fade).
Uses AdaIN to transfer these dynamics from reference to source.
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
        size = x.size()
        N, C, T, H, W = size
        
        style_mean = style_mean.view(N, C, 1, 1, 1)
        style_std = style_std.view(N, C, 1, 1, 1) + 1e-5
        
        return self.norm(x) * style_std + style_mean

class EffectEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        # Input: (B, 3, T=64, H=9, W=16)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=(2, 1, 2)), # T/2, H, W/2 -> (32, 9, 8)
            
            nn.Conv3d(base_channels, base_channels*2, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # T/4, H/2, W/4 -> (16, 4, 4)
            
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=(3, 3, 3), padding=1),
            nn.LeakyReLU(0.2),
            # Output: (B, 128, 16, 4, 4)
        )

class EffectDecoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=64):
        super().__init__()
        # Input: (B, 128, 16, 4, 4)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=(3, 3, 3), padding=1),
            nn.Upsample(scale_factor=(2, 2, 2)), # -> (32, 8, 8)
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=(3, 3, 3), padding=1),
            nn.Upsample(scale_factor=(2, 1, 2)), # -> (64, 8, 16)
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(base_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
        )
        
    def forward(self, x, target_size):
        out = self.net(x)
        # Interpolate to match target size (T, H, W) exactly
        if out.shape[2:] != target_size:
            out = nn.functional.interpolate(out, size=target_size, mode='trilinear')
        return out


class NeuralEffectVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EffectEncoder(in_channels=3)
        self.decoder = EffectDecoder(out_channels=3)
        self.adain = AdaptiveInstanceNorm3d(128) # latent channels
        
    def calc_stats(self, feat):
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
            
        T, H, W = content.shape[2:]
        return self.decoder(stylized, (T, H, W))


class UniversalEffectModel:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.device = self.config.device
        self.model = NeuralEffectVAE().to(self.device)
        
    def train(self, field_grids: np.ndarray, epochs=50, batch_size=8):
        """Train on (N, 3, T, H, W) field grids."""
        data = torch.FloatTensor(field_grids)
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
                
    def transfer_effect(self, content_grid: np.ndarray, style_grid: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Transfer dynamic effect style from reference to content."""
        self.model.eval()
        with torch.no_grad():
            c = torch.FloatTensor(content_grid).unsqueeze(0).to(self.device) # (1, 3, T, H, W)
            s = torch.FloatTensor(style_grid).unsqueeze(0).to(self.device)   # (1, 3, T, H, W)
            
            c_feat = self.model.encoder(c)
            s_feat = self.model.encoder(s)
            s_mean, s_std = self.model.calc_stats(s_feat)
            
            stylized = self.model.adain(c_feat, s_mean, s_std)
            
            if strength < 1.0:
                stylized = (1 - strength) * c_feat + strength * stylized
                
            out = self.model.decoder(stylized, c.shape[2:])
            
        return out.cpu().numpy()[0] # (3, T, H, W)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
