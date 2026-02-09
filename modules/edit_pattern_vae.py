"""
Editing Pattern VAE

A 1D Convolutional VAE that learns the "rhythm and flow" of editing parameters.
It learns patterns like:
- Snap zooms
- Rhythmic cuts (if integrated with cuts)
- Camera shake intensity
- Speed ramping curves

Uses 1D AdaIN to transfer the "motion style" of a reference video to a source video.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm
from config import get_config

class AdaIN1D(nn.Module):
    """Adaptive Instance Normalization for 1D time-series."""
    def __init__(self):
        super().__init__()
        
    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        """
        content_feat: (B, C, T)
        style_feat: (B, C) - Mean/std statistics
        """
        # Content stats
        size = content_feat.size()
        N, C, T = size
        
        c_mean = content_feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
        c_std = content_feat.view(N, C, -1).std(dim=2).view(N, C, 1) + 1e-5
        
        # Style stats (assuming style_feat is already stats or features)
        # If input is features (B, C, T), compute stats
        if style_feat.dim() == 3:
            s_mean = style_feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
            s_std = style_feat.view(N, C, -1).std(dim=2).view(N, C, 1) + 1e-5
        else:
            # Assume (B, 2*C) passed or something? 
            # Let's assume input is Features for flexibility
            raise ValueError("Style features must be (B, C, T)")
            
        normalized = (content_feat - c_mean) / c_std
        return normalized * s_std + s_mean


class CurveEncoder(nn.Module):
    """Encodes (B, C, T) curves to features."""
    def __init__(self, in_channels: int = 6, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels*2, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CurveDecoder(nn.Module):
    """Decodes features to (B, C, T) curves."""
    def __init__(self, out_channels: int = 6, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, out_channels, kernel_size=5, padding=2),
            # No sigmoid/tanh because motion params can be unbounded (e.g. shift)
            # But usually we normalize inputs.
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EditPatternVAE(nn.Module):
    def __init__(self, in_channels: int = 6, base_channels: int = 32):
        super().__init__()
        self.encoder = CurveEncoder(in_channels, base_channels)
        self.decoder = CurveDecoder(in_channels, base_channels)
        self.adain = AdaIN1D()
        
    def forward(self, content: torch.Tensor, style: Optional[torch.Tensor] = None) -> torch.Tensor:
        c_feat = self.encoder(content)
        
        if style is not None:
            s_feat = self.encoder(style)
            stylized = self.adain(c_feat, s_feat)
        else:
            stylized = c_feat
            
        return self.decoder(stylized)


class CurveStyleModel:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.device = self.config.device
        self.model = EditPatternVAE(in_channels=6).to(self.device)
        
    def train(self, curves: np.ndarray, epochs=50, batch_size=16):
        """Train on collected curves."""
        # curves: (N, T, C) -> (N, C, T)
        data = torch.FloatTensor(curves.transpose(0, 2, 1))
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
                recon = self.model(x, style=None) # Autoencoder mode
                loss = criterion(recon, x)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if (epoch+1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")
                
    def transfer_style(self, content_curve: np.ndarray, style_curve: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Apply style to content curve."""
        self.model.eval()
        with torch.no_grad():
            c = torch.FloatTensor(content_curve.T).unsqueeze(0).to(self.device) # (1, C, T)
            s = torch.FloatTensor(style_curve.T).unsqueeze(0).to(self.device)   # (1, C, T)
            
            # Encode
            c_feat = self.model.encoder(c)
            s_feat = self.model.encoder(s)
            
            # AdaIN
            stylized = self.model.adain(c_feat, s_feat)
            
            # Interpolate features
            if strength < 1.0:
                stylized = (1 - strength) * c_feat + strength * stylized
            
            # Decode
            out = self.model.decoder(stylized)
            
        return out.cpu().numpy()[0].T # (T, C)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
