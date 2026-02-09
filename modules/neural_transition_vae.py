"""
Neural Transition VAE

A Variational Autoencoder that learns the latent space of transition transformations.
Unlike fixed effect categories (zoom, flash, etc.), this learns ANY visual effect
from the actual pixel transformations in reference videos.

Architecture:
- Encoder: 3D conv layers to process frame sequences → latent code
- Decoder: Latent code → 3D conv layers → reconstructed frames
- Latent: 64-dim continuous space where similar transitions are close together
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
from tqdm import tqdm

from config import get_config


class Conv3DEncoder(nn.Module):
    """
    3D Convolutional Encoder for transition sequences.
    
    Input: (batch, channels, frames, height, width)
    Output: (mu, logvar) for VAE latent space
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 64,
        base_channels: int = 32
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # (B, 3, F, H, W) -> (B, 32, F, H/2, W/2)
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(base_channels),
            nn.LeakyReLU(0.2),
            
            # -> (B, 64, F, H/4, W/4)
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2),
            
            # -> (B, 128, F/2, H/8, W/8)
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2),
            
            # -> (B, 256, F/4, H/16, W/16)
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.LeakyReLU(0.2),
        )
        
        # Will be set dynamically based on input size
        self.fc_input_size = None
        self.fc_mu = None
        self.fc_logvar = None
        self.latent_dim = latent_dim
        self.base_channels = base_channels
    
    def _init_fc_layers(self, x: torch.Tensor):
        """Initialize FC layers based on encoder output size."""
        with torch.no_grad():
            h = self.encoder(x)
            self.fc_input_size = h.view(h.size(0), -1).size(1)
            
        self.fc_mu = nn.Linear(self.fc_input_size, self.latent_dim).to(x.device)
        self.fc_logvar = nn.Linear(self.fc_input_size, self.latent_dim).to(x.device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize FC layers on first forward pass
        if self.fc_mu is None:
            self._init_fc_layers(x)
        
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Conv3DDecoder(nn.Module):
    """
    3D Convolutional Decoder for transition sequences.
    
    Input: latent code
    Output: (batch, channels, frames, height, width)
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        out_channels: int = 3,
        base_channels: int = 32,
        output_frames: int = 10,
        output_size: Tuple[int, int] = (128, 128)
    ):
        super().__init__()
        
        self.output_frames = output_frames
        self.output_height, self.output_width = output_size
        self.base_channels = base_channels
        
        # Calculate initial spatial dimensions (after all downsampling)
        self.init_frames = max(1, output_frames // 4)
        self.init_h = output_size[0] // 16
        self.init_w = output_size[1] // 16
        
        self.fc = nn.Linear(latent_dim, base_channels * 8 * self.init_frames * self.init_h * self.init_w)
        
        self.decoder = nn.Sequential(
            # (B, 256, F/4, H/16, W/16) -> (B, 128, F/2, H/8, W/8)
            nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=4, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2),
            
            # -> (B, 64, F, H/4, W/4)
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=4, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2),
            
            # -> (B, 32, F, H/2, W/2)
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels),
            nn.LeakyReLU(0.2),
            
            # -> (B, 3, F, H, W)
            nn.ConvTranspose3d(base_channels, out_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.base_channels * 8, self.init_frames, self.init_h, self.init_w)
        
        out = self.decoder(h)
        
        # Ensure output matches expected dimensions
        if out.shape[2:] != (self.output_frames, self.output_height, self.output_width):
            out = nn.functional.interpolate(
                out, 
                size=(self.output_frames, self.output_height, self.output_width),
                mode='trilinear', 
                align_corners=False
            )
        
        return out


class TransitionSequenceVAE(nn.Module):
    """
    Variational Autoencoder for learning transition transformations.
    
    Learns a continuous latent space where:
    - Similar transitions (zooms, flashes, custom effects) cluster together
    - New transitions can be sampled and interpolated
    - ANY visual effect can be represented, not just predefined categories
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        base_channels: int = 32,
        output_frames: int = 10,
        output_size: Tuple[int, int] = (128, 128)
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_frames = output_frames
        self.output_size = output_size
        
        self.encoder = Conv3DEncoder(
            in_channels=3,
            latent_dim=latent_dim,
            base_channels=base_channels
        )
        
        self.decoder = Conv3DDecoder(
            latent_dim=latent_dim,
            out_channels=3,
            base_channels=base_channels,
            output_frames=output_frames,
            output_size=output_size
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, frames, height, width)
            
        Returns:
            x_recon: Reconstructed tensor
            mu: Latent mean
            logvar: Latent log-variance
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (returns mean)."""
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample random transitions from the prior."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss: reconstruction + KL divergence.
    
    Args:
        x_recon: Reconstructed sequences
        x: Original sequences
        mu: Latent mean
        logvar: Latent log-variance
        beta: Weight for KL term (beta-VAE)
        
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


class NeuralTransitionVAE:
    """
    High-level interface for training and using the transition sequence VAE.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model: Optional[TransitionSequenceVAE] = None
        self.device = self.config.device
        
        # Training stats
        self.train_losses = []
        
        # Model parameters (set during training)
        # M1 CPU Optimized: 64-dim latent space, 32-base channels
        self.latent_dim = 64
        self.base_channels = 32 
        self.output_frames = 20 # Supports longer, complex transitions
        self.output_size = (128, 128)
    
    def train(
        self,
        transition_sequences: np.ndarray,
        epochs: int = 150, # More epochs for deep learning
        batch_size: int = 8,
        lr: float = 1e-4,
        beta: float = 0.05, # Lower beta = more reconstruction fidelity (less regularization)
        verbose: bool = True
    ) -> 'NeuralTransitionVAE':
        """
        Train the VAE on extracted transition sequences.
        
        Args:
            transition_sequences: Array (N, frames, H, W, C) or (N, C, frames, H, W)
                                  Normalized to 0-1 range
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            beta: KL divergence weight
            verbose: Show progress
            
        Returns:
            self for chaining
        """
        # Ensure correct format (N, C, F, H, W) for PyTorch 3D conv
        if transition_sequences.ndim == 5:
            if transition_sequences.shape[-1] == 3:  # (N, F, H, W, C)
                # Convert to (N, C, F, H, W)
                transition_sequences = np.transpose(transition_sequences, (0, 4, 1, 2, 3))
        
        n_samples, n_channels, n_frames, h, w = transition_sequences.shape
        
        self.output_frames = n_frames
        self.output_size = (h, w)
        
        if verbose:
            print(f"  Training data shape: {transition_sequences.shape}")
            print(f"  Frames per transition: {n_frames}")
            print(f"  Frame size: {h}x{w}")
        
        # Initialize model
        # Initialize model with High Capacity
        self.model = TransitionSequenceVAE(
            latent_dim=self.latent_dim,
            base_channels=self.base_channels, # 64
            output_frames=n_frames,
            output_size=(h, w)
        ).to(self.device)
        
        # Create dataloader
        dataset = TensorDataset(torch.FloatTensor(transition_sequences))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        
        # Training loop
        self.model.train()
        epoch_iter = range(epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training Neural Transition VAE")
        
        for epoch in epoch_iter:
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                x_recon, mu, logvar = self.model(x)
                loss, recon, kl = vae_loss(x_recon, x, mu, logvar, beta)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += recon.item()
                epoch_kl += kl.item()
            
            n_batches = len(dataloader)
            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                tqdm.write(f"  Epoch {epoch+1}: loss={avg_loss:.4f} (recon={epoch_recon/n_batches:.4f}, kl={epoch_kl/n_batches:.4f})")
        
        # Training summary
        if verbose:
            print(f"\n  === Neural Transition VAE Training Summary ===")
            print(f"  Final loss: {self.train_losses[-1]:.4f}")
            print(f"  Initial loss: {self.train_losses[0]:.4f}")
            if self.train_losses[0] > 0:
                improvement = (self.train_losses[0] - self.train_losses[-1]) / self.train_losses[0] * 100
                print(f"  Improvement: {improvement:.1f}%")
        
        return self
    
    def encode(self, sequences: np.ndarray) -> np.ndarray:
        """Encode transition sequences to latent embeddings."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        # Ensure correct format
        if sequences.ndim == 5 and sequences.shape[-1] == 3:
            sequences = np.transpose(sequences, (0, 4, 1, 2, 3))
        
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sequences).to(self.device)
            z = self.model.encode(x)
            return z.cpu().numpy()
    
    def decode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Decode latent embeddings to transition sequences.
        
        Returns: Array (N, frames, H, W, C) in 0-1 range
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            z = torch.FloatTensor(embeddings).to(self.device)
            x_recon = self.model.decode(z)
            # Convert back to (N, F, H, W, C)
            x_recon = x_recon.permute(0, 2, 3, 4, 1).cpu().numpy()
            return x_recon
    
    def sample(self, num_samples: int) -> np.ndarray:
        """Sample random transitions from the learned distribution."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            x = self.model.decode(z)
            # Convert to (N, F, H, W, C)
            x = x.permute(0, 2, 3, 4, 1).cpu().numpy()
            return x
    
    def interpolate(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        n_steps: int = 5
    ) -> np.ndarray:
        """
        Interpolate between two transition styles.
        
        Returns: Array of transitions at interpolated positions
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        alphas = np.linspace(0, 1, n_steps)
        interpolated = []
        
        for alpha in alphas:
            z = (1 - alpha) * embedding1 + alpha * embedding2
            decoded = self.decode(z.reshape(1, -1))
            interpolated.append(decoded[0])
        
        return np.array(interpolated)
    
    def save(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        torch.save({
            'model_state': self.model.state_dict(),
            'latent_dim': self.latent_dim,
            'output_frames': self.output_frames,
            'output_size': self.output_size,
            'train_losses': self.train_losses
        }, path)
    
    def load(self, path: str) -> 'NeuralTransitionVAE':
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.latent_dim = checkpoint['latent_dim']
        self.output_frames = checkpoint['output_frames']
        self.output_size = checkpoint['output_size']
        self.train_losses = checkpoint.get('train_losses', [])
        
        self.model = TransitionSequenceVAE(
            latent_dim=self.latent_dim,
            base_channels=32,
            output_frames=self.output_frames,
            output_size=self.output_size
        ).to(self.device)
        
        # Initialize FC layers by running a forward pass on dummy data
        # This is required because FC layers are created dynamically based on input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.output_frames, self.output_size[0], self.output_size[1]).to(self.device)
            self.model.encoder(dummy_input)
        
        self.model.load_state_dict(checkpoint['model_state'])
        
        return self
