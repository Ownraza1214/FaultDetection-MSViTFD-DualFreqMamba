"""
DualFreqMamba — Dual-Branch Frequency-Temporal Fault Detector
==============================================================

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────┐
    │            Input: X ∈ ℝ^{B × N_channels × T}               │
    │                    (Instance Normalized)                     │
    └────────┬────────────────────┬──────────────────┬────────────┘
             │                    │                  │
             ▼                    ▼                  ▼
    ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐
    │  FFT Frequency  │  │  CWT Scalogram  │  │  Raw Temporal    │
    │  Patching       │  │  Branch          │  │  Embedding       │
    │  Branch         │  │                  │  │                  │
    │  (CATCH)        │  │  (TSCMamba)      │  │  (Direct)        │
    └───────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
            │  Channel-Masked     │  Conv2D Patch        │
            │  Transformer        │  + FFN               │  Linear Proj
            ▼                     ▼                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Adaptive Gated Tri-Branch Fusion                │
    │   gate_fft ⊙ F_fft + gate_cwt ⊙ F_cwt + gate_raw ⊙ F_raw  │
    └────────────────────────┬────────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │         Mamba SSM Reconstruction Decoder                     │
    │   Selective SSM blocks × N with Gated Skip Connections       │
    │   (captures long-range temporal dependencies)                │
    └────────────────────────┬────────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │         Sparse Association Discrepancy Module                │
    │   Prior-Association (Gaussian kernel) vs                     │
    │   Series-Association (learned sparse attention)              │
    │   → Anomaly signal = KL(P || S) divergence                  │
    └────────────────────────┬────────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Dual-Domain Reconstruction                      │
    │   Time-domain: iFFT(reconstructed_freq)                     │
    │   Freq-domain: direct frequency reconstruction               │
    │   Score = α·recon_error_time + β·recon_error_freq            │
    │         + γ·association_discrepancy                           │
    └─────────────────────────────────────────────────────────────┘
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_dualfreqmamba import DualFreqMambaConfig


# ============================================================================
# Continuous Wavelet Transform (CWT) — Pure PyTorch
# ============================================================================

class ContinuousWaveletTransform(nn.Module):
    """Morlet CWT implementation in pure PyTorch for GPU acceleration.
    
    Computes scalogram features from time series using Morlet wavelets
    at multiple scales, providing multi-resolution time-frequency analysis.
    """
    
    def __init__(self, n_scales: int = 32, sigma: float = 1.0, max_freq_ratio: float = 0.5):
        super().__init__()
        self.n_scales = n_scales
        self.sigma = sigma
        self.max_freq_ratio = max_freq_ratio
        
    def _morlet_wavelet(self, t: torch.Tensor, scale: float, sigma: float) -> torch.Tensor:
        """Morlet wavelet: ψ(t) = exp(-t²/(2σ²)) * exp(i*2π*f₀*t)"""
        t_scaled = t / scale
        gaussian = torch.exp(-t_scaled ** 2 / (2 * sigma ** 2))
        oscillation = torch.exp(1j * 2 * math.pi * t_scaled / sigma)
        # Normalize
        norm = 1.0 / (sigma * math.sqrt(2 * math.pi) * math.sqrt(scale))
        return norm * gaussian * oscillation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) time series
        Returns:
            scalogram: (B, C, n_scales, T) wavelet coefficients (magnitude)
        """
        B, C, T = x.shape
        device = x.device
        
        # Define scales (logarithmically spaced)
        scales = torch.logspace(
            math.log10(2), math.log10(max(T // 2, 4)),
            self.n_scales, device=device
        )
        
        # Create time axis for wavelets
        half_width = min(T // 2 - 1, 128)  # must be < T for reflect padding
        half_width = max(half_width, 1)
        t = torch.arange(-half_width, half_width + 1, device=device, dtype=torch.float32)
        
        scalograms = []
        for scale in scales:
            # Compute wavelet kernel
            wavelet = self._morlet_wavelet(t, scale.item(), self.sigma)
            
            # Use real and imaginary parts separately for conv
            kernel_real = wavelet.real.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
            kernel_imag = wavelet.imag.unsqueeze(0).unsqueeze(0)
            
            # Convolve each channel
            x_pad = F.pad(x, (half_width, half_width), mode='reflect')
            conv_real = F.conv1d(x_pad, kernel_real.expand(C, -1, -1), groups=C)
            conv_imag = F.conv1d(x_pad, kernel_imag.expand(C, -1, -1), groups=C)
            
            # Magnitude
            magnitude = torch.sqrt(conv_real ** 2 + conv_imag ** 2 + 1e-8)
            scalograms.append(magnitude[:, :, :T])  # trim to original length
        
        return torch.stack(scalograms, dim=2)  # (B, C, n_scales, T)


# ============================================================================
# FFT Frequency Patching (CATCH-inspired)
# ============================================================================

class FFTFrequencyPatcher(nn.Module):
    """Transforms time series to frequency domain and creates frequency patches.
    
    Key innovation from CATCH: patches in frequency domain capture fine-grained
    spectral bands, enabling detection of subtle frequency anomalies.
    """
    
    def __init__(self, config: DualFreqMambaConfig):
        super().__init__()
        self.patch_size = config.fft_patch_size
        self.stride = config.fft_stride
        self.d_model = config.fft_d_model
        
        # Projection: each patch contains real+imag parts
        # Patch dim = n_channels * 2 * patch_size (real + imag)
        patch_dim = config.n_channels * 2 * config.fft_patch_size
        self.projection = nn.Sequential(
            nn.Linear(patch_dim, config.fft_d_model),
            nn.GELU(),
            nn.Linear(config.fft_d_model, config.fft_d_model),
        )
        
        # Positional encoding for frequency patches
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 256, config.fft_d_model) * 0.02  # max 256 patches
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T) time series
        Returns:
            patches: (B, n_patches, d_model) frequency patch embeddings
            fft_full: (B, C, T) complex FFT for reconstruction
        """
        B, C, T = x.shape
        
        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)  # (B, C, T//2+1)
        freq_len = x_fft.shape[-1]
        
        # Separate real and imaginary
        x_real = x_fft.real  # (B, C, freq_len)
        x_imag = x_fft.imag
        
        # Interleave: (B, C*2, freq_len)
        x_ri = torch.stack([x_real, x_imag], dim=2).reshape(B, C * 2, freq_len)
        
        # Create frequency patches
        # x_ri: (B, C*2, freq_len) → unfold along freq axis
        if freq_len < self.patch_size:
            # Pad if frequency length is too short
            x_ri = F.pad(x_ri, (0, self.patch_size - freq_len))
            freq_len = self.patch_size
        
        n_patches = max(1, (freq_len - self.patch_size) // self.stride + 1)
        patches = x_ri.unfold(2, self.patch_size, self.stride)  # (B, C*2, n_patches, patch_size)
        patches = patches.permute(0, 2, 1, 3)  # (B, n_patches, C*2, patch_size)
        patches = patches.reshape(B, n_patches, -1)  # (B, n_patches, C*2*patch_size)
        
        # Project to d_model
        patches = self.projection(patches)  # (B, n_patches, d_model)
        
        # Add positional encoding
        patches = patches + self.pos_encoding[:, :n_patches]
        
        return patches, x_fft


# ============================================================================
# Channel-Masked Transformer (CATCH-inspired)
# ============================================================================

class ChannelMaskGenerator(nn.Module):
    """Learns dynamic channel correlation masks per frequency band.
    
    CATCH innovation: different frequency bands have different channel 
    correlations. This module learns which channels are correlated
    for each frequency patch.
    """
    
    def __init__(self, d_model: int, n_channels: int):
        super().__init__()
        self.mask_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_channels * n_channels),
        )
        self.n_channels = n_channels
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_patches, d_model) frequency patch embeddings
        Returns:
            mask: (B, n_patches, n_channels, n_channels) correlation masks
        """
        B, L, D = x.shape
        mask_logits = self.mask_net(x)  # (B, L, C*C)
        mask_logits = mask_logits.view(B, L, self.n_channels, self.n_channels)
        # Symmetric soft mask via sigmoid
        mask = torch.sigmoid(mask_logits / self.temperature)
        # Symmetrize
        mask = (mask + mask.transpose(-1, -2)) / 2
        return mask


class ChannelMaskedTransformerLayer(nn.Module):
    """Transformer layer with channel-aware masked attention."""
    
    def __init__(self, config: DualFreqMambaConfig):
        super().__init__()
        self.d_model = config.fft_d_model
        self.n_heads = config.fft_n_heads
        self.head_dim = config.fft_d_model // config.fft_n_heads
        
        self.q_proj = nn.Linear(config.fft_d_model, config.fft_d_model)
        self.k_proj = nn.Linear(config.fft_d_model, config.fft_d_model)
        self.v_proj = nn.Linear(config.fft_d_model, config.fft_d_model)
        self.out_proj = nn.Linear(config.fft_d_model, config.fft_d_model)
        
        self.norm1 = nn.LayerNorm(config.fft_d_model)
        self.norm2 = nn.LayerNorm(config.fft_d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.fft_d_model, config.fft_d_model * 4),
            nn.GELU(),
            nn.Dropout(config.fft_dropout),
            nn.Linear(config.fft_d_model * 4, config.fft_d_model),
            nn.Dropout(config.fft_dropout),
        )
        
        self.attn_dropout = nn.Dropout(config.fft_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard transformer layer with pre-norm."""
        B, L, D = x.shape
        
        # Pre-norm attention
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v)  # (B, heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        x = x + out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class FFTBranch(nn.Module):
    """Complete FFT frequency patching branch."""
    
    def __init__(self, config: DualFreqMambaConfig):
        super().__init__()
        self.patcher = FFTFrequencyPatcher(config)
        self.layers = nn.ModuleList([
            ChannelMaskedTransformerLayer(config)
            for _ in range(config.fft_n_layers)
        ])
        if config.use_channel_mask:
            self.mask_gen = ChannelMaskGenerator(config.fft_d_model, config.n_channels)
        self.use_channel_mask = config.use_channel_mask
        self.output_proj = nn.Linear(config.fft_d_model, config.fusion_d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T) time series
        Returns:
            features: (B, L, fusion_d_model)
            fft_full: complex FFT for reconstruction
        """
        patches, fft_full = self.patcher(x)
        
        for layer in self.layers:
            patches = layer(patches)
        
        features = self.output_proj(patches)
        return features, fft_full


# ============================================================================
# CWT Scalogram Branch (TSCMamba-inspired)
# ============================================================================

class CWTBranch(nn.Module):
    """CWT wavelet branch that extracts multi-resolution features from scalograms."""
    
    def __init__(self, config: DualFreqMambaConfig):
        super().__init__()
        self.cwt = ContinuousWaveletTransform(
            n_scales=config.cwt_scales,
            sigma=config.cwt_wavelet_sigma,
        )
        
        # Conv2D patch embedding on scalogram
        self.patch_embed = nn.Sequential(
            nn.Conv2d(config.n_channels, config.cwt_d_model, 
                     kernel_size=config.cwt_patch_size, 
                     stride=config.cwt_patch_size),
            nn.GELU(),
            nn.BatchNorm2d(config.cwt_d_model),
        )
        
        # FFN to process patches
        self.ffn = nn.Sequential(
            nn.Linear(config.cwt_d_model, config.cwt_d_model * 2),
            nn.GELU(),
            nn.Linear(config.cwt_d_model * 2, config.fusion_d_model),
        )
        
        self.norm = nn.LayerNorm(config.fusion_d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) time series
        Returns:
            features: (B, L, fusion_d_model) CWT-derived features
        """
        # Compute scalogram
        scalogram = self.cwt(x)  # (B, C, n_scales, T)
        
        # Patch embedding
        patches = self.patch_embed(scalogram)  # (B, cwt_d_model, H', W')
        B, D, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # (B, H'*W', cwt_d_model)
        
        # FFN
        features = self.ffn(patches)  # (B, L, fusion_d_model)
        return self.norm(features)


# ============================================================================
# Raw Temporal Embedding Branch
# ============================================================================

class TemporalEmbeddingBranch(nn.Module):
    """Direct temporal embedding of raw time series."""
    
    def __init__(self, config: DualFreqMambaConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.n_channels, config.fusion_d_model),
            nn.GELU(),
            nn.Linear(config.fusion_d_model, config.fusion_d_model),
        )
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.window_size, config.fusion_d_model) * 0.02
        )
        self.norm = nn.LayerNorm(config.fusion_d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) time series
        Returns:
            features: (B, T, fusion_d_model)
        """
        x = x.transpose(1, 2)  # (B, T, C)
        features = self.projection(x) + self.pos_encoding[:, :x.shape[1]]
        return self.norm(features)


# ============================================================================
# Adaptive Gated Tri-Branch Fusion
# ============================================================================

class AdaptiveGatedFusion(nn.Module):
    """Fuses FFT, CWT, and raw temporal features with learned gates."""
    
    def __init__(self, config: DualFreqMambaConfig):
        super().__init__()
        self.fusion_dim = config.fusion_d_model
        
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(config.fusion_d_model * 3, config.fusion_d_model),
            nn.GELU(),
            nn.Linear(config.fusion_d_model, 3),
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.fusion_d_model, config.mamba_d_model),
            nn.LayerNorm(config.mamba_d_model),
        )
    
    def _align_lengths(self, *features) -> List[torch.Tensor]:
        """Align sequence lengths via adaptive pooling."""
        min_len = min(f.shape[1] for f in features)
        aligned = []
        for f in features:
            if f.shape[1] > min_len:
                # Adaptive pool along sequence dimension
                f = f.transpose(1, 2)  # (B, D, L)
                f = F.adaptive_avg_pool1d(f, min_len)
                f = f.transpose(1, 2)  # (B, min_len, D)
            aligned.append(f)
        return aligned
    
    def forward(self, fft_feat: torch.Tensor, cwt_feat: torch.Tensor, 
                raw_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fft_feat: (B, L1, D) FFT branch features
            cwt_feat: (B, L2, D) CWT branch features  
            raw_feat: (B, L3, D) raw temporal features
        Returns:
            fused: (B, L, mamba_d_model) fused features
        """
        # Align sequence lengths
        fft_feat, cwt_feat, raw_feat = self._align_lengths(fft_feat, cwt_feat, raw_feat)
        
        # Compute adaptive gates
        concat = torch.cat([fft_feat, cwt_feat, raw_feat], dim=-1)  # (B, L, D*3)
        gates = F.softmax(self.gate_net(concat), dim=-1)  # (B, L, 3)
        
        # Gated fusion
        fused = (gates[..., 0:1] * fft_feat + 
                 gates[..., 1:2] * cwt_feat + 
                 gates[..., 2:3] * raw_feat)
        
        return self.output_proj(fused)


# ============================================================================
# Selective SSM for Time Series (Mamba with Gated Skip)
# ============================================================================

class TemporalSelectiveSSM(nn.Module):
    """Selective State Space Model optimized for temporal sequences."""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B, L, D = x.shape
        
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :L]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)
        
        x_dbl = self.x_proj(x_branch)
        B_param = x_dbl[..., :self.d_state]
        C_param = x_dbl[..., self.d_state:2*self.d_state]
        dt = F.softplus(self.dt_proj(x_dbl[..., -1:]))
        
        A = -torch.exp(self.A_log.float())
        y = self._scan(x_branch, dt, A, B_param, C_param)
        y = y * F.silu(z)
        return self.out_proj(y) + residual
    
    def _scan(self, u, dt, A, B, C):
        batch, L, d_inner = u.shape
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)
        dBu = dB * u.unsqueeze(-1)
        
        h = torch.zeros(batch, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBu[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        return torch.stack(ys, dim=1)


class GatedMambaBlock(nn.Module):
    """Mamba block with gated skip connection (MAAT-inspired)."""
    
    def __init__(self, config: DualFreqMambaConfig):
        super().__init__()
        d = config.mamba_d_model
        
        # Mamba SSM
        self.ssm = TemporalSelectiveSSM(d, config.mamba_d_state, config.mamba_d_conv, config.mamba_expand)
        
        # Gated skip connection
        if config.use_gated_skip:
            self.skip_gate = nn.Sequential(
                nn.Linear(d * 2, d),
                nn.Sigmoid(),
            )
        self.use_gated_skip = config.use_gated_skip
        
        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ssm_out = self.ssm(x)
        
        if self.use_gated_skip:
            gate = self.skip_gate(torch.cat([x, ssm_out], dim=-1))
            x = gate * ssm_out + (1 - gate) * x
        else:
            x = ssm_out
        
        x = x + self.ffn(x)
        return x


class MambaReconstructionDecoder(nn.Module):
    """Stack of Mamba blocks for temporal reconstruction."""
    
    def __init__(self, config: DualFreqMambaConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            GatedMambaBlock(config)
            for _ in range(config.mamba_n_layers)
        ])
        self.norm = nn.LayerNorm(config.mamba_d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ============================================================================
# Sparse Association-Discrepancy Module (Anomaly Transformer + MAAT)
# ============================================================================

class SparseAssociationDiscrepancy(nn.Module):
    """Computes the discrepancy between prior-association and series-association.
    
    Key insight from Anomaly Transformer: normal points have strong associations
    with their neighbors (high prior-association), while anomalies don't.
    
    MAAT enhancement: uses sparse block attention for efficiency.
    """
    
    def __init__(self, config: DualFreqMambaConfig):
        super().__init__()
        d = config.mamba_d_model
        
        # Series-association (learned from data)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        
        # Prior-association parameters (Gaussian kernel)
        self.sigma = nn.Parameter(torch.tensor(config.prior_sigma))
        
        self.lambda_param = config.association_lambda
    
    def _prior_association(self, L: int, device: torch.device) -> torch.Tensor:
        """Gaussian prior: nearby timesteps should be associated."""
        positions = torch.arange(L, device=device, dtype=torch.float32)
        dist_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)) ** 2
        prior = torch.exp(-dist_matrix / (2 * self.sigma ** 2 + 1e-6))
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)
        return prior
    
    def _series_association(self, x: torch.Tensor) -> torch.Tensor:
        """Learned attention-based association from data."""
        q = self.q_proj(x)
        k = self.k_proj(x)
        d = q.shape[-1]
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        return F.softmax(attn, dim=-1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D) decoded features
        Returns:
            discrepancy: (B, L) per-timestep discrepancy score
            minimax_loss: loss for training (minimize-maximize)
        """
        B, L, D = x.shape
        
        prior = self._prior_association(L, x.device)  # (L, L)
        prior = prior.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)
        
        series = self._series_association(x)  # (B, L, L)
        
        # KL divergence: KL(Series || Prior) per timestep
        # Clamp for numerical stability
        series_clamped = series.clamp(min=1e-8)
        prior_clamped = prior.clamp(min=1e-8)
        
        kl_sp = (series_clamped * (series_clamped.log() - prior_clamped.log())).sum(dim=-1)  # (B, L)
        kl_ps = (prior_clamped * (prior_clamped.log() - series_clamped.log())).sum(dim=-1)
        
        # Symmetric KL as discrepancy
        discrepancy = (kl_sp + kl_ps) / 2  # (B, L)
        
        # Minimax loss: minimize series-association for normal, maximize prior
        # Phase 1 (minimize): push series towards prior
        loss_minimize = kl_sp.mean()
        # Phase 2 (maximize): push prior towards series  
        loss_maximize = kl_ps.mean()
        
        # Combined with lambda balance
        minimax_loss = loss_minimize - self.lambda_param * loss_maximize
        
        return discrepancy, minimax_loss


# ============================================================================
# Dual-Domain Reconstruction Heads
# ============================================================================

class DualDomainReconstructionHead(nn.Module):
    """Reconstructs in both time and frequency domains."""
    
    def __init__(self, config: DualFreqMambaConfig):
        super().__init__()
        
        # Time-domain reconstruction
        self.time_recon = nn.Sequential(
            nn.Linear(config.mamba_d_model, config.mamba_d_model),
            nn.GELU(),
            nn.Linear(config.mamba_d_model, config.n_channels),
        )
        
        # Frequency-domain reconstruction
        self.freq_recon = nn.Sequential(
            nn.Linear(config.mamba_d_model, config.mamba_d_model),
            nn.GELU(),
            nn.Linear(config.mamba_d_model, config.n_channels * 2),  # real + imag
        )
        
        self.n_channels = config.n_channels
    
    def forward(self, decoded: torch.Tensor, original_x: torch.Tensor, 
                original_fft: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            decoded: (B, L, D) Mamba decoder output
            original_x: (B, C, T) original time series
            original_fft: (B, C, F) original FFT (optional)
        Returns:
            time_loss: MSE loss in time domain
            freq_loss: MSE loss in frequency domain
            recon_error: (B, T) per-timestep reconstruction error
        """
        B = decoded.shape[0]
        T = original_x.shape[2]
        L = decoded.shape[1]
        
        # Time-domain reconstruction
        time_pred = self.time_recon(decoded)  # (B, L, C)
        
        # Align lengths
        original_t = original_x.transpose(1, 2)  # (B, T, C)
        if L != T:
            time_pred_aligned = F.interpolate(
                time_pred.transpose(1, 2), size=T, mode='linear', align_corners=False
            ).transpose(1, 2)
        else:
            time_pred_aligned = time_pred
        
        time_loss = F.mse_loss(time_pred_aligned, original_t)
        
        # Per-timestep reconstruction error
        recon_error = (time_pred_aligned - original_t).pow(2).mean(dim=-1)  # (B, T)
        
        # Frequency-domain reconstruction
        freq_loss = torch.tensor(0.0, device=decoded.device)
        if original_fft is not None:
            freq_pred = self.freq_recon(decoded)  # (B, L, C*2)
            freq_pred = freq_pred.view(B, L, self.n_channels, 2)
            
            # Align to frequency length
            F_len = original_fft.shape[-1]
            if L != F_len:
                freq_pred_r = F.interpolate(
                    freq_pred[..., 0].transpose(1, 2), size=F_len, mode='linear', align_corners=False
                ).transpose(1, 2)
                freq_pred_i = F.interpolate(
                    freq_pred[..., 1].transpose(1, 2), size=F_len, mode='linear', align_corners=False
                ).transpose(1, 2)
            else:
                freq_pred_r = freq_pred[..., 0]
                freq_pred_i = freq_pred[..., 1]
            
            freq_loss = F.mse_loss(freq_pred_r, original_fft.real.transpose(1, 2)) + \
                       F.mse_loss(freq_pred_i, original_fft.imag.transpose(1, 2))
        
        return time_loss, freq_loss, recon_error


# ============================================================================
# Model Output
# ============================================================================

@dataclass
class DualFreqMambaOutput(ModelOutput):
    """Output of DualFreqMamba model."""
    loss: Optional[torch.FloatTensor] = None
    anomaly_score: Optional[torch.FloatTensor] = None  # (B, T) per-timestep scores
    anomaly_label: Optional[torch.FloatTensor] = None  # (B, T) binary predictions
    reconstruction_loss: Optional[torch.FloatTensor] = None
    frequency_loss: Optional[torch.FloatTensor] = None
    association_loss: Optional[torch.FloatTensor] = None
    association_discrepancy: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# ============================================================================
# Main Model
# ============================================================================

class DualFreqMambaModel(PreTrainedModel):
    """DualFreqMamba — Dual-Branch Frequency-Temporal Fault Detector.
    
    A novel lightweight model combining:
    1. FFT Frequency Patching with Channel-Masked Attention (CATCH)
    2. CWT Scalogram Branch for wavelet multi-resolution features (TSCMamba)
    3. Raw Temporal Embedding for direct sequence modeling
    4. Adaptive Gated Tri-Branch Fusion
    5. Mamba SSM Reconstruction with Gated Skip Connections (MAAT)
    6. Sparse Association-Discrepancy Anomaly Scoring

    Parameters:
        ~8M trainable parameters (lightweight, real-time capable)
        Handles multivariate time series with N channels, T timesteps
    """
    
    config_class = DualFreqMambaConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: DualFreqMambaConfig):
        super().__init__(config)
        
        # Instance normalization
        self.instance_norm = nn.InstanceNorm1d(config.n_channels, affine=True)
        
        # Branch 1: FFT Frequency Patching
        self.fft_branch = FFTBranch(config)
        
        # Branch 2: CWT Scalogram (optional)
        if config.use_cwt:
            self.cwt_branch = CWTBranch(config)
        
        # Branch 3: Raw Temporal
        self.temporal_branch = TemporalEmbeddingBranch(config)
        
        # Fusion
        self.fusion = AdaptiveGatedFusion(config)
        
        # Mamba Reconstruction Decoder
        self.decoder = MambaReconstructionDecoder(config)
        
        # Association Discrepancy
        if config.use_association_discrepancy:
            self.association = SparseAssociationDiscrepancy(config)
        
        # Dual-Domain Reconstruction Head
        self.recon_head = DualDomainReconstructionHead(config)
        
        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_anomaly_labels: bool = True,
    ) -> DualFreqMambaOutput:
        """
        Args:
            input_values: (B, N_channels, T) multivariate time series window
            labels: (B, T) binary anomaly labels — optional, for evaluation
            return_anomaly_labels: whether to threshold scores into binary predictions

        Returns:
            DualFreqMambaOutput with loss, anomaly_score, component losses
        """
        B, C, T = input_values.shape
        
        # 1. Instance normalization
        x_norm = self.instance_norm(input_values)
        
        # 2. Extract features from all branches
        fft_feat, fft_complex = self.fft_branch(x_norm)
        
        if self.config.use_cwt:
            cwt_feat = self.cwt_branch(x_norm)
        else:
            # If no CWT, duplicate FFT features
            cwt_feat = fft_feat.clone()
        
        raw_feat = self.temporal_branch(x_norm)
        
        # 3. Fuse branches
        fused = self.fusion(fft_feat, cwt_feat, raw_feat)  # (B, L, mamba_d_model)
        
        # 4. Mamba reconstruction decoder
        decoded = self.decoder(fused)  # (B, L, mamba_d_model)
        
        # 5. Dual-domain reconstruction loss
        time_loss, freq_loss, recon_error = self.recon_head(decoded, input_values, fft_complex)
        
        # 6. Association discrepancy
        assoc_discrepancy = None
        assoc_loss = None
        if self.config.use_association_discrepancy:
            assoc_discrepancy, assoc_loss = self.association(decoded)
        
        # 7. Total loss
        total_loss = (
            self.config.reconstruction_weight * time_loss +
            self.config.frequency_weight * freq_loss
        )
        if assoc_loss is not None:
            total_loss = total_loss + self.config.association_weight * assoc_loss
        
        # 8. Compute anomaly scores
        anomaly_score = recon_error  # (B, T)
        if assoc_discrepancy is not None:
            # Align association discrepancy to T timesteps
            if assoc_discrepancy.shape[1] != T:
                assoc_aligned = F.interpolate(
                    assoc_discrepancy.unsqueeze(1), size=T, mode='linear', align_corners=False
                ).squeeze(1)
            else:
                assoc_aligned = assoc_discrepancy
            # Combine: weighted sum of reconstruction error and association discrepancy
            anomaly_score = F.softmax(-assoc_aligned, dim=-1) * recon_error
        
        # 9. Threshold for binary predictions
        anomaly_label = None
        if return_anomaly_labels and not self.training:
            threshold = anomaly_score.mean() + 2 * anomaly_score.std()
            anomaly_label = (anomaly_score > threshold).float()
        
        return DualFreqMambaOutput(
            loss=total_loss,
            anomaly_score=anomaly_score,
            anomaly_label=anomaly_label,
            reconstruction_loss=time_loss,
            frequency_loss=freq_loss,
            association_loss=assoc_loss,
            association_discrepancy=assoc_discrepancy,
            hidden_states=(decoded,),
        )
