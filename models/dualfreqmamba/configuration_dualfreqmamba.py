"""
DualFreqMamba — Dual-Branch Frequency-Temporal Fault Detector
==============================================================

A novel lightweight time series fault/anomaly detection model combining:
- FFT Frequency Patching with Channel-Aware Masked Attention (CATCH-inspired)
- CWT Scalogram Branch for wavelet-domain multi-resolution analysis (TSCMamba-inspired)
- Selective State Space Model (Mamba) reconstruction with Gated Skip (MAAT-inspired)
- Sparse Association-Discrepancy scoring for interpretable anomaly detection
- Adaptive Channel Fusion with bi-level mask optimization

References:
- CATCH (arXiv:2410.12261) — Frequency patching + channel correlation
- MAAT (arXiv:2502.07858) — Mamba + sparse attention association discrepancy
- TSCMamba (arXiv:2406.04419) — CWT multi-view + Mamba fusion
- Anomaly Transformer (ICLR 2022) — Association discrepancy framework
"""

from transformers import PretrainedConfig


class DualFreqMambaConfig(PretrainedConfig):
    """Configuration for DualFreqMamba — Dual-Branch Frequency-Temporal Fault Detector."""

    model_type = "dual_freq_mamba"

    def __init__(
        self,
        # Input
        n_channels: int = 25,
        window_size: int = 100,
        
        # FFT Branch (CATCH-style frequency patching)
        fft_patch_size: int = 4,
        fft_stride: int = 2,
        fft_d_model: int = 128,
        fft_n_heads: int = 4,
        fft_n_layers: int = 2,
        fft_dropout: float = 0.1,
        use_channel_mask: bool = True,

        # CWT Branch (TSCMamba-style wavelet features)
        use_cwt: bool = True,
        cwt_scales: int = 32,
        cwt_wavelet_sigma: float = 1.0,
        cwt_patch_size: int = 8,
        cwt_d_model: int = 64,

        # Mamba Reconstruction Decoder
        mamba_d_model: int = 128,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_n_layers: int = 3,
        use_gated_skip: bool = True,

        # Sparse Association Discrepancy (MAAT/Anomaly Transformer)
        use_association_discrepancy: bool = True,
        prior_sigma: float = 1.0,
        association_lambda: float = 3.0,
        sparse_block_size: int = 10,

        # Fusion
        fusion_d_model: int = 128,
        fusion_method: str = "adaptive_gate",  # "adaptive_gate", "concat", "attention"

        # Scoring
        scoring_method: str = "combined",  # "reconstruction", "association", "combined"
        anomaly_threshold: float = 0.5,

        # Training
        reconstruction_weight: float = 1.0,
        frequency_weight: float = 0.5,
        association_weight: float = 1.0,

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_channels = n_channels
        self.window_size = window_size

        self.fft_patch_size = fft_patch_size
        self.fft_stride = fft_stride
        self.fft_d_model = fft_d_model
        self.fft_n_heads = fft_n_heads
        self.fft_n_layers = fft_n_layers
        self.fft_dropout = fft_dropout
        self.use_channel_mask = use_channel_mask

        self.use_cwt = use_cwt
        self.cwt_scales = cwt_scales
        self.cwt_wavelet_sigma = cwt_wavelet_sigma
        self.cwt_patch_size = cwt_patch_size
        self.cwt_d_model = cwt_d_model

        self.mamba_d_model = mamba_d_model
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_n_layers = mamba_n_layers
        self.use_gated_skip = use_gated_skip

        self.use_association_discrepancy = use_association_discrepancy
        self.prior_sigma = prior_sigma
        self.association_lambda = association_lambda
        self.sparse_block_size = sparse_block_size

        self.fusion_d_model = fusion_d_model
        self.fusion_method = fusion_method

        self.scoring_method = scoring_method
        self.anomaly_threshold = anomaly_threshold

        self.reconstruction_weight = reconstruction_weight
        self.frequency_weight = frequency_weight
        self.association_weight = association_weight
