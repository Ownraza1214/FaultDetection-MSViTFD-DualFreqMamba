"""
MSViTFD — Multi-Scale Vision Transformer Fault Detector
=========================================================

A novel lightweight image fault/anomaly detection model combining:
- EfficientNet-B2 multi-scale feature extraction (frozen or trainable)
- Hilbert Space-Filling Curve scanning for 2D→1D linearization (MambaAD-inspired)
- Multi-Scale Locality-Enhanced State Space (LSS) decoder with multi-kernel DWConv
- Feature-Space Gaussian Noise Discriminator (SimpleNet-inspired)
- Bidirectional Forward-Backward Feature Transfer auxiliary loss (L2BT-inspired)

References:
- MambaAD (arXiv:2404.06564) — Hilbert scanning + Mamba decoder
- SimpleNet (arXiv:2303.15140, CVPR 2023) — Feature-space anomaly generation
- L2BT (arXiv:2407.04092) — Bidirectional MLP transfer
"""

from transformers import PretrainedConfig


class MSViTFDConfig(PretrainedConfig):
    """Configuration for MSViTFD — Multi-Scale Vision Transformer Fault Detector."""

    model_type = "msviTfd"

    def __init__(
        self,
        # Encoder
        encoder_name: str = "efficientnet_b2",
        encoder_pretrained: bool = True,
        freeze_encoder: bool = True,
        encoder_layer_indices: list = None,  # which feature levels to extract

        # Feature dimensions from encoder (auto-set based on encoder_name)
        feature_dims: list = None,  # e.g. [48, 120, 352] for efficientnet_b2

        # Decoder (Mamba-style reconstruction)
        hidden_dim: int = 256,
        decoder_depths: list = None,  # Mamba blocks per scale
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,

        # Multi-kernel DWConv for local features
        dwconv_kernels: list = None,  # parallel depthwise conv kernels

        # Hilbert Scanning
        use_hilbert_scan: bool = True,
        n_scan_directions: int = 4,  # 4 or 8

        # Discriminator head (SimpleNet-style)
        use_discriminator: bool = True,
        discriminator_hidden: int = 512,
        noise_std: float = 0.015,
        truncated_loss_margin: float = 0.5,

        # Bidirectional Feature Transfer (L2BT)
        use_fbft: bool = True,
        fbft_hidden: int = 256,

        # Input
        image_size: int = 256,
        in_channels: int = 3,

        # Training
        reconstruction_weight: float = 1.0,
        discriminator_weight: float = 0.5,
        fbft_weight: float = 0.3,

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.encoder_name = encoder_name
        self.encoder_pretrained = encoder_pretrained
        self.freeze_encoder = freeze_encoder
        self.encoder_layer_indices = encoder_layer_indices or [2, 3, 4]

        # Default feature dims for common encoders
        if feature_dims is None:
            encoder_feature_map = {
                "efficientnet_b0": [40, 112, 320],
                "efficientnet_b1": [40, 112, 320],
                "efficientnet_b2": [48, 120, 352],
                "efficientnet_b3": [48, 136, 384],
                "efficientnet_b4": [56, 160, 448],
                "resnet18": [128, 256, 512],
                "resnet34": [128, 256, 512],
                "resnet50": [512, 1024, 2048],
                "mobilenetv3_small_100": [24, 40, 96],
                "mobilenetv3_large_100": [40, 112, 160],
            }
            self.feature_dims = encoder_feature_map.get(encoder_name, [48, 120, 352])
        else:
            self.feature_dims = feature_dims

        self.hidden_dim = hidden_dim
        self.decoder_depths = decoder_depths or [2, 3, 2]
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.dwconv_kernels = dwconv_kernels or [3, 5, 7]
        self.use_hilbert_scan = use_hilbert_scan
        self.n_scan_directions = n_scan_directions

        self.use_discriminator = use_discriminator
        self.discriminator_hidden = discriminator_hidden
        self.noise_std = noise_std
        self.truncated_loss_margin = truncated_loss_margin

        self.use_fbft = use_fbft
        self.fbft_hidden = fbft_hidden

        self.image_size = image_size
        self.in_channels = in_channels

        self.reconstruction_weight = reconstruction_weight
        self.discriminator_weight = discriminator_weight
        self.fbft_weight = fbft_weight
