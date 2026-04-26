# MSViTFD & DualFreqMamba: Novel Fault Detection Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-blue)](https://huggingface.co/razasahi13232)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Two novel lightweight fault/anomaly detection models designed for industrial deployment:

1. **MSViTFD** — Multi-Scale Vision Transformer Fault Detector (for images)
2. **DualFreqMamba** — Dual-Branch Frequency-Temporal Fault Detector (for time series)

---

## 🏗️ Architecture Overview

### MSViTFD (Image Anomaly Detection)

```
Input Image (256×256×3)
    │
    ▼
EfficientNet-B2 Encoder (frozen, ImageNet pretrained)
    │── F1 (H/4, 48ch) ── F2 (H/8, 120ch) ── F3 (H/16, 352ch)
    │
    ▼
Multi-Scale Locality-Enhanced State Space (LSS) Decoder
    │── Hilbert Space-Filling Curve 2D→1D scanning (4 directions)
    │── Selective State Space Model (Mamba SSM) — global context
    │── Multi-Kernel Depthwise Conv [3×, 5×, 7×] — local context
    │── Gated Global-Local Fusion
    │
    ▼
Feature-Space Discriminator (SimpleNet-style)
    │── Gaussian noise in feature space → synthetic anomalies
    │── Truncated L1 loss for anomaly scoring
    │
    ▼
Bidirectional Feature Transfer (L2BT)
    │── Forward MLP: low-level → high-level prediction
    │── Backward MLP: high-level → low-level prediction
    │── Cosine similarity consistency loss
    │
    ▼
Output: Anomaly Map (H×W) + Image-Level Score
```

**Key innovations:**
- First combination of Hilbert-curve Mamba SSM with multi-kernel depthwise convolution gated fusion
- Feature-space anomaly generation with discriminator scoring (SimpleNet-style)
- Bidirectional feature transfer for cross-scale consistency (L2BT-style)
- **14.3M params** (7.1M trainable, 7.2M frozen encoder)

**Novel components from:**
- [MambaAD](https://arxiv.org/abs/2404.06564) — Hilbert scanning + Mamba decoder concept
- [SimpleNet](https://arxiv.org/abs/2303.15140) (CVPR 2023) — Feature-space anomaly discrimination
- [L2BT](https://arxiv.org/abs/2407.04092) — Bidirectional feature transfer

### DualFreqMamba (Time-Series Anomaly Detection)

```
Input: X ∈ ℝ^{B × C × T} (multivariate time series)
    │
    ├─── FFT Frequency Patching Branch (CATCH-style)
    │    └── Channel-Masked Transformer layers
    │
    ├─── CWT Scalogram Branch (TSCMamba-style)
    │    └── Morlet wavelet → Conv2D patch embedding
    │
    └─── Raw Temporal Embedding Branch
         └── Linear projection + positional encoding
    │
    ▼
Adaptive Gated Tri-Branch Fusion
    │── Learned gate weights per timestep
    │
    ▼
Mamba SSM Reconstruction Decoder (3 layers)
    │── Gated skip connections (MAAT-style)
    │
    ▼
Sparse Association Discrepancy
    │── Prior: Gaussian kernel (expected neighbor associations)
    │── Series: Learned attention (actual data associations)
    │── KL divergence → anomaly signal
    │
    ▼
Dual-Domain Reconstruction
    │── Time-domain: MSE reconstruction loss
    │── Frequency-domain: FFT reconstruction loss
    │
    ▼
Output: Per-timestep anomaly scores + binary predictions
```

**Key innovations:**
- First tri-branch (FFT + CWT + temporal) fusion with adaptive gating
- Mamba SSM decoder with gated skip connections for temporal reconstruction
- Dual-domain (time + frequency) reconstruction scoring
- **~1.6M params** (all trainable, lightweight for edge deployment)

**Novel components from:**
- [CATCH](https://arxiv.org/abs/2410.12261) — FFT frequency patching
- [TSCMamba](https://arxiv.org/abs/2406.04419) — CWT + Mamba fusion
- [MAAT](https://arxiv.org/abs/2502.07858) — Mamba + sparse association
- [Anomaly Transformer](https://arxiv.org/abs/2110.02642) (ICLR 2022) — Association discrepancy

---

## 📊 Benchmark Results

### MSViTFD on MVTec AD

| Method | Venue | I-AUROC (%) | P-AUROC (%) | Params (M) |
|--------|-------|-------------|-------------|------------|
| SimpleNet | CVPR 2023 | 99.6 | 98.1 | 25.0 |
| PatchCore | CVPR 2022 | 99.1 | 98.1 | 43.0 |
| EfficientAD | WACV 2024 | 99.1 | 98.8 | — |
| Reverse Distillation | CVPR 2022 | 98.5 | 97.8 | 51.0 |
| PaDiM | ICPR 2021 | 95.3 | 97.5 | 23.5 |
| **MSViTFD (Ours)** | — | **TBD** | **TBD** | **14.3** |

### DualFreqMamba on Time-Series Benchmarks

| Method | Venue | SMD | MSL | SMAP | SWaT | PSM | Avg F1 (%) |
|--------|-------|-----|-----|------|------|-----|------------|
| Anomaly Transformer | ICLR 2022 | 92.33 | 93.59 | 96.69 | 94.07 | 97.89 | 94.91 |
| TimesNet | ICLR 2023 | 85.81 | 85.15 | 71.52 | 91.74 | 97.47 | 86.34 |
| OmniAnomaly | KDD 2019 | 85.22 | 87.67 | 86.92 | 82.83 | 80.83 | 84.69 |
| USAD | KDD 2020 | 82.00 | 86.00 | 87.00 | 81.70 | 90.00 | 85.34 |
| **DualFreqMamba (Ours)** | — | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

### Complexity Analysis

| Model | Params (M) | GFLOPs | Memory (MB) | Inference (ms) |
|-------|-----------|--------|-------------|----------------|
| MSViTFD | 14.3 (7.1 trainable) | 11.82 | 54.7 | ~2400 (CPU) |
| DualFreqMamba | 1.60 | 0.075 | 6.1 | ~33 (CPU) |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Ownraza1214/FaultDetection-MSViTFD-DualFreqMamba.git
cd FaultDetection-MSViTFD-DualFreqMamba
pip install -r requirements.txt
```

### Load Pre-trained Models

```python
# MSViTFD (Image Anomaly Detection)
from models.msviTfd import MSViTFDConfig, MSViTFDModel

config = MSViTFDConfig(hidden_dim=256, decoder_depths=[2, 3, 2])
model = MSViTFDModel(config)
# model = MSViTFDModel.from_pretrained("razasahi13232/MSViTFD-fault-detector")

import torch
output = model(pixel_values=torch.randn(1, 3, 256, 256))
print(f"Anomaly score: {output.anomaly_score}")
print(f"Anomaly map shape: {output.anomaly_map.shape}")
```

```python
# DualFreqMamba (Time-Series Anomaly Detection)
from models.dualfreqmamba import DualFreqMambaConfig, DualFreqMambaModel

config = DualFreqMambaConfig(n_channels=38, window_size=100, mamba_d_model=128)
model = DualFreqMambaModel(config)
# model = DualFreqMambaModel.from_pretrained("razasahi13232/DualFreqMamba-fault-detector")

import torch
output = model(input_values=torch.randn(1, 38, 100))
print(f"Anomaly scores: {output.anomaly_score.shape}")
print(f"Loss: {output.loss.item()}")
```

---

## 🏋️ Training

### MSViTFD on MVTec AD

```bash
# Download MVTec AD dataset
# wget https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/mvtec_anomaly_detection.tar.xz
# tar xf mvtec_anomaly_detection.tar.xz -C ./datasets/MVTec

# Train on all 15 categories (publication protocol)
python training/train_msviTfd.py \
    --data_root ./datasets/MVTec \
    --category all \
    --output_dir ./checkpoints/msviTfd \
    --epochs 200 \
    --batch_size 8 \
    --lr 2e-4 \
    --seed 42

# Train on a single category
python training/train_msviTfd.py \
    --data_root ./datasets/MVTec \
    --category bottle \
    --epochs 200 \
    --batch_size 8
```

### DualFreqMamba on Standard TS Benchmarks

```bash
# Datasets are automatically downloaded from HuggingFace Hub (thuml/Time-Series-Library)

# Train on SMD
python training/train_dualfreqmamba.py \
    --dataset SMD \
    --output_dir ./checkpoints/dualfreqmamba \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --seed 42

# Train on all benchmarks
python training/train_dualfreqmamba.py \
    --dataset all \
    --output_dir ./checkpoints/dualfreqmamba
```

### Training Hyperparameters (Publication Defaults)

#### MSViTFD
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning Rate | 2×10⁻⁴ |
| LR Schedule | Cosine Annealing (η_min=10⁻⁶) |
| Weight Decay | 10⁻⁴ |
| Batch Size | 8 |
| Epochs | 200 |
| Image Size | 256×256 |
| Encoder | EfficientNet-B2 (frozen) |
| Hidden Dim | 256 |
| Decoder Depths | [2, 3, 2] |
| Mamba d_state | 16 |
| DWConv Kernels | [3, 5, 7] |
| Noise σ | 0.015 |
| Loss: λ_recon | 1.0 |
| Loss: λ_disc | 0.5 |
| Loss: λ_fbft | 0.3 |
| Gradient Clip | 1.0 |
| Augmentation | RandomCrop, HFlip, VFlip, Rotation(±10°), ColorJitter |

#### DualFreqMamba
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Learning Rate | 10⁻⁴ |
| LR Schedule | Cosine Annealing (η_min=10⁻⁶) |
| Batch Size | 32 |
| Epochs | 10 (early stop patience=3) |
| Window Size | 100 |
| FFT d_model | 128 |
| FFT Patch Size | 4, Stride 2 |
| CWT Scales | 32 |
| Mamba d_model | 128 |
| Mamba Layers | 3 |
| Association λ | 3.0 |
| Loss: λ_recon | 1.0 |
| Loss: λ_freq | 0.5 |
| Loss: λ_assoc | 1.0 |
| Gradient Clip | 1.0 |

---

## 🔬 Ablation Studies

Remove each novel component to quantify its contribution:

```bash
# MSViTFD ablations
python training/run_ablations.py --model msviTfd --data_root ./datasets/MVTec --category bottle

# DualFreqMamba ablations  
python training/run_ablations.py --model dualfreqmamba --dataset SMD
```

### MSViTFD Ablation Configurations
| # | Configuration | What's Changed |
|---|--------------|----------------|
| 1 | Full Model | Baseline (all components) |
| 2 | w/o Hilbert Scan | Raster-only scanning |
| 3 | w/o Multi-Kernel DWConv | Single kernel [3] instead of [3,5,7] |
| 4 | w/o Discriminator | Remove SimpleNet head |
| 5 | w/o L2BT | Remove bidirectional transfer |
| 6 | Reconstruction Only | Remove both Disc + L2BT |
| 7 | Single Scan Direction | 1 instead of 4 scan directions |

### DualFreqMamba Ablation Configurations
| # | Configuration | What's Changed |
|---|--------------|----------------|
| 1 | Full Model | Baseline (all components) |
| 2 | w/o CWT Branch | Remove wavelet features |
| 3 | w/o Association Discrepancy | Reconstruction-only scoring |
| 4 | w/o Gated Skip | Plain Mamba without skip |
| 5 | w/o Freq Reconstruction | Time-domain only |
| 6 | w/o Channel Mask | Plain transformer in FFT branch |
| 7 | Concat Fusion | Replace adaptive gate with concatenation |
| 8 | Recon Scoring Only | Ignore association in scoring |

---

## 📏 Complexity Analysis

```bash
python training/complexity_analysis.py --output_dir ./complexity_results
```

---

## 🔁 Multi-Seed Reproducibility

Run with 3 seeds for mean ± std:

```bash
# MSViTFD (3 seeds × 15 categories × 200 epochs each)
python training/multi_seed_eval.py --model msviTfd --data_root ./datasets/MVTec --seeds 42 123 456

# DualFreqMamba (3 seeds × 5 datasets × 10 epochs each)
python training/multi_seed_eval.py --model dualfreqmamba --dataset all --seeds 42 123 456
```

---

## 📈 Baseline Comparisons

```bash
# Generate comparison tables with published results
python training/run_baselines.py --type tables --output_dir ./baseline_results

# For running actual baselines (requires anomalib):
pip install anomalib
anomalib train --model Patchcore --data MVTec --data.category=bottle
anomalib train --model Padim --data MVTec --data.category=bottle
anomalib train --model ReverseDistillation --data MVTec --data.category=bottle
```

---

## 📁 Repository Structure

```
FaultDetection-MSViTFD-DualFreqMamba/
├── README.md                       # This file
├── CITATION.bib                    # BibTeX citation
├── requirements.txt                # Python dependencies
├── models/
│   ├── msviTfd/
│   │   ├── __init__.py
│   │   ├── configuration_msviTfd.py    # MSViTFDConfig
│   │   ├── modeling_msviTfd.py         # MSViTFDModel
│   │   ├── config.json                 # Saved config
│   │   └── model.safetensors           # Saved weights
│   └── dualfreqmamba/
│       ├── __init__.py
│       ├── configuration_dualfreqmamba.py  # DualFreqMambaConfig
│       ├── modeling_dualfreqmamba.py       # DualFreqMambaModel
│       ├── config.json
│       └── model.safetensors
├── training/
│   ├── train_msviTfd.py                # Full MSViTFD training pipeline
│   ├── train_dualfreqmamba.py          # Full DualFreqMamba training pipeline
│   ├── run_ablations.py                # Ablation study runner
│   ├── run_baselines.py                # Baseline comparison tables
│   ├── complexity_analysis.py          # FLOPs, params, inference time
│   └── multi_seed_eval.py             # Multi-seed reproducibility
├── scripts/
│   ├── eval_msviTfd.py                 # Synthetic evaluation
│   ├── eval_dualfreqmamba.py           # Synthetic evaluation
│   ├── eval_summary.py                 # Cross-model comparison
│   ├── realworld_msviTfd.py            # Real-world image testing
│   ├── realworld_dualfreqmamba.py      # Real-world TS testing
│   └── test_models.py                  # Basic model tests
├── evaluation_results/
│   ├── msviTfd/                        # Synthetic eval PNGs
│   ├── dualfreqmamba/                  # Synthetic eval PNGs
│   └── summary/                        # Cross-model comparison
├── realworld_test/
│   ├── msviTfd/                        # Real-world image results
│   └── dualfreqmamba/                  # Real-world TS results
├── complexity_results/                 # Complexity analysis outputs
├── baseline_results/                   # Baseline comparison tables
└── ablation_results/                   # Ablation study outputs
```

---

## 📄 Datasets Used

### Image Anomaly Detection
- **MVTec AD** ([mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad)): 15 categories, 5,354 images, pixel-level ground truth masks
  - Also available on HuggingFace: [`Voxel51/mvtec-ad`](https://huggingface.co/datasets/Voxel51/mvtec-ad)

### Time-Series Anomaly Detection
All datasets from [`thuml/Time-Series-Library`](https://huggingface.co/datasets/thuml/Time-Series-Library):

| Dataset | Channels | Train Samples | Test Samples | Anomaly Ratio | Domain |
|---------|----------|---------------|--------------|---------------|--------|
| SMD | 38 | 708,405 | 708,420 | ~4.2% | Server machines |
| MSL | 55 | 58,317 | 73,729 | ~10.5% | NASA Mars rover |
| SMAP | 25 | 135,183 | 427,617 | ~12.8% | NASA satellite |
| SWaT | 51 | 496,800 | 449,919 | ~11.7% | Water treatment |
| PSM | 25 | 132,481 | 87,841 | ~27.8% | eBay server |

---

## 🔧 Evaluation Protocol

### Image (MVTec AD)
- **Protocol:** One model per category, trained on normal images only
- **Metrics:** Image-level AUROC (I-AUROC), Pixel-level AUROC (P-AUROC), Image F1, Average Precision
- **Preprocessing:** Resize 256→CenterCrop 256, ImageNet normalization

### Time Series
- **Protocol:** Point-adjust F1 (standard per Anomaly Transformer, ICLR 2022)
  - If any point in a contiguous anomaly segment is detected → entire segment credited
- **Metrics:** Precision, Recall, F1-Score (primary), AUROC (secondary)
- **Windowing:** Sliding window, size=100, stride=1
- **Threshold:** Best F1 threshold on test set

---

## 📝 Citation

If you use MSViTFD or DualFreqMamba in your research, please cite:

```bibtex
@article{raza2026msviTfd,
  title={{MSViTFD}: Multi-Scale Vision Transformer Fault Detector with 
         Hilbert Scanning and Feature-Space Discrimination},
  author={Raza, Own and [Co-authors]},
  journal={[Journal Name]},
  year={2026}
}

@article{raza2026dualfreqmamba,
  title={{DualFreqMamba}: Dual-Branch Frequency-Temporal Fault Detector 
         with Adaptive Gated Tri-Branch Fusion and Mamba Reconstruction},
  author={Raza, Own and [Co-authors]},
  journal={[Journal Name]},
  year={2026}
}
```

---

## 🤗 HuggingFace Models

- MSViTFD: [`razasahi13232/MSViTFD-fault-detector`](https://huggingface.co/razasahi13232/MSViTFD-fault-detector)
- DualFreqMamba: [`razasahi13232/DualFreqMamba-fault-detector`](https://huggingface.co/razasahi13232/DualFreqMamba-fault-detector)
- Evaluation Results: [`razasahi13232/fault-detection-evaluation-results`](https://huggingface.co/datasets/razasahi13232/fault-detection-evaluation-results)

---

## ⚖️ License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This work builds upon ideas from:
- [MambaAD](https://arxiv.org/abs/2404.06564) — Hilbert curve scanning
- [SimpleNet](https://arxiv.org/abs/2303.15140) — Feature-space discrimination
- [L2BT](https://arxiv.org/abs/2407.04092) — Bidirectional feature transfer
- [CATCH](https://arxiv.org/abs/2410.12261) — Frequency patching
- [TSCMamba](https://arxiv.org/abs/2406.04419) — CWT + Mamba
- [MAAT](https://arxiv.org/abs/2502.07858) — Mamba + association discrepancy
- [Anomaly Transformer](https://arxiv.org/abs/2110.02642) — Association discrepancy framework
- [timm](https://github.com/huggingface/pytorch-image-models) — Pretrained encoders
- [anomalib](https://github.com/open-edge-platform/anomalib) — Anomaly detection framework
- [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) — TS benchmarks
