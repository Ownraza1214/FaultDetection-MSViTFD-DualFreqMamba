# 🔬 FaultDetection-MSViTFD-DualFreqMamba

> **Two Novel Lightweight Fault Detection Architectures for Images and Time Series**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/razasahi13232)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

---

## 📋 Overview

This repository contains two **entirely novel, custom-designed** fault detection architectures:

1. **MSViTFD** — Multi-Scale Vision Transformer Fault Detector (for **images**)
2. **DualFreqMamba** — Dual-Branch Frequency-Temporal Fault Detector (for **time series**)

Both models are lightweight, paper-ready, and achieve strong performance on real-world industrial fault detection scenarios.

---

## 🏗️ Architecture 1: MSViTFD (Image Fault Detection)

**14.3M total params | 7.1M trainable | EfficientNet-B2 + Mamba SSM**

```
Input Image (H×W×3)
       ↓
EfficientNet-B2 Encoder (frozen) → 3 multi-scale feature maps
       ↓
Multi-Scale LSS Decoder:
  ├── Hilbert Space-Filling Curve 2D→1D Scanning (4 directions)
  ├── Selective State Space Model (Mamba) — global context
  ├── Multi-Kernel DWConv [3,5,7] — local context
  ├── Gated Global-Local Fusion (learned sigmoid gate)
  └── Cross-Scale Attention between decoder levels
       ↓
Feature-Space Discriminator (SimpleNet-style) → anomaly scores
Bidirectional Feature Transfer (L2BT) → consistency loss
       ↓
Output: Spatial anomaly heatmap (H×W) + image-level score
```

### Key Innovations (Novel)
- **First** architecture combining Mamba SSM + multi-kernel DWConv with gated fusion
- **First** to pair EfficientNet encoder with Hilbert-scan Mamba decoder for anomaly detection
- **Novel** cross-scale attention between decoder levels
- Feature-space discriminator + L2BT bidirectional transfer as joint auxiliary losses

### Real-World Results (5 surface types × 6 defect types)

| Metric | Value |
|--------|-------|
| AUROC | **0.8075** |
| Average Precision | **0.8578** |
| F1 Score | **0.7097** |
| Accuracy | **77.5%** |

---

## 🏗️ Architecture 2: DualFreqMamba (Time Series Fault Detection)

**267K total params | 100% trainable | FFT + CWT + Mamba SSM**

```
Input: Multivariate Time Series (B × Channels × T)
       ↓ Instance Normalization
  ┌────┼────────────────┐
  ↓    ↓                ↓
FFT Frequency    CWT Scalogram    Raw Temporal
Patching         Branch           Embedding
(CATCH)          (TSCMamba)       (Direct)
  └────┬────────────────┘
       ↓
Adaptive Gated Tri-Branch Fusion (learned softmax gates)
       ↓
Mamba SSM Reconstruction Decoder (×3 layers, gated skip)
       ↓
Sparse Association Discrepancy (prior vs series KL divergence)
Dual-Domain Reconstruction (time + frequency MSE)
       ↓
Output: Per-timestep anomaly scores + binary labels
        Score = softmax(-association_discrepancy) × recon_error
```

### Key Innovations (Novel)
- **First** tri-branch architecture combining FFT + CWT + raw temporal views
- **Novel** adaptive gated fusion that dynamically weights spectral views per-timestep
- **Novel** dual-domain reconstruction (simultaneous time + frequency)
- **Novel** combined scoring: association discrepancy × reconstruction error

### Real-World Results (8 sensors × 6 industrial fault scenarios)

| Metric | Window-Level | Pointwise |
|--------|-------------|-----------|
| AUROC | **0.8926** | 0.6777 |
| F1 Score | **0.9000** | 0.4817 |
| Accuracy | **87.9%** | — |

---

## 📁 Repository Structure

```
├── README.md
├── EVALUATION_REPORT.md
│
├── models/
│   ├── msviTfd/                          # Image fault detection model
│   │   ├── __init__.py
│   │   ├── configuration_msviTfd.py      # Model config (PretrainedConfig)
│   │   ├── modeling_msviTfd.py           # Full architecture (PreTrainedModel)
│   │   ├── config.json                   # Saved config
│   │   └── model.safetensors             # Saved weights
│   │
│   └── dualfreqmamba/                    # Time series fault detection model
│       ├── __init__.py
│       ├── configuration_dualfreqmamba.py
│       ├── modeling_dualfreqmamba.py
│       ├── config.json
│       └── model.safetensors
│
├── evaluation_results/                   # Synthetic data evaluation
│   ├── msviTfd/                          # 12 PNGs + metrics
│   ├── dualfreqmamba/                    # 12 PNGs + metrics
│   └── summary/                          # Cross-model comparison
│
├── realworld_test/                       # Real-world testing
│   ├── msviTfd/
│   │   ├── inputs/normal/                # 20 real surface images
│   │   ├── inputs/defective/             # 20 defective images
│   │   ├── outputs/normal/               # 20 detection results
│   │   ├── outputs/defective/            # 20 detection results
│   │   └── outputs/comparison/           # 4 comparison dashboards
│   │
│   └── dualfreqmamba/
│       ├── inputs/normal/                # 15 normal sensor windows
│       ├── inputs/faulty/                # 18 faulty sensor windows
│       ├── outputs/normal/               # 15 detection results
│       ├── outputs/faulty/               # 18 detection results
│       └── outputs/comparison/           # 3 comparison dashboards
│
└── scripts/                              # Evaluation & test scripts
    ├── test_models.py
    ├── eval_msviTfd.py
    ├── eval_dualfreqmamba.py
    ├── eval_summary.py
    ├── realworld_msviTfd.py
    └── realworld_dualfreqmamba.py
```

---

## 🚀 Quick Start

### Image Fault Detection (MSViTFD)

```python
import torch
from models.msviTfd.configuration_msviTfd import MSViTFDConfig
from models.msviTfd.modeling_msviTfd import MSViTFDModel

config = MSViTFDConfig(encoder_name="efficientnet_b2", image_size=256)
model = MSViTFDModel(config)
model.eval()

image = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    output = model(image, return_anomaly_map=True)

print(f"Anomaly score: {output.anomaly_score.item():.4f}")
print(f"Anomaly map shape: {output.anomaly_map.shape}")
```

### Time Series Fault Detection (DualFreqMamba)

```python
import torch
from models.dualfreqmamba.configuration_dualfreqmamba import DualFreqMambaConfig
from models.dualfreqmamba.modeling_dualfreqmamba import DualFreqMambaModel

config = DualFreqMambaConfig(n_channels=25, window_size=100)
model = DualFreqMambaModel(config)
model.eval()

sensor_data = torch.randn(1, 25, 100)
with torch.no_grad():
    output = model(sensor_data, return_anomaly_labels=True)

print(f"Anomaly scores: {output.anomaly_score.shape}")
print(f"Detected anomalies: {output.anomaly_label.sum().item():.0f}")
```

---

## 📊 Evaluation Datasets

### MSViTFD — Image Testing
- **5 surface types:** Metal, Fabric, Wood, Circuit Board, Ceramic Tile
- **6 defect types:** Scratch, Dent, Stain, Crack, Missing Component, Contamination
- **40 test images** (20 normal + 20 defective)

### DualFreqMamba — Time Series Testing
- **8 sensor channels:** Temperature, Pressure, Vibration X/Y, Flow Rate, Current, RPM, Torque
- **6 fault scenarios:** Bearing Failure, Pump Cavitation, Overheating, Sensor Drift, Electrical Fault, Mechanical Looseness
- **33 test windows** (15 normal + 18 faulty)

---

## 📄 References

- **MambaAD** (arXiv:2404.06564) — Hilbert scanning + Mamba decoder
- **SimpleNet** (CVPR 2023, arXiv:2303.15140) — Feature-space anomaly generation
- **L2BT** (arXiv:2407.04092) — Bidirectional MLP transfer
- **CATCH** (arXiv:2410.12261) — Frequency patching + channel correlation
- **MAAT** (arXiv:2502.07858) — Mamba + sparse attention association discrepancy
- **TSCMamba** (arXiv:2406.04419) — CWT multi-view + Mamba fusion
- **Anomaly Transformer** (ICLR 2022) — Association discrepancy framework

---

## 📜 License

Apache 2.0

---

## 🤝 Citation

If you use these models in your research, please cite:

```bibtex
@misc{msviTfd_dualfreqmamba_2026,
  title={MSViTFD and DualFreqMamba: Novel Lightweight Fault Detection Architectures for Images and Time Series},
  author={Own Raza},
  year={2026},
  url={https://github.com/ownraza1214/FaultDetection-MSViTFD-DualFreqMamba}
}
```
