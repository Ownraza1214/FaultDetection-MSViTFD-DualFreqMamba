"""Test both fault detection models with synthetic data (CPU-optimized)."""
import sys
sys.path.insert(0, '/app')

import torch
import torch.nn as nn
import time

print("=" * 70)
print("TESTING MSViTFD — Multi-Scale Vision Transformer Fault Detector")
print("=" * 70)

from msviTfd.configuration_msviTfd import MSViTFDConfig
from msviTfd.modeling_msviTfd import MSViTFDModel

# Create config — smaller for CPU testing
config = MSViTFDConfig(
    encoder_name="efficientnet_b2",
    encoder_pretrained=True,
    freeze_encoder=True,
    hidden_dim=64,
    decoder_depths=[1, 1, 1],
    mamba_d_state=8,
    mamba_expand=1,
    dwconv_kernels=[3, 5],
    use_hilbert_scan=True,
    n_scan_directions=2,
    use_discriminator=True,
    discriminator_hidden=128,
    noise_std=0.015,
    use_fbft=True,
    fbft_hidden=64,
    image_size=64,
)

print(f"\n[Config] Model type: {config.model_type}")
print(f"[Config] Encoder: {config.encoder_name}")
print(f"[Config] Feature dims: {config.feature_dims}")
print(f"[Config] Decoder depths: {config.decoder_depths}")
print(f"[Config] Hidden dim: {config.hidden_dim}")

# Create model
model = MSViTFDModel(config)
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[Model] Total parameters: {total_params:,}")
print(f"[Model] Trainable parameters: {trainable_params:,}")
print(f"[Model] Frozen parameters: {total_params - trainable_params:,}")

# Test forward pass with small input
batch = torch.randn(1, 3, 64, 64)
print(f"\n[Test] Input shape: {batch.shape}")

t0 = time.time()
with torch.no_grad():
    output = model(batch, return_anomaly_map=True)
t1 = time.time()

print(f"[Test] Forward time: {t1-t0:.2f}s")
print(f"[Test] Loss: {output.loss.item():.4f}")
print(f"[Test] Reconstruction loss: {output.reconstruction_loss.item():.4f}")
if output.discriminator_loss is not None:
    print(f"[Test] Discriminator loss: {output.discriminator_loss.item():.4f}")
if output.fbft_loss is not None:
    print(f"[Test] FBFT loss: {output.fbft_loss.item():.4f}")
if output.anomaly_map is not None:
    print(f"[Test] Anomaly map shape: {output.anomaly_map.shape}")
if output.anomaly_score is not None:
    print(f"[Test] Anomaly scores: {output.anomaly_score.tolist()}")
print(f"[Test] Hidden states: {len(output.hidden_states)} scales")
for i, hs in enumerate(output.hidden_states):
    print(f"  Scale {i}: {hs.shape}")

# Test training mode
model.train()
output_train = model(batch, return_anomaly_map=False)
print(f"\n[Train] Loss: {output_train.loss.item():.4f}")
print(f"[Train] Loss requires grad: {output_train.loss.requires_grad}")

# Quick backward pass test
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
output_train.loss.backward()
optimizer.step()
print("[Train] Backward pass ✓")
print("[Train] Optimizer step ✓")

# Also count full-size model params
full_config = MSViTFDConfig(
    encoder_name="efficientnet_b2",
    hidden_dim=256,
    decoder_depths=[2, 3, 2],
    mamba_d_state=16,
    mamba_expand=2,
)
full_model = MSViTFDModel(full_config)
full_total = sum(p.numel() for p in full_model.parameters())
full_train = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
print(f"\n[Full-Size Model] Total: {full_total:,}, Trainable: {full_train:,}")
del full_model

print("\n✅ MSViTFD test PASSED!\n")


print("=" * 70)
print("TESTING DualFreqMamba — Dual-Branch Frequency-Temporal Fault Detector")
print("=" * 70)

from dualfreqmamba.configuration_dualfreqmamba import DualFreqMambaConfig
from dualfreqmamba.modeling_dualfreqmamba import DualFreqMambaModel

# Create config — smaller CWT for speed
ts_config = DualFreqMambaConfig(
    n_channels=10,
    window_size=50,
    fft_patch_size=4,
    fft_stride=2,
    fft_d_model=64,
    fft_n_heads=4,
    fft_n_layers=1,
    use_channel_mask=True,
    use_cwt=True,
    cwt_scales=8,
    cwt_d_model=32,
    mamba_d_model=64,
    mamba_d_state=8,
    mamba_n_layers=2,
    use_gated_skip=True,
    use_association_discrepancy=True,
    association_lambda=3.0,
    fusion_d_model=64,
)

print(f"\n[Config] Model type: {ts_config.model_type}")
print(f"[Config] Channels: {ts_config.n_channels}, Window: {ts_config.window_size}")
print(f"[Config] FFT: patch={ts_config.fft_patch_size}, stride={ts_config.fft_stride}")
print(f"[Config] CWT: scales={ts_config.cwt_scales}")
print(f"[Config] Mamba: d_model={ts_config.mamba_d_model}, layers={ts_config.mamba_n_layers}")

# Create model
ts_model = DualFreqMambaModel(ts_config)
ts_model.eval()

# Count parameters
ts_total = sum(p.numel() for p in ts_model.parameters())
ts_trainable = sum(p.numel() for p in ts_model.parameters() if p.requires_grad)
print(f"\n[Model] Total parameters: {ts_total:,}")
print(f"[Model] Trainable parameters: {ts_trainable:,}")

# Test forward pass
ts_batch = torch.randn(2, 10, 50)
print(f"\n[Test] Input shape: {ts_batch.shape}")

t0 = time.time()
with torch.no_grad():
    ts_output = ts_model(ts_batch, return_anomaly_labels=True)
t1 = time.time()

print(f"[Test] Forward time: {t1-t0:.2f}s")
print(f"[Test] Loss: {ts_output.loss.item():.4f}")
print(f"[Test] Reconstruction loss: {ts_output.reconstruction_loss.item():.4f}")
if ts_output.frequency_loss is not None:
    print(f"[Test] Frequency loss: {ts_output.frequency_loss.item():.4f}")
if ts_output.association_loss is not None:
    print(f"[Test] Association loss: {ts_output.association_loss.item():.4f}")
print(f"[Test] Anomaly score shape: {ts_output.anomaly_score.shape}")
print(f"[Test] Anomaly score stats: mean={ts_output.anomaly_score.mean():.6f}, "
      f"max={ts_output.anomaly_score.max():.6f}")
if ts_output.anomaly_label is not None:
    print(f"[Test] Anomaly label shape: {ts_output.anomaly_label.shape}")
    n_detected = ts_output.anomaly_label.sum().item()
    print(f"[Test] Detected anomalies: {n_detected:.0f}/{ts_output.anomaly_label.numel()}")
if ts_output.association_discrepancy is not None:
    print(f"[Test] Association discrepancy shape: {ts_output.association_discrepancy.shape}")

# Test training mode
ts_model.train()
ts_train_output = ts_model(ts_batch)
print(f"\n[Train] Loss: {ts_train_output.loss.item():.4f}")
print(f"[Train] Loss requires grad: {ts_train_output.loss.requires_grad}")

# Backward pass
ts_optimizer = torch.optim.Adam(ts_model.parameters(), lr=1e-3)
ts_optimizer.zero_grad()
ts_train_output.loss.backward()
ts_optimizer.step()
print("[Train] Backward pass ✓")
print("[Train] Optimizer step ✓")

# Test flexibility with different configs (no CWT for speed)
print("\n[Flex Test] Testing different input sizes...")
for n_ch, win in [(5, 30), (25, 100)]:
    flex_config = DualFreqMambaConfig(
        n_channels=n_ch, window_size=win, use_cwt=False,
        fft_d_model=32, mamba_d_model=32, fusion_d_model=32,
        mamba_n_layers=1, fft_n_layers=1, mamba_d_state=4
    )
    flex_model = DualFreqMambaModel(flex_config)
    flex_model.eval()
    x = torch.randn(1, n_ch, win)
    with torch.no_grad():
        out = flex_model(x)
    print(f"  channels={n_ch:3d}, window={win:3d} → loss={out.loss.item():.4f}, "
          f"score_shape={out.anomaly_score.shape}")

# Full-size model params
full_ts_config = DualFreqMambaConfig(
    n_channels=25, window_size=100,
    fft_d_model=128, fft_n_layers=2,
    cwt_scales=32, cwt_d_model=64,
    mamba_d_model=128, mamba_n_layers=3,
    fusion_d_model=128,
)
full_ts_model = DualFreqMambaModel(full_ts_config)
full_ts_total = sum(p.numel() for p in full_ts_model.parameters())
full_ts_train = sum(p.numel() for p in full_ts_model.parameters() if p.requires_grad)
print(f"\n[Full-Size Model] Total: {full_ts_total:,}, Trainable: {full_ts_train:,}")
del full_ts_model

print("\n✅ DualFreqMamba test PASSED!\n")

print("=" * 70)
print("BOTH MODELS TESTED SUCCESSFULLY!")
print("=" * 70)
print(f"\n📊 MSViTFD (full):       {full_train:>10,} trainable params")
print(f"📊 DualFreqMamba (full): {full_ts_train:>10,} trainable params")
