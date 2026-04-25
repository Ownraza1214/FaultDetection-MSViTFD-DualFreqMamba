"""
==========================================================================
DualFreqMamba — Comprehensive Evaluation Suite
Dual-Branch Frequency-Temporal Fault Detector
==========================================================================
Generates:
  01_architecture_overview.png
  02_synthetic_timeseries_samples.png
  03_anomaly_detection_timeline.png
  04_roc_curve.png
  05_precision_recall_curve.png
  06_confusion_matrix.png
  07_score_distributions.png
  08_training_convergence.png
  09_parameter_breakdown.png
  10_frequency_analysis.png
  11_per_anomaly_type_performance.png
  12_complete_dashboard.png
==========================================================================
"""
import sys, os
sys.path.insert(0, '/app')
os.environ['MPLBACKEND'] = 'Agg'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, accuracy_score,
    matthews_corrcoef, cohen_kappa_score, precision_score, recall_score
)
import json
import time

from dualfreqmamba.configuration_dualfreqmamba import DualFreqMambaConfig
from dualfreqmamba.modeling_dualfreqmamba import DualFreqMambaModel

OUT = "/app/evaluation_results/dualfreqmamba"
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# 1. Create model
# ============================================================================
print("=" * 70)
print("DualFreqMamba — COMPREHENSIVE EVALUATION")
print("=" * 70)

config = DualFreqMambaConfig(
    n_channels=10,
    window_size=100,
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
    fusion_method="adaptive_gate",
)

model = DualFreqMambaModel(config)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model loaded: {total_params:,} params ({trainable_params:,} trainable)")

# ============================================================================
# 2. Generate synthetic multivariate time series with faults
# ============================================================================
print("\n[1/12] Generating synthetic time series fault dataset...")

np.random.seed(42)
torch.manual_seed(42)

N_CHANNELS = 10
WINDOW_SIZE = 100
N_WINDOWS = 200

def generate_normal_window():
    """Generate normal multivariate time series window."""
    t = np.linspace(0, 4*np.pi, WINDOW_SIZE)
    data = np.zeros((N_CHANNELS, WINDOW_SIZE), dtype=np.float32)
    for c in range(N_CHANNELS):
        freq = 0.5 + c * 0.3
        phase = np.random.uniform(0, 2*np.pi)
        data[c] = (0.5 * np.sin(freq * t + phase) + 
                   0.2 * np.cos(2 * freq * t) +
                   0.05 * np.random.randn(WINDOW_SIZE))
    return data, np.zeros(WINDOW_SIZE, dtype=np.int32), 'normal'

def generate_fault_window(fault_type='spike'):
    """Generate time series window with injected faults."""
    data, labels, _ = generate_normal_window()
    labels = np.zeros(WINDOW_SIZE, dtype=np.int32)
    
    if fault_type == 'spike':
        # Point anomaly: sudden spike in 2-3 channels
        n_spikes = np.random.randint(2, 5)
        for _ in range(n_spikes):
            pos = np.random.randint(10, WINDOW_SIZE-10)
            channels = np.random.choice(N_CHANNELS, np.random.randint(1, 4), replace=False)
            for ch in channels:
                data[ch, pos] += np.random.choice([-1, 1]) * np.random.uniform(3, 6)
            labels[pos] = 1
            
    elif fault_type == 'drift':
        # Trend anomaly: gradual drift in subset of channels
        start = np.random.randint(20, 50)
        end = np.random.randint(start+20, min(start+50, WINDOW_SIZE))
        channels = np.random.choice(N_CHANNELS, np.random.randint(2, 5), replace=False)
        drift = np.linspace(0, np.random.uniform(2, 4), end-start)
        for ch in channels:
            data[ch, start:end] += drift
        labels[start:end] = 1
        
    elif fault_type == 'frequency_shift':
        # Frequency anomaly: sudden change in oscillation frequency
        start = np.random.randint(30, 60)
        end = min(start + 30, WINDOW_SIZE)
        t = np.linspace(0, 4*np.pi, WINDOW_SIZE)
        channels = np.random.choice(N_CHANNELS, np.random.randint(2, 4), replace=False)
        for ch in channels:
            data[ch, start:end] = 2.0 * np.sin(8 * t[start:end]) + 0.05 * np.random.randn(end-start)
        labels[start:end] = 1
        
    elif fault_type == 'dropout':
        # Sensor dropout: channels go to zero/constant
        start = np.random.randint(20, 50)
        end = np.random.randint(start+10, min(start+30, WINDOW_SIZE))
        channels = np.random.choice(N_CHANNELS, np.random.randint(1, 3), replace=False)
        for ch in channels:
            data[ch, start:end] = 0.0
        labels[start:end] = 1
        
    elif fault_type == 'noise_burst':
        # Noise anomaly: sudden increase in noise level
        start = np.random.randint(20, 60)
        end = min(start + 25, WINDOW_SIZE)
        channels = np.random.choice(N_CHANNELS, np.random.randint(3, 6), replace=False)
        for ch in channels:
            data[ch, start:end] += np.random.randn(end-start) * 2.0
        labels[start:end] = 1
    
    return data, labels, fault_type

fault_types = ['spike', 'drift', 'frequency_shift', 'dropout', 'noise_burst']
N_NORMAL = N_WINDOWS // 2
N_ANOMALY = N_WINDOWS // 2

windows = []
window_labels = []  # per-timestep labels
window_classes = []  # per-window: 0=normal, 1=anomaly
fault_type_labels = []

for i in range(N_NORMAL):
    data, labels, ft = generate_normal_window()
    windows.append(data)
    window_labels.append(labels)
    window_classes.append(0)
    fault_type_labels.append('normal')

for i in range(N_ANOMALY):
    ft = fault_types[i % len(fault_types)]
    data, labels, ft_actual = generate_fault_window(ft)
    windows.append(data)
    window_labels.append(labels)
    window_classes.append(1)
    fault_type_labels.append(ft_actual)

window_classes = np.array(window_classes)
all_pointwise_labels = np.array(window_labels)

print(f"  Windows: {N_NORMAL} normal + {N_ANOMALY} anomaly = {N_WINDOWS}")
print(f"  Channels: {N_CHANNELS}, Window size: {WINDOW_SIZE}")
print(f"  Fault types: {dict(zip(*np.unique([ft for ft in fault_type_labels if ft != 'normal'], return_counts=True)))}")
total_anomaly_points = sum(l.sum() for l in window_labels)
print(f"  Total anomaly timesteps: {total_anomaly_points}/{N_WINDOWS*WINDOW_SIZE} ({100*total_anomaly_points/(N_WINDOWS*WINDOW_SIZE):.1f}%)")

# ============================================================================
# 3. PNG 02: Synthetic Time Series Samples
# ============================================================================
print("[2/12] Generating synthetic time series samples visualization...")

fig, axes = plt.subplots(6, 2, figsize=(24, 20))
fig.suptitle('DualFreqMamba — Synthetic Time Series Fault Dataset', fontsize=20, fontweight='bold', y=1.01)

# Normal sample
ax = axes[0, 0]
sample = windows[0]
for ch in range(min(5, N_CHANNELS)):
    ax.plot(sample[ch], label=f'Ch {ch}', alpha=0.8, linewidth=1.2)
ax.set_title('Normal Sample (No Faults)', fontsize=13, fontweight='bold', color='green')
ax.legend(fontsize=8, ncol=5, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylabel('Amplitude')

# Normal FFT
ax = axes[0, 1]
fft_normal = np.abs(np.fft.rfft(windows[0], axis=1))
for ch in range(min(5, N_CHANNELS)):
    ax.plot(fft_normal[ch], label=f'Ch {ch}', alpha=0.8, linewidth=1.2)
ax.set_title('Normal — FFT Magnitude', fontsize=13, fontweight='bold', color='green')
ax.set_xlabel('Frequency Bin')
ax.grid(True, alpha=0.3)

# Each fault type
colors = {'spike': '#FF6B6B', 'drift': '#4ECDC4', 'frequency_shift': '#45B7D1', 
          'dropout': '#96CEB4', 'noise_burst': '#DDA0DD'}
for row, ft in enumerate(fault_types):
    idx = N_NORMAL + fault_type_labels[N_NORMAL:].index(ft)
    sample = windows[idx]
    labels = window_labels[idx]
    
    # Time domain
    ax = axes[row+1, 0]
    for ch in range(min(5, N_CHANNELS)):
        ax.plot(sample[ch], alpha=0.7, linewidth=1)
    # Highlight anomaly region
    anomaly_mask = labels == 1
    if anomaly_mask.any():
        start = np.where(anomaly_mask)[0][0]
        end = np.where(anomaly_mask)[0][-1]
        ax.axvspan(start, end, alpha=0.3, color='red', label='Fault Region')
    ax.set_title(f'Fault: {ft.replace("_", " ").title()}', fontsize=13, fontweight='bold', color='red')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Amplitude')
    
    # FFT
    ax = axes[row+1, 1]
    fft_data = np.abs(np.fft.rfft(sample, axis=1))
    for ch in range(min(5, N_CHANNELS)):
        ax.plot(fft_data[ch], alpha=0.7, linewidth=1)
    ax.set_title(f'{ft.replace("_", " ").title()} — FFT Magnitude', fontsize=13, fontweight='bold', color='red')
    ax.set_xlabel('Frequency Bin')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/02_synthetic_timeseries_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_synthetic_timeseries_samples.png")

# ============================================================================
# 4. Run inference
# ============================================================================
print("[3/12] Running inference on all windows...")
model.eval()

all_window_scores = []   # per-window max score
all_pointwise_scores = [] # per-timestep scores
all_recon_losses = []
all_freq_losses = []
all_assoc_losses = []

batch_size = 16
for i in range(0, N_WINDOWS, batch_size):
    batch_data = windows[i:i+batch_size]
    batch_tensor = torch.tensor(np.stack(batch_data))
    
    with torch.no_grad():
        output = model(batch_tensor, return_anomaly_labels=True)
    
    scores = output.anomaly_score.cpu().numpy()  # (B, T)
    all_pointwise_scores.extend(scores)
    all_window_scores.extend(scores.max(axis=1).tolist())
    
    all_recon_losses.append(output.reconstruction_loss.item())
    if output.frequency_loss is not None:
        all_freq_losses.append(output.frequency_loss.item())
    if output.association_loss is not None:
        all_assoc_losses.append(output.association_loss.item())

all_window_scores = np.array(all_window_scores)
all_pointwise_scores = np.array(all_pointwise_scores)

print(f"  Window scores: min={all_window_scores.min():.6f}, max={all_window_scores.max():.6f}")
print(f"  Pointwise scores shape: {all_pointwise_scores.shape}")

# ============================================================================
# 5. PNG 03: Anomaly Detection Timeline
# ============================================================================
print("[4/12] Generating anomaly detection timeline...")

fig, axes = plt.subplots(6, 1, figsize=(24, 20))
fig.suptitle('DualFreqMamba — Anomaly Detection Timeline', fontsize=20, fontweight='bold', y=1.01)

# Show 6 windows: 1 normal + 5 fault types
show_indices = [0]  # normal
for ft in fault_types:
    idx = N_NORMAL + fault_type_labels[N_NORMAL:].index(ft)
    show_indices.append(idx)

for ax_idx, widx in enumerate(show_indices):
    ax = axes[ax_idx]
    data = windows[widx]
    gt_labels = window_labels[widx]
    pred_scores = all_pointwise_scores[widx]
    
    # Normalize pred_scores for visualization
    ps_norm = (pred_scores - pred_scores.min()) / (pred_scores.max() - pred_scores.min() + 1e-10)
    
    # Plot channels (background)
    t = np.arange(WINDOW_SIZE)
    for ch in range(min(3, N_CHANNELS)):
        ax.plot(t, data[ch] * 0.3 + ch * 0.5, alpha=0.4, linewidth=0.8, color='gray')
    
    # Plot anomaly score
    ax2 = ax.twinx()
    ax2.fill_between(t, 0, ps_norm, alpha=0.4, color='orange', label='Anomaly Score')
    ax2.set_ylabel('Score', color='orange', fontsize=10)
    ax2.set_ylim(0, 1.5)
    
    # Ground truth highlighting
    if gt_labels.sum() > 0:
        anomaly_regions = np.where(gt_labels == 1)[0]
        if len(anomaly_regions) > 0:
            start = anomaly_regions[0]
            end = anomaly_regions[-1]
            ax.axvspan(start, end, alpha=0.2, color='red', label='Ground Truth Fault')
    
    ft_name = fault_type_labels[widx]
    color = 'green' if ft_name == 'normal' else 'red'
    ax.set_title(f'{ft_name.replace("_", " ").title()} — Window #{widx}', 
                fontsize=13, fontweight='bold', color=color)
    ax.set_ylabel('Signal', fontsize=10)
    ax.grid(True, alpha=0.2)
    if ax_idx == len(show_indices)-1:
        ax.set_xlabel('Timestep', fontsize=12)

plt.tight_layout()
plt.savefig(f'{OUT}/03_anomaly_detection_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_anomaly_detection_timeline.png")

# ============================================================================
# 6. Compute metrics — both window-level and pointwise
# ============================================================================
print("[5/12] Computing evaluation metrics...")

# === Window-level metrics ===
wscores_norm = (all_window_scores - all_window_scores.min()) / (all_window_scores.max() - all_window_scores.min() + 1e-10)
fpr_w, tpr_w, thresh_w = roc_curve(window_classes, wscores_norm)
roc_auc_w = auc(fpr_w, tpr_w)
prec_w, rec_w, _ = precision_recall_curve(window_classes, wscores_norm)
ap_w = average_precision_score(window_classes, wscores_norm)

j_w = tpr_w - fpr_w
opt_w = np.argmax(j_w)
preds_w = (wscores_norm >= thresh_w[opt_w]).astype(int)
acc_w = accuracy_score(window_classes, preds_w)
f1_w = f1_score(window_classes, preds_w)
mcc_w = matthews_corrcoef(window_classes, preds_w)
kappa_w = cohen_kappa_score(window_classes, preds_w)

# === Pointwise metrics ===
flat_gt = all_pointwise_labels.flatten()
flat_pred_scores = all_pointwise_scores.flatten()
flat_norm = (flat_pred_scores - flat_pred_scores.min()) / (flat_pred_scores.max() - flat_pred_scores.min() + 1e-10)

fpr_p, tpr_p, thresh_p = roc_curve(flat_gt, flat_norm)
roc_auc_p = auc(fpr_p, tpr_p)
prec_p, rec_p, _ = precision_recall_curve(flat_gt, flat_norm)
ap_p = average_precision_score(flat_gt, flat_norm)

j_p = tpr_p - fpr_p
opt_p = np.argmax(j_p)
preds_p = (flat_norm >= thresh_p[opt_p]).astype(int)
acc_p = accuracy_score(flat_gt, preds_p)
f1_p = f1_score(flat_gt, preds_p, zero_division=0)
prec_p_val = precision_score(flat_gt, preds_p, zero_division=0)
rec_p_val = recall_score(flat_gt, preds_p, zero_division=0)

metrics = {
    "model": "DualFreqMamba",
    "total_params": total_params,
    "trainable_params": trainable_params,
    "n_windows": N_WINDOWS,
    "n_channels": N_CHANNELS,
    "window_size": WINDOW_SIZE,
    "window_level": {
        "AUROC": float(roc_auc_w),
        "AP": float(ap_w),
        "Accuracy": float(acc_w),
        "F1": float(f1_w),
        "MCC": float(mcc_w),
        "Kappa": float(kappa_w),
    },
    "pointwise": {
        "AUROC": float(roc_auc_p),
        "AP": float(ap_p),
        "Accuracy": float(acc_p),
        "F1": float(f1_p),
        "Precision": float(prec_p_val),
        "Recall": float(rec_p_val),
    },
}

print(f"\n  ═══ DualFreqMamba EVALUATION RESULTS ═══")
print(f"\n  --- Window-Level ---")
print(f"  AUROC:    {roc_auc_w:.4f}")
print(f"  AP:       {ap_w:.4f}")
print(f"  Accuracy: {acc_w:.4f}")
print(f"  F1:       {f1_w:.4f}")
print(f"  MCC:      {mcc_w:.4f}")
print(f"\n  --- Pointwise ---")
print(f"  AUROC:    {roc_auc_p:.4f}")
print(f"  AP:       {ap_p:.4f}")
print(f"  F1:       {f1_p:.4f}")
print(f"  Precision:{prec_p_val:.4f}")
print(f"  Recall:   {rec_p_val:.4f}")

# ============================================================================
# 7. PNG 04: ROC Curves (window + pointwise)
# ============================================================================
print("[6/12] Generating ROC curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('DualFreqMamba — ROC Curves', fontsize=18, fontweight='bold', y=1.02)

ax1.plot(fpr_w, tpr_w, 'b-', linewidth=2.5, label=f'Window-Level (AUROC = {roc_auc_w:.4f})')
ax1.plot([0,1],[0,1], 'k--', alpha=0.4)
ax1.fill_between(fpr_w, tpr_w, alpha=0.15, color='blue')
ax1.plot(fpr_w[opt_w], tpr_w[opt_w], 'ro', markersize=12, label=f'Optimal Threshold')
ax1.set_xlabel('False Positive Rate', fontsize=14)
ax1.set_ylabel('True Positive Rate', fontsize=14)
ax1.set_title('Window-Level ROC', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(fpr_p, tpr_p, 'r-', linewidth=2.5, label=f'Pointwise (AUROC = {roc_auc_p:.4f})')
ax2.plot([0,1],[0,1], 'k--', alpha=0.4)
ax2.fill_between(fpr_p, tpr_p, alpha=0.15, color='red')
ax2.set_xlabel('False Positive Rate', fontsize=14)
ax2.set_ylabel('True Positive Rate', fontsize=14)
ax2.set_title('Pointwise ROC', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/04_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 04_roc_curve.png")

# ============================================================================
# 8. PNG 05: Precision-Recall Curves
# ============================================================================
print("[7/12] Generating precision-recall curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('DualFreqMamba — Precision-Recall Curves', fontsize=18, fontweight='bold', y=1.02)

ax1.plot(rec_w, prec_w, 'g-', linewidth=2.5, label=f'Window (AP = {ap_w:.4f})')
ax1.fill_between(rec_w, prec_w, alpha=0.15, color='green')
ax1.set_xlabel('Recall', fontsize=14)
ax1.set_ylabel('Precision', fontsize=14)
ax1.set_title('Window-Level P-R', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(rec_p, prec_p, 'm-', linewidth=2.5, label=f'Pointwise (AP = {ap_p:.4f})')
ax2.fill_between(rec_p, prec_p, alpha=0.15, color='magenta')
ax2.set_xlabel('Recall', fontsize=14)
ax2.set_ylabel('Precision', fontsize=14)
ax2.set_title('Pointwise P-R', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/05_precision_recall_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 05_precision_recall_curve.png")

# ============================================================================
# 9. PNG 06: Confusion Matrices
# ============================================================================
print("[8/12] Generating confusion matrices...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('DualFreqMamba — Confusion Matrices', fontsize=18, fontweight='bold', y=1.02)

# Window-level
cm_w = confusion_matrix(window_classes, preds_w)
sns.heatmap(cm_w, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
            annot_kws={'size': 20})
axes[0].set_xlabel('Predicted', fontsize=13)
axes[0].set_ylabel('Actual', fontsize=13)
axes[0].set_title(f'Window-Level (F1={f1_w:.3f})', fontsize=14, fontweight='bold')

# Pointwise
cm_p = confusion_matrix(flat_gt, preds_p)
cm_p_norm = cm_p.astype(float) / cm_p.sum(axis=1, keepdims=True)
sns.heatmap(cm_p_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1],
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
            annot_kws={'size': 18}, vmin=0, vmax=1)
axes[1].set_xlabel('Predicted', fontsize=13)
axes[1].set_ylabel('Actual', fontsize=13)
axes[1].set_title(f'Pointwise Normalized (F1={f1_p:.3f})', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}/06_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 06_confusion_matrix.png")

# ============================================================================
# 10. PNG 07: Score Distributions
# ============================================================================
print("[9/12] Generating score distributions...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('DualFreqMamba — Anomaly Score Analysis', fontsize=18, fontweight='bold', y=1.02)

# Window-level histogram
normal_wscores = wscores_norm[window_classes == 0]
anomaly_wscores = wscores_norm[window_classes == 1]

axes[0,0].hist(normal_wscores, bins=25, alpha=0.7, color='green', label='Normal', edgecolor='black')
axes[0,0].hist(anomaly_wscores, bins=25, alpha=0.7, color='red', label='Anomaly', edgecolor='black')
axes[0,0].axvline(thresh_w[opt_w], color='blue', linestyle='--', linewidth=2, label=f'Threshold={thresh_w[opt_w]:.3f}')
axes[0,0].set_title('Window-Level Score Distribution', fontsize=13, fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Window-level box plot
bp = axes[0,1].boxplot([normal_wscores, anomaly_wscores], labels=['Normal', 'Anomaly'], 
                        patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightsalmon')
axes[0,1].set_title('Window-Level Score Box Plot', fontsize=13, fontweight='bold')
axes[0,1].grid(True, alpha=0.3)

# Pointwise — by fault type
normal_pscores = flat_norm[flat_gt == 0]
anomaly_pscores = flat_norm[flat_gt == 1]

axes[1,0].hist(normal_pscores, bins=50, alpha=0.7, color='green', label='Normal Timesteps', 
               edgecolor='black', density=True)
if len(anomaly_pscores) > 0:
    axes[1,0].hist(anomaly_pscores, bins=50, alpha=0.7, color='red', label='Anomaly Timesteps',
                   edgecolor='black', density=True)
axes[1,0].set_title('Pointwise Score Distribution (Density)', fontsize=13, fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Score over time for one example
example_idx = N_NORMAL + 1  # first anomaly window
axes[1,1].plot(all_pointwise_scores[example_idx], 'orange', linewidth=2, label='Anomaly Score')
gt = all_pointwise_labels[example_idx]
if gt.sum() > 0:
    axes[1,1].fill_between(range(WINDOW_SIZE), 0, gt * all_pointwise_scores[example_idx].max(),
                           alpha=0.3, color='red', label='Ground Truth')
axes[1,1].set_title(f'Score Timeline (Window #{example_idx}, {fault_type_labels[example_idx]})', 
                    fontsize=13, fontweight='bold')
axes[1,1].set_xlabel('Timestep')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/07_score_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 07_score_distributions.png")

# ============================================================================
# 11. Training convergence
# ============================================================================
print("[10/12] Running training convergence simulation...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses = []
recon_hist = []
freq_hist = []
assoc_hist = []

n_epochs = 30
for epoch in range(n_epochs):
    epoch_total = []
    epoch_recon = []
    epoch_freq = []
    epoch_assoc = []
    
    indices = np.random.permutation(N_NORMAL)
    for i in range(0, N_NORMAL, batch_size):
        bi = indices[i:i+batch_size]
        batch_tensor = torch.tensor(np.stack([windows[j] for j in bi]))
        
        output = model(batch_tensor)
        loss = output.loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_total.append(loss.item())
        epoch_recon.append(output.reconstruction_loss.item())
        if output.frequency_loss is not None:
            epoch_freq.append(output.frequency_loss.item())
        if output.association_loss is not None:
            epoch_assoc.append(output.association_loss.item())
    
    train_losses.append(np.mean(epoch_total))
    recon_hist.append(np.mean(epoch_recon))
    freq_hist.append(np.mean(epoch_freq) if epoch_freq else 0)
    assoc_hist.append(np.mean(epoch_assoc) if epoch_assoc else 0)
    
    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}: loss={train_losses[-1]:.4f}")

# PNG 08: Training convergence
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('DualFreqMamba — Training Convergence', fontsize=18, fontweight='bold', y=1.02)

ep = range(1, n_epochs+1)

axes[0,0].plot(ep, train_losses, 'b-o', markersize=3, linewidth=2, label='Total Loss')
axes[0,0].set_title('Total Training Loss', fontsize=13, fontweight='bold')
axes[0,0].set_xlabel('Epoch'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(ep, recon_hist, 'r-s', markersize=3, linewidth=2, label='Time Reconstruction')
axes[0,1].set_title('Time-Domain Reconstruction Loss', fontsize=13, fontweight='bold')
axes[0,1].set_xlabel('Epoch'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(ep, freq_hist, 'g-^', markersize=3, linewidth=2, label='Frequency Reconstruction')
axes[1,0].set_title('Frequency-Domain Reconstruction Loss', fontsize=13, fontweight='bold')
axes[1,0].set_xlabel('Epoch'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(ep, assoc_hist, 'm-D', markersize=3, linewidth=2, label='Association Discrepancy')
axes[1,1].set_title('Association Discrepancy Loss (Minimax)', fontsize=13, fontweight='bold')
axes[1,1].set_xlabel('Epoch'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/08_training_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 08_training_convergence.png")

# ============================================================================
# 12. Re-evaluate after training
# ============================================================================
print("[11/12] Re-evaluating after training...")
model.eval()

trained_window_scores = []
trained_pointwise_scores = []

for i in range(0, N_WINDOWS, batch_size):
    batch_data = windows[i:i+batch_size]
    batch_tensor = torch.tensor(np.stack(batch_data))
    with torch.no_grad():
        output = model(batch_tensor, return_anomaly_labels=True)
    scores = output.anomaly_score.cpu().numpy()
    trained_pointwise_scores.extend(scores)
    trained_window_scores.extend(scores.max(axis=1).tolist())

trained_window_scores = np.array(trained_window_scores)
trained_pointwise_scores = np.array(trained_pointwise_scores)

# Post-training window metrics
tw_norm = (trained_window_scores - trained_window_scores.min()) / (trained_window_scores.max() - trained_window_scores.min() + 1e-10)
fpr_w2, tpr_w2, thresh_w2 = roc_curve(window_classes, tw_norm)
roc_auc_w2 = auc(fpr_w2, tpr_w2)
ap_w2 = average_precision_score(window_classes, tw_norm)
j_w2 = tpr_w2 - fpr_w2
opt_w2 = np.argmax(j_w2)
preds_w2 = (tw_norm >= thresh_w2[opt_w2]).astype(int)
f1_w2 = f1_score(window_classes, preds_w2)
acc_w2 = accuracy_score(window_classes, preds_w2)
mcc_w2 = matthews_corrcoef(window_classes, preds_w2)

# Post-training pointwise metrics
tp_flat = trained_pointwise_scores.flatten()
tp_norm = (tp_flat - tp_flat.min()) / (tp_flat.max() - tp_flat.min() + 1e-10)
fpr_p2, tpr_p2, thresh_p2 = roc_curve(flat_gt, tp_norm)
roc_auc_p2 = auc(fpr_p2, tpr_p2)
ap_p2 = average_precision_score(flat_gt, tp_norm)
j_p2 = tpr_p2 - fpr_p2
opt_p2 = np.argmax(j_p2)
preds_p2 = (tp_norm >= thresh_p2[opt_p2]).astype(int)
f1_p2 = f1_score(flat_gt, preds_p2, zero_division=0)

print(f"\n  ═══ POST-TRAINING RESULTS ═══")
print(f"  Window AUROC: {roc_auc_w:.4f} → {roc_auc_w2:.4f}")
print(f"  Window F1:    {f1_w:.4f} → {f1_w2:.4f}")
print(f"  Window Acc:   {acc_w:.4f} → {acc_w2:.4f}")
print(f"  Point AUROC:  {roc_auc_p:.4f} → {roc_auc_p2:.4f}")
print(f"  Point F1:     {f1_p:.4f} → {f1_p2:.4f}")

metrics["post_training"] = {
    "window": {"AUROC": float(roc_auc_w2), "AP": float(ap_w2), "F1": float(f1_w2), 
               "Accuracy": float(acc_w2), "MCC": float(mcc_w2)},
    "pointwise": {"AUROC": float(roc_auc_p2), "AP": float(ap_p2), "F1": float(f1_p2)},
}

# ============================================================================
# 13. PNG 09: Parameter Breakdown
# ============================================================================
print("[12/12] Generating remaining plots...")

param_groups = {}
for name, param in model.named_parameters():
    group = name.split('.')[0]
    if group == 'instance_norm':
        group = 'inst_norm'
    if group not in param_groups:
        param_groups[group] = 0
    param_groups[group] += param.numel()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('DualFreqMamba — Model Parameter Analysis', fontsize=18, fontweight='bold', y=1.02)

labels_pie = [f"{k}\n({v/1e3:.1f}K)" for k, v in param_groups.items()]
sizes = list(param_groups.values())
colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
wedges, texts, autotexts = ax1.pie(sizes, labels=labels_pie, autopct='%1.1f%%', colors=colors,
                                     textprops={'fontsize': 9}, startangle=90)
ax1.set_title('Parameter Distribution by Component', fontsize=13, fontweight='bold')

groups_sorted = sorted(param_groups.items(), key=lambda x: x[1], reverse=True)
g_names = [g[0] for g in groups_sorted]
g_vals = [g[1]/1e3 for g in groups_sorted]
bars = ax2.barh(g_names, g_vals, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(g_names))))
for bar, val in zip(bars, g_vals):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{val:.1f}K', va='center', fontsize=10)
ax2.set_xlabel('Parameters (Thousands)', fontsize=12)
ax2.set_title('Parameters by Component (Sorted)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUT}/09_parameter_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 09_parameter_breakdown.png")

# ============================================================================
# PNG 10: Frequency Analysis
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle('DualFreqMamba — Frequency Domain Analysis', fontsize=18, fontweight='bold', y=1.02)

# Normal vs anomaly FFT comparison
normal_avg_fft = np.mean([np.abs(np.fft.rfft(windows[i], axis=1)) for i in range(N_NORMAL)], axis=0)
anomaly_avg_fft = np.mean([np.abs(np.fft.rfft(windows[i], axis=1)) for i in range(N_NORMAL, N_WINDOWS)], axis=0)

for ch in range(min(3, N_CHANNELS)):
    axes[0,0].plot(normal_avg_fft[ch], label=f'Ch {ch}', alpha=0.8)
axes[0,0].set_title('Average FFT — Normal Windows', fontsize=13, fontweight='bold', color='green')
axes[0,0].set_xlabel('Frequency Bin'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

for ch in range(min(3, N_CHANNELS)):
    axes[0,1].plot(anomaly_avg_fft[ch], label=f'Ch {ch}', alpha=0.8)
axes[0,1].set_title('Average FFT — Anomaly Windows', fontsize=13, fontweight='bold', color='red')
axes[0,1].set_xlabel('Frequency Bin'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# Difference
diff_fft = np.abs(anomaly_avg_fft - normal_avg_fft)
im = axes[0,2].imshow(diff_fft, aspect='auto', cmap='hot', interpolation='bilinear')
axes[0,2].set_title('FFT Difference |Anomaly - Normal|', fontsize=13, fontweight='bold')
axes[0,2].set_xlabel('Frequency Bin'); axes[0,2].set_ylabel('Channel')
plt.colorbar(im, ax=axes[0,2])

# Per fault type spectral signatures
for i, ft in enumerate(fault_types[:3]):
    ft_indices = [N_NORMAL + j for j, l in enumerate(fault_type_labels[N_NORMAL:]) if l == ft][:5]
    ft_avg_fft = np.mean([np.abs(np.fft.rfft(windows[idx], axis=1)) for idx in ft_indices], axis=0)
    
    for ch in range(min(3, N_CHANNELS)):
        axes[1,i].plot(ft_avg_fft[ch], alpha=0.8)
    axes[1,i].set_title(f'FFT — {ft.replace("_"," ").title()} Faults', fontsize=13, fontweight='bold')
    axes[1,i].set_xlabel('Frequency Bin')
    axes[1,i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/10_frequency_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 10_frequency_analysis.png")

# ============================================================================
# PNG 11: Per Anomaly Type Performance
# ============================================================================
per_type_metrics = {}
for ft in fault_types:
    ft_win_indices = [N_NORMAL + i for i, l in enumerate(fault_type_labels[N_NORMAL:]) if l == ft]
    normal_indices = list(range(N_NORMAL))
    
    eval_idx = normal_indices + ft_win_indices
    eval_labels = np.array([0]*len(normal_indices) + [1]*len(ft_win_indices))
    eval_scores = tw_norm[eval_idx]
    
    fpr_ft, tpr_ft, _ = roc_curve(eval_labels, eval_scores)
    auc_ft = auc(fpr_ft, tpr_ft)
    ap_ft = average_precision_score(eval_labels, eval_scores)
    
    per_type_metrics[ft] = {'AUROC': auc_ft, 'AP': ap_ft, 'fpr': fpr_ft, 'tpr': tpr_ft, 'n': len(ft_win_indices)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('DualFreqMamba — Per-Fault-Type Performance (Post-Training)', fontsize=18, fontweight='bold', y=1.02)

ft_colors = {'spike': '#FF6B6B', 'drift': '#4ECDC4', 'frequency_shift': '#45B7D1', 
             'dropout': '#96CEB4', 'noise_burst': '#DDA0DD'}

for ft, m in per_type_metrics.items():
    ax1.plot(m['fpr'], m['tpr'], linewidth=2, color=ft_colors[ft],
             label=f'{ft.replace("_"," ").title()} (AUROC={m["AUROC"]:.3f})')
ax1.plot([0,1],[0,1], 'k--', alpha=0.4)
ax1.set_xlabel('FPR', fontsize=13); ax1.set_ylabel('TPR', fontsize=13)
ax1.set_title('ROC per Fault Type', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

faults = list(per_type_metrics.keys())
aurocs_ft = [per_type_metrics[ft]['AUROC'] for ft in faults]
aps_ft = [per_type_metrics[ft]['AP'] for ft in faults]

x = np.arange(len(faults))
bars1 = ax2.bar(x - 0.2, aurocs_ft, 0.35, label='AUROC', color='steelblue')
bars2 = ax2.bar(x + 0.2, aps_ft, 0.35, label='AP', color='coral')
for b in bars1:
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{b.get_height():.3f}', ha='center', fontsize=9)
for b in bars2:
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{b.get_height():.3f}', ha='center', fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels([ft.replace('_',' ').title() for ft in faults], fontsize=10)
ax2.set_ylabel('Score', fontsize=13)
ax2.set_title('AUROC & AP per Fault Type', fontsize=14, fontweight='bold')
ax2.legend(); ax2.set_ylim(0, 1.15); ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT}/11_per_anomaly_type_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 11_per_anomaly_type_performance.png")

# ============================================================================
# PNG 12: Complete Dashboard
# ============================================================================
fig = plt.figure(figsize=(24, 16))
fig.suptitle('DualFreqMamba — Complete Evaluation Dashboard', fontsize=22, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.4)

# ROC comparison
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(fpr_w, tpr_w, 'b--', linewidth=1.5, alpha=0.5, label=f'Pre (AUROC={roc_auc_w:.3f})')
ax1.plot(fpr_w2, tpr_w2, 'b-', linewidth=2.5, label=f'Post (AUROC={roc_auc_w2:.3f})')
ax1.plot([0,1],[0,1],'k--',alpha=0.3)
ax1.set_title('Window ROC: Pre vs Post', fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Pointwise ROC comparison
ax2 = fig.add_subplot(gs[0, 2:4])
ax2.plot(fpr_p, tpr_p, 'r--', linewidth=1.5, alpha=0.5, label=f'Pre (AUROC={roc_auc_p:.3f})')
ax2.plot(fpr_p2, tpr_p2, 'r-', linewidth=2.5, label=f'Post (AUROC={roc_auc_p2:.3f})')
ax2.plot([0,1],[0,1],'k--',alpha=0.3)
ax2.set_title('Pointwise ROC: Pre vs Post', fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

# Training loss
ax3 = fig.add_subplot(gs[1, 0:2])
ax3.plot(ep, train_losses, 'b-o', markersize=3, linewidth=2)
ax3.set_title('Training Loss Convergence', fontweight='bold')
ax3.set_xlabel('Epoch'); ax3.grid(True, alpha=0.3)

# Per-type AUROC
ax4 = fig.add_subplot(gs[1, 2:4])
bars = ax4.bar([ft.replace('_','\n').title() for ft in faults], 
               aurocs_ft, color=[ft_colors[ft] for ft in faults])
for b, v in zip(bars, aurocs_ft):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=10)
ax4.set_title('AUROC per Fault Type (Post-Training)', fontweight='bold')
ax4.set_ylim(0, 1.15); ax4.grid(True, alpha=0.3, axis='y')

# Post-training confusion
ax5 = fig.add_subplot(gs[2, 0:2])
cm_w2 = confusion_matrix(window_classes, preds_w2)
sns.heatmap(cm_w2, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'],
            annot_kws={'size': 16})
ax5.set_title('Post-Training Window Confusion Matrix', fontweight='bold')

# Metrics table
ax6 = fig.add_subplot(gs[2, 2:4])
ax6.axis('off')
table_data = [
    ['Metric', 'Pre-Train', 'Post-Train', 'Δ'],
    ['Window AUROC', f'{roc_auc_w:.4f}', f'{roc_auc_w2:.4f}', f'+{roc_auc_w2-roc_auc_w:.4f}'],
    ['Window F1', f'{f1_w:.4f}', f'{f1_w2:.4f}', f'+{f1_w2-f1_w:.4f}'],
    ['Window Acc', f'{acc_w:.4f}', f'{acc_w2:.4f}', f'+{acc_w2-acc_w:.4f}'],
    ['Window MCC', f'{mcc_w:.4f}', f'{mcc_w2:.4f}', f'+{mcc_w2-mcc_w:.4f}'],
    ['Point AUROC', f'{roc_auc_p:.4f}', f'{roc_auc_p2:.4f}', f'+{roc_auc_p2-roc_auc_p:.4f}'],
    ['Point F1', f'{f1_p:.4f}', f'{f1_p2:.4f}', f'+{f1_p2-f1_p:.4f}'],
    ['Params', f'{total_params:,}', '-', '-'],
]
table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
for i in range(len(table_data[0])):
    table[0, i].set_facecolor('#4472C4')
    table[0, i].set_text_props(color='white', fontweight='bold')
ax6.set_title('Complete Metrics Summary', fontweight='bold', pad=20)

plt.savefig(f'{OUT}/12_complete_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 12_complete_dashboard.png")

# ============================================================================
# PNG 01: Architecture Overview
# ============================================================================
fig, ax = plt.subplots(figsize=(22, 16))
ax.set_xlim(0, 22); ax.set_ylim(0, 16)
ax.axis('off')
ax.set_title('DualFreqMamba — Architecture Overview', fontsize=22, fontweight='bold', pad=20)

def draw_box(ax, x, y, w, h, text, color='lightblue', fontsize=9, text_color='black'):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, 
            fontweight='bold', color=text_color, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='black'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Input
draw_box(ax, 7, 14.5, 8, 0.8, 'Input: X ∈ ℝ^(B × 10ch × 100t)\n+ Instance Normalization', '#FFE0B2', 10)

# Three branches
draw_box(ax, 0.5, 11.5, 6, 2.2, 'FFT Frequency Patching\n(CATCH-inspired)\n• rfft → real + imag\n• Stride-2 patches\n• Channel-Masked\n  Transformer ×1', '#BBDEFB', 9)
draw_box(ax, 8, 11.5, 6, 2.2, 'CWT Scalogram Branch\n(TSCMamba-inspired)\n• Morlet wavelet (8 scales)\n• Conv2D patch embed\n• FFN projection', '#C8E6C9', 9)
draw_box(ax, 15.5, 11.5, 6, 2.2, 'Raw Temporal Branch\n(Direct Embedding)\n• Linear projection\n• Positional encoding\n• LayerNorm', '#FFF9C4', 9)

draw_arrow(ax, 9, 14.5, 3.5, 13.7)
draw_arrow(ax, 11, 14.5, 11, 13.7)
draw_arrow(ax, 13, 14.5, 18.5, 13.7)

# Fusion
draw_box(ax, 5, 9.5, 12, 1.4, 'Adaptive Gated Tri-Branch Fusion\ngate_fft ⊙ F_fft + gate_cwt ⊙ F_cwt + gate_raw ⊙ F_raw\n(Learned softmax gates)', '#E1BEE7', 10)
draw_arrow(ax, 3.5, 11.5, 8, 10.9)
draw_arrow(ax, 11, 11.5, 11, 10.9)
draw_arrow(ax, 18.5, 11.5, 14, 10.9)

# Mamba decoder
draw_box(ax, 5, 7, 12, 1.8, 'Mamba SSM Reconstruction Decoder (×2 layers)\n├ Selective State Space Model (input-dependent transitions)\n├ Gated Skip Connection: gate ⊙ SSM + (1-gate) ⊙ input\n└ Feed-Forward Network with LayerNorm', '#FFCDD2', 9)
draw_arrow(ax, 11, 9.5, 11, 8.8)

# Association discrepancy
draw_box(ax, 0.5, 4.5, 10, 1.8, 'Sparse Association Discrepancy\n• Prior: Gaussian kernel (neighbor-based)\n• Series: Learned attention patterns\n• Score: KL(Prior ‖ Series) per timestep\n• Minimax optimization', '#B3E5FC', 9)
draw_arrow(ax, 8, 7, 5.5, 6.3)

# Reconstruction
draw_box(ax, 12, 4.5, 9.5, 1.8, 'Dual-Domain Reconstruction\n• Time: MSE(X, X̂_time)\n• Freq: MSE(FFT(X), FFT(X̂))\n• Combined loss with weights\n  [1.0×time + 0.5×freq + 1.0×assoc]', '#DCEDC8', 9)
draw_arrow(ax, 14, 7, 16.75, 6.3)

# Output
draw_box(ax, 4, 2, 14, 1.8, 'OUTPUT\n• Anomaly Score: (B, T) per-timestep scores\n• Anomaly Labels: (B, T) binary — threshold = μ + 2σ\n• Score = softmax(-association_discrepancy) × recon_error', '#B2DFDB', 10, 'darkgreen')
draw_arrow(ax, 5.5, 4.5, 9, 3.8)
draw_arrow(ax, 16.75, 4.5, 13, 3.8)

stats = f"Total: {total_params/1e3:.1f}K params | All Trainable | Channels: {N_CHANNELS} | Window: {WINDOW_SIZE}"
ax.text(11, 1.2, stats, ha='center', fontsize=12, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(f'{OUT}/01_architecture_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 01_architecture_overview.png")

# Save final metrics
with open(f'{OUT}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

with open(f'{OUT}/classification_report.txt', 'w') as f:
    f.write("DualFreqMamba — Classification Report\n")
    f.write("=" * 50 + "\n\n")
    f.write("WINDOW-LEVEL (Pre-Training):\n")
    f.write(classification_report(window_classes, preds_w, target_names=['Normal', 'Anomaly']))
    f.write(f"\n\nWINDOW-LEVEL (Post-Training):\n")
    f.write(classification_report(window_classes, preds_w2, target_names=['Normal', 'Anomaly']))

print(f"\n✅ DualFreqMamba evaluation complete! {len(os.listdir(OUT))} files in {OUT}")
print("Files:", sorted(os.listdir(OUT)))
