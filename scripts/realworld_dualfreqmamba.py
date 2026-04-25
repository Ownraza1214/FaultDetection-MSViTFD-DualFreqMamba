"""
==========================================================================
REAL-WORLD TESTING — DualFreqMamba Time Series Fault Detector
==========================================================================
Tests with:
  - Realistic industrial sensor data (temperature, pressure, vibration, etc.)
  - Real-world fault scenarios (bearing failure, pump cavitation, etc.)
  - Proper train→eval pipeline
Saves all input/output pairs in organized folders
==========================================================================
"""
import sys, os
sys.path.insert(0, '/app')
os.environ['MPLBACKEND'] = 'Agg'

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json

from dualfreqmamba.configuration_dualfreqmamba import DualFreqMambaConfig
from dualfreqmamba.modeling_dualfreqmamba import DualFreqMambaModel

# Create output directories
BASE = "/app/realworld_test"
for d in [
    f"{BASE}/dualfreqmamba/inputs/normal",
    f"{BASE}/dualfreqmamba/inputs/faulty",
    f"{BASE}/dualfreqmamba/outputs/normal",
    f"{BASE}/dualfreqmamba/outputs/faulty",
    f"{BASE}/dualfreqmamba/outputs/comparison",
    f"{BASE}/dualfreqmamba/outputs/training",
]:
    os.makedirs(d, exist_ok=True)

print("=" * 70)
print("REAL-WORLD TESTING — DualFreqMamba Time Series Fault Detector")
print("=" * 70)

# ============================================================================
# 1. Load model
# ============================================================================
N_CH = 8  # 8 sensor channels
WIN = 200  # 200 timesteps per window

config = DualFreqMambaConfig(
    n_channels=N_CH,
    window_size=WIN,
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

model = DualFreqMambaModel(config)
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# ============================================================================
# 2. Create REALISTIC industrial sensor data
# ============================================================================
print("\n[1] Generating realistic industrial sensor data...")

np.random.seed(42)
t = np.linspace(0, 10, WIN)
SENSOR_NAMES = ['Temperature', 'Pressure', 'Vibration_X', 'Vibration_Y', 
                'Flow_Rate', 'Current', 'RPM', 'Torque']
SCENARIO_COLORS = {'normal': 'green', 'bearing_failure': '#FF4444', 'pump_cavitation': '#FF8800',
                   'overheating': '#CC0066', 'sensor_drift': '#6644CC', 
                   'electrical_fault': '#0088FF', 'mechanical_looseness': '#44AA00'}

def generate_normal_machine_data(variant=0):
    """Generate normal operating machine sensor data."""
    data = np.zeros((N_CH, WIN), dtype=np.float32)
    
    # Temperature: 60-80°C, slow variation
    data[0] = 70 + 5 * np.sin(0.3 * t + variant) + 0.5 * np.random.randn(WIN)
    # Pressure: 4-6 bar, correlated with temp
    data[1] = 5 + 0.5 * np.sin(0.3 * t + variant + 0.5) + 0.2 * np.random.randn(WIN)
    # Vibration X: low amplitude, high frequency
    data[2] = 0.1 * np.sin(25 * t) + 0.05 * np.sin(50 * t) + 0.02 * np.random.randn(WIN)
    # Vibration Y: similar to X, 90° phase shift
    data[3] = 0.1 * np.cos(25 * t) + 0.05 * np.cos(50 * t) + 0.02 * np.random.randn(WIN)
    # Flow rate: steady ~10 L/min
    data[4] = 10 + 0.3 * np.sin(0.5 * t) + 0.1 * np.random.randn(WIN)
    # Current: 5A nominal
    data[5] = 5 + 0.2 * np.sin(0.4 * t) + 0.1 * np.random.randn(WIN)
    # RPM: 1500 nominal
    data[6] = 1500 + 10 * np.sin(0.2 * t + variant * 0.3) + 2 * np.random.randn(WIN)
    # Torque: correlated with RPM
    data[7] = 50 + 2 * np.sin(0.2 * t) + data[6] / 1500 * 3 + 0.5 * np.random.randn(WIN)
    
    return data, np.zeros(WIN, dtype=np.int32)

def generate_bearing_failure(variant=0):
    """Bearing degradation: increasing vibration with characteristic frequencies."""
    data, _ = generate_normal_machine_data(variant)
    labels = np.zeros(WIN, dtype=np.int32)
    
    # Fault starts at t=100, progressively worsens
    start = 80 + variant * 10
    fault_region = slice(start, min(start + 80, WIN))
    
    # Increasing vibration amplitude with bearing defect frequency
    progress = np.linspace(0, 1, fault_region.stop - fault_region.start)
    defect_freq = 12.5  # BPFO characteristic frequency
    
    data[2, fault_region] += progress * 0.8 * np.sin(defect_freq * 2 * np.pi * t[fault_region])
    data[3, fault_region] += progress * 0.6 * np.cos(defect_freq * 2 * np.pi * t[fault_region])
    # Temperature rises due to friction
    data[0, fault_region] += progress * 15
    # Current increases
    data[5, fault_region] += progress * 1.5
    
    labels[fault_region] = 1
    return data, labels

def generate_pump_cavitation(variant=0):
    """Pump cavitation: pressure drops, vibration bursts, flow instability."""
    data, _ = generate_normal_machine_data(variant)
    labels = np.zeros(WIN, dtype=np.int32)
    
    start = 90 + variant * 5
    end = min(start + 60, WIN)
    fault_region = slice(start, end)
    
    # Pressure drops and oscillates
    data[1, fault_region] -= 2.0 + 1.0 * np.sin(15 * t[fault_region])
    # Flow rate becomes unstable
    data[4, fault_region] += 3.0 * np.random.randn(end - start)
    # High-frequency vibration bursts
    data[2, fault_region] += 0.5 * np.abs(np.sin(40 * t[fault_region])) * np.random.randn(end - start)
    data[3, fault_region] += 0.4 * np.abs(np.cos(40 * t[fault_region])) * np.random.randn(end - start)
    
    labels[fault_region] = 1
    return data, labels

def generate_overheating(variant=0):
    """Overheating: temperature ramp-up, thermal runaway pattern."""
    data, _ = generate_normal_machine_data(variant)
    labels = np.zeros(WIN, dtype=np.int32)
    
    start = 70 + variant * 10
    fault_region = slice(start, WIN)
    n = WIN - start
    
    # Exponential temperature rise
    ramp = np.exp(np.linspace(0, 2, n)) - 1
    data[0, fault_region] += ramp * 20
    # Pressure increases with temperature
    data[1, fault_region] += ramp * 0.8
    # Lubricant breakdown → more vibration
    data[2, fault_region] += ramp * 0.15 * np.random.randn(n)
    data[3, fault_region] += ramp * 0.12 * np.random.randn(n)
    
    labels[fault_region] = 1
    return data, labels

def generate_sensor_drift(variant=0):
    """Sensor drift: gradual calibration error in 1-2 sensors."""
    data, _ = generate_normal_machine_data(variant)
    labels = np.zeros(WIN, dtype=np.int32)
    
    start = 60 + variant * 15
    fault_region = slice(start, WIN)
    n = WIN - start
    
    # Linear drift on temperature sensor
    data[0, fault_region] += np.linspace(0, 25, n)
    # Step offset on pressure
    data[1, fault_region] += 2.5
    
    labels[fault_region] = 1
    return data, labels

def generate_electrical_fault(variant=0):
    """Electrical fault: current spikes, RPM fluctuations."""
    data, _ = generate_normal_machine_data(variant)
    labels = np.zeros(WIN, dtype=np.int32)
    
    # Intermittent current spikes
    spike_positions = np.random.randint(50, WIN-10, size=8)
    for pos in spike_positions:
        width = np.random.randint(2, 6)
        data[5, pos:pos+width] += np.random.uniform(3, 8)  # Current spike
        data[6, pos:pos+width] -= np.random.uniform(50, 150)  # RPM dip
        data[7, pos:pos+width] += np.random.uniform(10, 30)  # Torque spike
        labels[pos:pos+width] = 1
    
    return data, labels

def generate_mechanical_looseness(variant=0):
    """Mechanical looseness: sub-harmonic vibrations, RPM instability."""
    data, _ = generate_normal_machine_data(variant)
    labels = np.zeros(WIN, dtype=np.int32)
    
    start = 80 + variant * 10
    fault_region = slice(start, WIN)
    n = WIN - start
    
    # Sub-harmonic vibration (1/2 × RPM frequency)
    data[2, fault_region] += 0.4 * np.sin(12.5 * t[fault_region])
    data[3, fault_region] += 0.3 * np.sin(12.5 * t[fault_region] + np.pi/4)
    # RPM becomes noisy
    data[6, fault_region] += 30 * np.random.randn(n)
    # Torque fluctuates
    data[7, fault_region] += 8 * np.sin(6.25 * t[fault_region])
    
    labels[fault_region] = 1
    return data, labels

# Generate data
fault_generators = {
    'bearing_failure': generate_bearing_failure,
    'pump_cavitation': generate_pump_cavitation,
    'overheating': generate_overheating,
    'sensor_drift': generate_sensor_drift,
    'electrical_fault': generate_electrical_fault,
    'mechanical_looseness': generate_mechanical_looseness,
}

# Normal: 15 windows
normal_data = []
normal_labels_ts = []
for i in range(15):
    d, l = generate_normal_machine_data(variant=i)
    normal_data.append(d)
    normal_labels_ts.append(l)

# Faulty: 3 per fault type = 18 windows
faulty_data = []
faulty_labels_ts = []
faulty_types = []
for fault_name, gen_func in fault_generators.items():
    for v in range(3):
        d, l = gen_func(variant=v)
        faulty_data.append(d)
        faulty_labels_ts.append(l)
        faulty_types.append(fault_name)

print(f"  Normal windows: {len(normal_data)}")
print(f"  Faulty windows: {len(faulty_data)}")
print(f"  Fault types: {dict(zip(*np.unique(faulty_types, return_counts=True)))}")
print(f"  Sensors: {SENSOR_NAMES}")

# ============================================================================
# 3. Save input visualizations
# ============================================================================
print("\n[2] Saving input visualizations...")

def save_ts_input(data, labels, name, folder, fault_type='normal'):
    """Save a time series input as a multi-channel visualization PNG."""
    fig, axes = plt.subplots(N_CH, 1, figsize=(16, N_CH * 1.8), sharex=True)
    fig.suptitle(f'{"Normal" if fault_type == "normal" else fault_type.replace("_"," ").title()} — {name}', 
                fontsize=14, fontweight='bold', 
                color='green' if fault_type == 'normal' else 'red', y=1.01)
    
    for ch in range(N_CH):
        axes[ch].plot(data[ch], linewidth=1, color='steelblue')
        axes[ch].set_ylabel(SENSOR_NAMES[ch], fontsize=8, rotation=0, ha='right', labelpad=70)
        axes[ch].grid(True, alpha=0.2)
        axes[ch].tick_params(labelsize=7)
        
        # Highlight fault regions
        if labels is not None and labels.sum() > 0:
            fault_idx = np.where(labels == 1)[0]
            if len(fault_idx) > 0:
                axes[ch].axvspan(fault_idx[0], fault_idx[-1], alpha=0.2, color='red')
    
    axes[-1].set_xlabel('Timestep', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{BASE}/dualfreqmamba/inputs/{folder}/{name}.png', dpi=120, bbox_inches='tight')
    plt.close()

for i, (d, l) in enumerate(zip(normal_data, normal_labels_ts)):
    save_ts_input(d, l, f'normal_{i+1:02d}', 'normal', 'normal')

for i, (d, l, ft) in enumerate(zip(faulty_data, faulty_labels_ts, faulty_types)):
    save_ts_input(d, l, f'{ft}_{i+1:02d}', 'faulty', ft)

print(f"  Saved {len(normal_data)} normal + {len(faulty_data)} faulty input PNGs")

# ============================================================================
# 4. Train on normal data
# ============================================================================
print("\n[3] Training on normal sensor data...")

normal_tensors = torch.tensor(np.stack(normal_data))
print(f"  Training data: {normal_tensors.shape}")

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

train_losses = []
batch_size = 8

for epoch in range(50):
    indices = torch.randperm(len(normal_tensors))
    epoch_loss = []
    for i in range(0, len(indices), batch_size):
        batch = normal_tensors[indices[i:i+batch_size]]
        output = model(batch)
        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss.append(loss.item())
    
    scheduler.step()
    train_losses.append(np.mean(epoch_loss))
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50: loss={train_losses[-1]:.4f}")

# Save training curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(train_losses)+1), train_losses, 'b-o', markersize=3, linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('DualFreqMamba — Real-World Training Convergence', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{BASE}/dualfreqmamba/outputs/training/training_curve.png', dpi=150)
plt.close()

# ============================================================================
# 5. Run inference on all data
# ============================================================================
print("\n[4] Running inference on all windows...")
model.eval()

all_data = normal_data + faulty_data
all_labels_pointwise = normal_labels_ts + faulty_labels_ts
all_window_labels = [0] * len(normal_data) + [1] * len(faulty_data)
all_type_labels = ['normal'] * len(normal_data) + faulty_types

all_window_scores = []
all_pointwise_scores = []

for i in range(len(all_data)):
    tensor = torch.tensor(all_data[i]).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor, return_anomaly_labels=True)
    
    scores = output.anomaly_score.squeeze().cpu().numpy()
    all_pointwise_scores.append(scores)
    all_window_scores.append(scores.max())

all_window_scores = np.array(all_window_scores)
all_pointwise_scores_arr = np.array(all_pointwise_scores)

# ============================================================================
# 6. Save output visualizations
# ============================================================================
print("[5] Saving output visualizations...")

def save_ts_output(data, gt_labels, pred_scores, name, folder, fault_type, window_score):
    """Save time series output with anomaly detection results."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.35, wspace=0.3)
    
    status = "NORMAL" if fault_type == 'normal' else f"FAULT: {fault_type.replace('_', ' ').title()}"
    color = 'green' if fault_type == 'normal' else 'red'
    fig.suptitle(f'{status} — Window Score: {window_score:.6f}', 
                fontsize=15, fontweight='bold', color=color, y=0.98)
    
    # Top: All channels with fault highlighting
    ax1 = fig.add_subplot(gs[0, :])
    for ch in range(N_CH):
        offset = ch * 3
        ax1.plot(data[ch] / (np.abs(data[ch]).max() + 1e-8) + offset, 
                linewidth=0.8, label=SENSOR_NAMES[ch], alpha=0.8)
    if gt_labels.sum() > 0:
        fi = np.where(gt_labels == 1)[0]
        ax1.axvspan(fi[0], fi[-1], alpha=0.15, color='red', label='Ground Truth Fault')
    ax1.set_title('Sensor Channels (normalized + stacked)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7, ncol=4, loc='upper right')
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, WIN)
    
    # Middle left: Anomaly score timeline
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(pred_scores, 'orange', linewidth=2, label='Anomaly Score')
    ax2.fill_between(range(len(pred_scores)), pred_scores, alpha=0.3, color='orange')
    if gt_labels.sum() > 0:
        fi = np.where(gt_labels == 1)[0]
        ax2.axvspan(fi[0], fi[-1], alpha=0.2, color='red', label='Ground Truth')
    ax2.set_title('Per-Timestep Anomaly Score', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, WIN)
    
    # Middle right: FFT of input
    ax3 = fig.add_subplot(gs[1, 1])
    for ch in range(min(4, N_CH)):
        fft_mag = np.abs(np.fft.rfft(data[ch]))
        ax3.plot(fft_mag, label=SENSOR_NAMES[ch], alpha=0.7)
    ax3.set_title('Frequency Spectrum (FFT)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Frequency Bin')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Bottom left: Channel-wise heatmap of anomaly contribution
    ax4 = fig.add_subplot(gs[2, 0])
    # Show raw data deviation from mean
    deviation = np.abs(data - data.mean(axis=1, keepdims=True))
    im = ax4.imshow(deviation, aspect='auto', cmap='YlOrRd', interpolation='bilinear')
    ax4.set_yticks(range(N_CH))
    ax4.set_yticklabels(SENSOR_NAMES, fontsize=8)
    ax4.set_xlabel('Timestep')
    ax4.set_title('Channel Deviation Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # Bottom right: Score distribution
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(pred_scores, bins=30, alpha=0.7, color='orange', edgecolor='black')
    thresh = pred_scores.mean() + 2 * pred_scores.std()
    ax5.axvline(thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold={thresh:.4f}')
    n_above = (pred_scores > thresh).sum()
    ax5.set_title(f'Score Distribution ({n_above} timesteps above threshold)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    plt.savefig(f'{BASE}/dualfreqmamba/outputs/{folder}/{name}_result.png', dpi=120, bbox_inches='tight')
    plt.close()

for i in range(len(normal_data)):
    save_ts_output(normal_data[i], normal_labels_ts[i], all_pointwise_scores[i],
                   f'normal_{i+1:02d}', 'normal', 'normal', all_window_scores[i])

for i in range(len(faulty_data)):
    idx = len(normal_data) + i
    save_ts_output(faulty_data[i], faulty_labels_ts[i], all_pointwise_scores[idx],
                   f'{faulty_types[i]}_{i+1:02d}', 'faulty', faulty_types[i], all_window_scores[idx])

print(f"  Saved {len(normal_data)} normal + {len(faulty_data)} faulty output PNGs")

# ============================================================================
# 7. Compute metrics
# ============================================================================
print("\n[6] Computing real-world metrics...")
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, matthews_corrcoef

# Window-level
wl = np.array(all_window_labels)
ws = all_window_scores
ws_norm = (ws - ws.min()) / (ws.max() - ws.min() + 1e-10)

fpr_w, tpr_w, thresh_w = roc_curve(wl, ws_norm)
roc_auc_w = auc(fpr_w, tpr_w)
ap_w = average_precision_score(wl, ws_norm)

j_w = tpr_w - fpr_w
opt_w = np.argmax(j_w)
preds_w = (ws_norm >= thresh_w[opt_w]).astype(int)
f1_w = f1_score(wl, preds_w)
acc_w = accuracy_score(wl, preds_w)
mcc_w = matthews_corrcoef(wl, preds_w)

# Pointwise
flat_gt = np.concatenate(all_labels_pointwise)
flat_scores = all_pointwise_scores_arr.flatten()
flat_norm = (flat_scores - flat_scores.min()) / (flat_scores.max() - flat_scores.min() + 1e-10)

fpr_p, tpr_p, thresh_p = roc_curve(flat_gt, flat_norm)
roc_auc_p = auc(fpr_p, tpr_p)
ap_p = average_precision_score(flat_gt, flat_norm)

j_p = tpr_p - fpr_p
opt_p = np.argmax(j_p)
preds_p = (flat_norm >= thresh_p[opt_p]).astype(int)
f1_p = f1_score(flat_gt, preds_p, zero_division=0)

print(f"\n  ═══ REAL-WORLD RESULTS ═══")
print(f"\n  --- Window-Level ---")
print(f"  AUROC:     {roc_auc_w:.4f}")
print(f"  AP:        {ap_w:.4f}")
print(f"  F1:        {f1_w:.4f}")
print(f"  Accuracy:  {acc_w:.4f}")
print(f"  MCC:       {mcc_w:.4f}")
print(f"\n  --- Pointwise ---")
print(f"  AUROC:     {roc_auc_p:.4f}")
print(f"  AP:        {ap_p:.4f}")
print(f"  F1:        {f1_p:.4f}")

# ============================================================================
# 8. Comparison visualizations
# ============================================================================
print("\n[7] Generating comparison visualizations...")

# --- Per-fault-type detection grid ---
fig, axes = plt.subplots(len(fault_generators), 3, figsize=(24, len(fault_generators) * 3.5))
fig.suptitle('DualFreqMamba — Real-World Fault Detection Results per Scenario', 
            fontsize=18, fontweight='bold', y=1.01)

for row, (fault_name, _) in enumerate(fault_generators.items()):
    fault_idx_list = [len(normal_data) + i for i, ft in enumerate(faulty_types) if ft == fault_name]
    idx = fault_idx_list[0]  # first instance
    
    data = all_data[idx]
    gt = all_labels_pointwise[idx]
    ps = all_pointwise_scores[idx]
    
    # Channels
    for ch in range(min(4, N_CH)):
        axes[row, 0].plot(data[ch] / (np.abs(data[ch]).max() + 1e-8), alpha=0.6, linewidth=0.8)
    if gt.sum() > 0:
        fi = np.where(gt == 1)[0]
        axes[row, 0].axvspan(fi[0], fi[-1], alpha=0.2, color='red')
    axes[row, 0].set_title(fault_name.replace('_', ' ').title(), fontsize=12, fontweight='bold', color='red')
    axes[row, 0].set_xlim(0, WIN)
    axes[row, 0].grid(True, alpha=0.2)
    
    # Anomaly scores
    axes[row, 1].fill_between(range(len(ps)), ps, alpha=0.5, color='orange')
    axes[row, 1].plot(ps, 'orange', linewidth=1.5)
    if gt.sum() > 0:
        axes[row, 1].axvspan(fi[0], fi[-1], alpha=0.15, color='red')
    axes[row, 1].set_title(f'Score (max={ps.max():.4f})', fontsize=11)
    axes[row, 1].set_xlim(0, WIN)
    axes[row, 1].grid(True, alpha=0.3)
    
    # FFT
    for ch in range(min(3, N_CH)):
        axes[row, 2].plot(np.abs(np.fft.rfft(data[ch])), alpha=0.7, linewidth=0.8)
    axes[row, 2].set_title('FFT Spectrum', fontsize=11)
    axes[row, 2].grid(True, alpha=0.3)

axes[0, 0].set_ylabel('Signals')
axes[0, 1].set_ylabel('Anomaly Score')
axes[0, 2].set_ylabel('FFT Magnitude')
axes[-1, 0].set_xlabel('Timestep')
axes[-1, 1].set_xlabel('Timestep')
axes[-1, 2].set_xlabel('Freq Bin')

plt.tight_layout()
plt.savefig(f'{BASE}/dualfreqmamba/outputs/comparison/fault_type_grid.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Performance metrics dashboard ---
fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle('DualFreqMamba — Real-World Performance Dashboard', fontsize=18, fontweight='bold', y=1.02)

# ROC
axes[0,0].plot(fpr_w, tpr_w, 'b-', linewidth=2.5, label=f'Window AUROC={roc_auc_w:.3f}')
axes[0,0].plot(fpr_p, tpr_p, 'r-', linewidth=2, alpha=0.7, label=f'Pointwise AUROC={roc_auc_p:.3f}')
axes[0,0].plot([0,1],[0,1], 'k--', alpha=0.3)
axes[0,0].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[0,0].legend(fontsize=11); axes[0,0].grid(True, alpha=0.3)

# Confusion matrix
cm = confusion_matrix(wl, preds_w)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
            xticklabels=['Normal', 'Faulty'], yticklabels=['Normal', 'Faulty'],
            annot_kws={'size': 18})
axes[0,1].set_title(f'Window Confusion (Acc={acc_w:.2%})', fontsize=14, fontweight='bold')

# Score distributions
norm_s = ws_norm[wl == 0]
fault_s = ws_norm[wl == 1]
axes[0,2].hist(norm_s, bins=15, alpha=0.7, color='green', label='Normal', edgecolor='black')
axes[0,2].hist(fault_s, bins=15, alpha=0.7, color='red', label='Faulty', edgecolor='black')
axes[0,2].axvline(thresh_w[opt_w], color='blue', linestyle='--', linewidth=2, label=f'Threshold')
axes[0,2].set_title('Window Score Distribution', fontsize=14, fontweight='bold')
axes[0,2].legend(fontsize=10); axes[0,2].grid(True, alpha=0.3)

# Training curve
axes[1,0].plot(range(1, len(train_losses)+1), train_losses, 'b-o', markersize=2, linewidth=2)
axes[1,0].set_title('Training Convergence', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Epoch'); axes[1,0].grid(True, alpha=0.3)

# Per-fault-type AUROC
per_type_auroc = {}
for ft_name in fault_generators.keys():
    ft_idxs = [len(normal_data) + i for i, ft in enumerate(faulty_types) if ft == ft_name]
    eval_idx = list(range(len(normal_data))) + ft_idxs
    eval_labels = np.array([0]*len(normal_data) + [1]*len(ft_idxs))
    eval_scores = ws_norm[eval_idx]
    fpr_ft, tpr_ft, _ = roc_curve(eval_labels, eval_scores)
    per_type_auroc[ft_name] = auc(fpr_ft, tpr_ft)

faults_sorted = sorted(per_type_auroc.items(), key=lambda x: x[1], reverse=True)
f_names = [f[0].replace('_', '\n') for f in faults_sorted]
f_vals = [f[1] for f in faults_sorted]
bars = axes[1,1].bar(f_names, f_vals, color=plt.cm.RdYlGn(np.array(f_vals)))
for b, v in zip(bars, f_vals):
    axes[1,1].text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=10)
axes[1,1].set_title('AUROC per Fault Type', fontsize=14, fontweight='bold')
axes[1,1].set_ylim(0, 1.15); axes[1,1].grid(True, alpha=0.3, axis='y')

# Summary table
axes[1,2].axis('off')
table_data = [
    ['Metric', 'Window', 'Pointwise'],
    ['AUROC', f'{roc_auc_w:.4f}', f'{roc_auc_p:.4f}'],
    ['AP', f'{ap_w:.4f}', f'{ap_p:.4f}'],
    ['F1', f'{f1_w:.4f}', f'{f1_p:.4f}'],
    ['Accuracy', f'{acc_w:.4f}', '-'],
    ['MCC', f'{mcc_w:.4f}', '-'],
    ['Params', f'{sum(p.numel() for p in model.parameters()):,}', '-'],
]
table = axes[1,2].table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 2.0)
for i in range(3):
    table[0, i].set_facecolor('#1565C0')
    table[0, i].set_text_props(color='white', fontweight='bold')
axes[1,2].set_title('Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{BASE}/dualfreqmamba/outputs/comparison/performance_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Normal vs Faulty side-by-side ---
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle('DualFreqMamba — Normal vs Faulty: Side-by-Side Comparison', fontsize=18, fontweight='bold', y=1.02)

for col in range(4):
    # Normal
    ps_n = all_pointwise_scores[col]
    axes[0, col].fill_between(range(len(ps_n)), ps_n, alpha=0.5, color='green')
    axes[0, col].set_title(f'Normal #{col+1}\nMax Score: {ps_n.max():.4f}', fontsize=11, color='green', fontweight='bold')
    axes[0, col].set_ylim(0, max(all_pointwise_scores_arr.max() * 1.1, 0.01))
    axes[0, col].grid(True, alpha=0.3)
    
    # Faulty
    f_idx = len(normal_data) + col * 3
    ps_f = all_pointwise_scores[f_idx]
    gt_f = all_labels_pointwise[f_idx]
    axes[1, col].fill_between(range(len(ps_f)), ps_f, alpha=0.5, color='red')
    if gt_f.sum() > 0:
        fi = np.where(gt_f == 1)[0]
        axes[1, col].axvspan(fi[0], fi[-1], alpha=0.2, color='yellow')
    axes[1, col].set_title(f'{faulty_types[col*3].replace("_"," ").title()}\nMax: {ps_f.max():.4f}', 
                          fontsize=11, color='red', fontweight='bold')
    axes[1, col].set_ylim(0, max(all_pointwise_scores_arr.max() * 1.1, 0.01))
    axes[1, col].grid(True, alpha=0.3)

axes[0, 0].set_ylabel('Normal\nAnomaly Score', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Faulty\nAnomaly Score', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{BASE}/dualfreqmamba/outputs/comparison/normal_vs_faulty.png', dpi=150, bbox_inches='tight')
plt.close()

# Save metrics
metrics = {
    "test_type": "real_world_industrial",
    "n_channels": N_CH,
    "window_size": WIN,
    "sensor_names": SENSOR_NAMES,
    "n_normal": len(normal_data),
    "n_faulty": len(faulty_data),
    "fault_types": list(fault_generators.keys()),
    "training_epochs": 50,
    "final_training_loss": float(train_losses[-1]),
    "window_level": {"AUROC": float(roc_auc_w), "AP": float(ap_w), "F1": float(f1_w), 
                     "Accuracy": float(acc_w), "MCC": float(mcc_w)},
    "pointwise": {"AUROC": float(roc_auc_p), "AP": float(ap_p), "F1": float(f1_p)},
    "per_fault_auroc": {k: float(v) for k, v in per_type_auroc.items()},
}
with open(f'{BASE}/dualfreqmamba/outputs/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ DualFreqMamba real-world test complete!")
print(f"  inputs/normal:  {len(os.listdir(f'{BASE}/dualfreqmamba/inputs/normal'))} PNGs")
print(f"  inputs/faulty:  {len(os.listdir(f'{BASE}/dualfreqmamba/inputs/faulty'))} PNGs")
print(f"  outputs/normal: {len(os.listdir(f'{BASE}/dualfreqmamba/outputs/normal'))} PNGs")
print(f"  outputs/faulty: {len(os.listdir(f'{BASE}/dualfreqmamba/outputs/faulty'))} PNGs")
print(f"  outputs/comparison: {len(os.listdir(f'{BASE}/dualfreqmamba/outputs/comparison'))} PNGs")
