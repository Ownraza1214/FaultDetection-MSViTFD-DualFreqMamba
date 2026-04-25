"""
==========================================================================
MSViTFD — Comprehensive Evaluation Suite
Multi-Scale Vision Transformer Fault Detector
==========================================================================
Generates:
  01_architecture_overview.png
  02_synthetic_fault_samples.png
  03_anomaly_heatmaps.png
  04_roc_curve.png
  05_precision_recall_curve.png
  06_confusion_matrix.png
  07_score_distributions.png
  08_loss_landscape.png
  09_parameter_breakdown.png
  10_multi_scale_features.png
  11_training_convergence.png
  12_per_class_performance.png
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
    matthews_corrcoef, cohen_kappa_score
)
import json
import time

from msviTfd.configuration_msviTfd import MSViTFDConfig
from msviTfd.modeling_msviTfd import MSViTFDModel

OUT = "/app/evaluation_results/msviTfd"
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# 1. Create model (CPU-optimized config for testing)
# ============================================================================
print("=" * 70)
print("MSViTFD — COMPREHENSIVE EVALUATION")
print("=" * 70)

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

model = MSViTFDModel(config)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
print(f"Model loaded: {total_params:,} params ({trainable_params:,} trainable)")

# ============================================================================
# 2. Generate synthetic fault data
# ============================================================================
print("\n[1/12] Generating synthetic fault dataset...")

np.random.seed(42)
torch.manual_seed(42)

N_NORMAL = 80
N_ANOMALY = 80
IMG_SIZE = 64

def generate_normal_image():
    """Generate clean textured surface."""
    img = np.random.randn(3, IMG_SIZE, IMG_SIZE) * 0.1 + 0.5
    # Add smooth texture
    for c in range(3):
        freq = np.random.uniform(2, 8)
        phase = np.random.uniform(0, 2*np.pi)
        x = np.linspace(0, freq*np.pi, IMG_SIZE)
        y = np.linspace(0, freq*np.pi, IMG_SIZE)
        xx, yy = np.meshgrid(x, y)
        img[c] += 0.15 * np.sin(xx + phase) * np.cos(yy + phase)
    return np.clip(img, 0, 1).astype(np.float32)

def generate_fault_image(fault_type='scratch'):
    """Generate image with synthetic fault."""
    img = generate_normal_image()
    if fault_type == 'scratch':
        c = np.random.randint(0, IMG_SIZE)
        width = np.random.randint(1, 4)
        img[:, max(0,c-width):c+width, :] *= 0.2
    elif fault_type == 'blob':
        cx, cy = np.random.randint(10, IMG_SIZE-10, 2)
        r = np.random.randint(3, 10)
        yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        mask = ((xx - cx)**2 + (yy - cy)**2) < r**2
        img[:, mask] = np.random.uniform(0, 0.3, size=(3, mask.sum()))
    elif fault_type == 'crack':
        y = np.random.randint(5, IMG_SIZE-5)
        for x in range(IMG_SIZE):
            dy = int(3 * np.sin(x * 0.3))
            yy = np.clip(y + dy, 0, IMG_SIZE-1)
            img[:, yy, x] *= 0.1
    elif fault_type == 'stain':
        cx, cy = np.random.randint(10, IMG_SIZE-10, 2)
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                d = np.sqrt((i-cy)**2 + (j-cx)**2)
                if d < 12:
                    img[:, i, j] += 0.4 * np.exp(-d**2/30)
        img = np.clip(img, 0, 1)
    return img.astype(np.float32), fault_type

fault_types = ['scratch', 'blob', 'crack', 'stain']
normal_images = [generate_normal_image() for _ in range(N_NORMAL)]
fault_images = []
fault_labels_detail = []
for i in range(N_ANOMALY):
    ft = fault_types[i % len(fault_types)]
    img, label = generate_fault_image(ft)
    fault_images.append(img)
    fault_labels_detail.append(label)

all_images = normal_images + fault_images
all_labels = [0]*N_NORMAL + [1]*N_ANOMALY
all_labels = np.array(all_labels)

print(f"  Normal: {N_NORMAL}, Anomaly: {N_ANOMALY}")
print(f"  Fault types: {dict(zip(*np.unique(fault_labels_detail, return_counts=True)))}")

# ============================================================================
# 3. PNG 02: Synthetic Fault Samples
# ============================================================================
print("[2/12] Generating synthetic fault sample visualization...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('MSViTFD — Synthetic Fault Dataset Samples', fontsize=18, fontweight='bold', y=1.02)

# Normal samples
for i in range(5):
    ax = axes[0, i]
    ax.imshow(normal_images[i*10].transpose(1, 2, 0))
    ax.set_title(f'Normal #{i+1}', fontsize=12, color='green', fontweight='bold')
    ax.axis('off')

# Fault samples
for i, ft in enumerate(fault_types):
    idx = fault_labels_detail.index(ft)
    ax = axes[1, i]
    ax.imshow(fault_images[idx].transpose(1, 2, 0))
    ax.set_title(f'Fault: {ft.capitalize()}', fontsize=12, color='red', fontweight='bold')
    ax.axis('off')

ax = axes[1, 4]
ax.imshow(fault_images[-1].transpose(1, 2, 0))
ax.set_title(f'Fault: {fault_labels_detail[-1].capitalize()}', fontsize=12, color='red', fontweight='bold')
ax.axis('off')

axes[0, 0].set_ylabel('NORMAL', fontsize=14, fontweight='bold', color='green', rotation=0, labelpad=60)
axes[1, 0].set_ylabel('FAULTY', fontsize=14, fontweight='bold', color='red', rotation=0, labelpad=60)

plt.tight_layout()
plt.savefig(f'{OUT}/02_synthetic_fault_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_synthetic_fault_samples.png")

# ============================================================================
# 4. Run inference on all samples
# ============================================================================
print("[3/12] Running inference on all samples...")
model.eval()

all_scores = []
all_anomaly_maps = []
all_recon_losses = []
all_disc_losses = []
all_fbft_losses = []

batch_size = 8
for i in range(0, len(all_images), batch_size):
    batch_imgs = all_images[i:i+batch_size]
    batch_tensor = torch.tensor(np.stack(batch_imgs))
    
    with torch.no_grad():
        output = model(batch_tensor, return_anomaly_map=True)
    
    if output.anomaly_score is not None:
        all_scores.extend(output.anomaly_score.cpu().numpy().tolist())
    if output.anomaly_map is not None:
        all_anomaly_maps.extend(output.anomaly_map.cpu().numpy())
    all_recon_losses.append(output.reconstruction_loss.item())
    if output.discriminator_loss is not None:
        all_disc_losses.append(output.discriminator_loss.item())
    if output.fbft_loss is not None:
        all_fbft_losses.append(output.fbft_loss.item())

all_scores = np.array(all_scores)
print(f"  Scores: min={all_scores.min():.6f}, max={all_scores.max():.6f}, mean={all_scores.mean():.6f}")

# ============================================================================
# 5. PNG 03: Anomaly Heatmaps
# ============================================================================
print("[4/12] Generating anomaly heatmaps...")

fig, axes = plt.subplots(3, 6, figsize=(24, 12))
fig.suptitle('MSViTFD — Anomaly Detection Heatmaps', fontsize=18, fontweight='bold', y=1.02)

# Pick samples: 3 normal, 3 anomaly
sample_idx_normal = [0, 20, 40]
sample_idx_anomaly = [N_NORMAL, N_NORMAL+5, N_NORMAL+10]
sample_indices = sample_idx_normal + sample_idx_anomaly
titles = ['Normal #1', 'Normal #2', 'Normal #3', 
          f'Fault: {fault_labels_detail[0]}', f'Fault: {fault_labels_detail[5]}', f'Fault: {fault_labels_detail[10]}']

for col, (idx, title) in enumerate(zip(sample_indices, titles)):
    # Original image
    axes[0, col].imshow(all_images[idx].transpose(1, 2, 0))
    axes[0, col].set_title(title, fontsize=11, fontweight='bold',
                           color='green' if idx < N_NORMAL else 'red')
    axes[0, col].axis('off')
    
    # Anomaly heatmap
    if idx < len(all_anomaly_maps):
        hmap = all_anomaly_maps[idx]
        im = axes[1, col].imshow(hmap, cmap='hot', interpolation='bilinear')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046)
    axes[1, col].set_title(f'Score: {all_scores[idx]:.2e}', fontsize=10)
    axes[1, col].axis('off')
    
    # Overlay
    if idx < len(all_anomaly_maps):
        hmap_resized = np.array(
            plt.cm.hot(hmap / (hmap.max() + 1e-10))[:, :, :3]
        )
        orig_resized = np.array(
            F.interpolate(
                torch.tensor(all_images[idx]).unsqueeze(0),
                size=hmap.shape, mode='bilinear', align_corners=False
            ).squeeze().permute(1,2,0).numpy()
        )
        overlay = 0.6 * orig_resized + 0.4 * hmap_resized
        axes[2, col].imshow(np.clip(overlay, 0, 1))
    axes[2, col].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=13, fontweight='bold', rotation=0, labelpad=50)
axes[1, 0].set_ylabel('Heatmap', fontsize=13, fontweight='bold', rotation=0, labelpad=50)
axes[2, 0].set_ylabel('Overlay', fontsize=13, fontweight='bold', rotation=0, labelpad=50)

plt.tight_layout()
plt.savefig(f'{OUT}/03_anomaly_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_anomaly_heatmaps.png")

# ============================================================================
# 6. Compute full metrics
# ============================================================================
print("[5/12] Computing evaluation metrics...")

# Normalize scores for better thresholding
scores_norm = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min() + 1e-10)

# ROC
fpr, tpr, roc_thresholds = roc_curve(all_labels, scores_norm)
roc_auc = auc(fpr, tpr)

# Precision-Recall
precision, recall, pr_thresholds = precision_recall_curve(all_labels, scores_norm)
ap = average_precision_score(all_labels, scores_norm)

# Find optimal threshold (Youden's J)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = roc_thresholds[optimal_idx]

# Binary predictions at optimal threshold
preds = (scores_norm >= optimal_threshold).astype(int)
acc = accuracy_score(all_labels, preds)
f1 = f1_score(all_labels, preds)
mcc = matthews_corrcoef(all_labels, preds)
kappa = cohen_kappa_score(all_labels, preds)

# Per-class F1
f1_normal = f1_score(all_labels, preds, pos_label=0)
f1_anomaly = f1_score(all_labels, preds, pos_label=1)

metrics = {
    "model": "MSViTFD",
    "total_params": total_params,
    "trainable_params": trainable_params,
    "frozen_params": frozen_params,
    "n_samples": len(all_labels),
    "n_normal": int(N_NORMAL),
    "n_anomaly": int(N_ANOMALY),
    "AUROC": float(roc_auc),
    "Average_Precision": float(ap),
    "Optimal_Threshold": float(optimal_threshold),
    "Accuracy": float(acc),
    "F1_Score": float(f1),
    "F1_Normal": float(f1_normal),
    "F1_Anomaly": float(f1_anomaly),
    "MCC": float(mcc),
    "Cohen_Kappa": float(kappa),
    "Sensitivity_TPR": float(tpr[optimal_idx]),
    "Specificity_1-FPR": float(1 - fpr[optimal_idx]),
    "classification_report": classification_report(all_labels, preds, target_names=['Normal', 'Anomaly']),
}

print(f"\n  ═══ MSViTFD EVALUATION RESULTS ═══")
print(f"  AUROC:              {roc_auc:.4f}")
print(f"  Average Precision:  {ap:.4f}")
print(f"  Accuracy:           {acc:.4f}")
print(f"  F1 Score:           {f1:.4f}")
print(f"  MCC:                {mcc:.4f}")
print(f"  Cohen's Kappa:      {kappa:.4f}")
print(f"  Sensitivity:        {tpr[optimal_idx]:.4f}")
print(f"  Specificity:        {1-fpr[optimal_idx]:.4f}")

with open(f'{OUT}/metrics.json', 'w') as f:
    json.dump({k: v for k, v in metrics.items() if k != 'classification_report'}, f, indent=2)

# ============================================================================
# 7. PNG 04: ROC Curve
# ============================================================================
print("[6/12] Generating ROC curve...")

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'MSViTFD (AUROC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12, 
        label=f'Optimal (t={optimal_threshold:.3f})')

# Fill area under curve
ax.fill_between(fpr, tpr, alpha=0.15, color='blue')

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14)
ax.set_title('MSViTFD — Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

# Add metrics box
textstr = f'AUROC: {roc_auc:.4f}\nAP: {ap:.4f}\nF1: {f1:.4f}\nAcc: {acc:.4f}'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.6, 0.25, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

plt.savefig(f'{OUT}/04_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 04_roc_curve.png")

# ============================================================================
# 8. PNG 05: Precision-Recall Curve
# ============================================================================
print("[7/12] Generating precision-recall curve...")

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(recall, precision, 'g-', linewidth=2.5, label=f'MSViTFD (AP = {ap:.4f})')
ax.axhline(y=N_ANOMALY/(N_NORMAL+N_ANOMALY), color='k', linestyle='--', alpha=0.5, label='No-Skill Baseline')
ax.fill_between(recall, precision, alpha=0.15, color='green')

ax.set_xlabel('Recall', fontsize=14)
ax.set_ylabel('Precision', fontsize=14)
ax.set_title('MSViTFD — Precision-Recall Curve', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

plt.savefig(f'{OUT}/05_precision_recall_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 05_precision_recall_curve.png")

# ============================================================================
# 9. PNG 06: Confusion Matrix
# ============================================================================
print("[8/12] Generating confusion matrix...")

cm = confusion_matrix(all_labels, preds)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Absolute counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
            annot_kws={'size': 20})
ax1.set_xlabel('Predicted', fontsize=14)
ax1.set_ylabel('Actual', fontsize=14)
ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

# Normalized
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax2,
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
            annot_kws={'size': 20}, vmin=0, vmax=1)
ax2.set_xlabel('Predicted', fontsize=14)
ax2.set_ylabel('Actual', fontsize=14)
ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

fig.suptitle('MSViTFD — Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/06_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 06_confusion_matrix.png")

# ============================================================================
# 10. PNG 07: Score Distributions
# ============================================================================
print("[9/12] Generating score distributions...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Histogram
normal_scores = scores_norm[all_labels == 0]
anomaly_scores = scores_norm[all_labels == 1]

ax1.hist(normal_scores, bins=30, alpha=0.7, color='green', label=f'Normal (n={N_NORMAL})', edgecolor='black')
ax1.hist(anomaly_scores, bins=30, alpha=0.7, color='red', label=f'Anomaly (n={N_ANOMALY})', edgecolor='black')
ax1.axvline(x=optimal_threshold, color='blue', linestyle='--', linewidth=2, 
            label=f'Threshold = {optimal_threshold:.3f}')
ax1.set_xlabel('Normalized Anomaly Score', fontsize=14)
ax1.set_ylabel('Count', fontsize=14)
ax1.set_title('Score Distribution by Class', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Box plot
data = [normal_scores, anomaly_scores]
bp = ax2.boxplot(data, labels=['Normal', 'Anomaly'], patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightsalmon')
ax2.set_ylabel('Normalized Anomaly Score', fontsize=14)
ax2.set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add stats
stats_text = (f"Normal: μ={normal_scores.mean():.4f}, σ={normal_scores.std():.4f}\n"
              f"Anomaly: μ={anomaly_scores.mean():.4f}, σ={anomaly_scores.std():.4f}\n"
              f"Separability: {abs(anomaly_scores.mean()-normal_scores.mean())/max(normal_scores.std()+anomaly_scores.std(), 1e-8):.4f}")
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.suptitle('MSViTFD — Anomaly Score Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/07_score_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 07_score_distributions.png")

# ============================================================================
# 11. PNG 08: Training Convergence (simulated training loop)
# ============================================================================
print("[10/12] Simulating training convergence...")

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

train_losses = []
recon_losses_hist = []
disc_losses_hist = []
fbft_losses_hist = []

n_epochs = 25
for epoch in range(n_epochs):
    epoch_losses = []
    epoch_recon = []
    epoch_disc = []
    epoch_fbft = []
    
    # Mini-batches from normal images only (anomaly detection paradigm)
    indices = np.random.permutation(N_NORMAL)
    for i in range(0, N_NORMAL, batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_tensor = torch.tensor(np.stack([normal_images[j] for j in batch_idx]))
        
        output = model(batch_tensor, return_anomaly_map=False)
        loss = output.loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_losses.append(loss.item())
        epoch_recon.append(output.reconstruction_loss.item())
        if output.discriminator_loss is not None:
            epoch_disc.append(output.discriminator_loss.item())
        if output.fbft_loss is not None:
            epoch_fbft.append(output.fbft_loss.item())
    
    train_losses.append(np.mean(epoch_losses))
    recon_losses_hist.append(np.mean(epoch_recon))
    disc_losses_hist.append(np.mean(epoch_disc) if epoch_disc else 0)
    fbft_losses_hist.append(np.mean(epoch_fbft) if epoch_fbft else 0)
    
    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}: loss={train_losses[-1]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('MSViTFD — Training Convergence', fontsize=18, fontweight='bold', y=1.02)

epochs = range(1, n_epochs+1)

axes[0,0].plot(epochs, train_losses, 'b-o', linewidth=2, markersize=4, label='Total Loss')
axes[0,0].set_title('Total Training Loss', fontsize=13, fontweight='bold')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(epochs, recon_losses_hist, 'r-s', linewidth=2, markersize=4, label='Reconstruction Loss')
axes[0,1].set_title('Reconstruction Loss (MSE)', fontsize=13, fontweight='bold')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(epochs, disc_losses_hist, 'g-^', linewidth=2, markersize=4, label='Discriminator Loss')
axes[1,0].set_title('Discriminator Loss (Truncated L1)', fontsize=13, fontweight='bold')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Loss')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(epochs, fbft_losses_hist, 'm-D', linewidth=2, markersize=4, label='FBFT Loss')
axes[1,1].set_title('Bidirectional Feature Transfer Loss', fontsize=13, fontweight='bold')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Loss')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/08_training_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 08_training_convergence.png")

# ============================================================================
# 12. Re-evaluate after training
# ============================================================================
print("[11/12] Re-evaluating after training...")
model.eval()

trained_scores = []
trained_maps = []
for i in range(0, len(all_images), batch_size):
    batch_imgs = all_images[i:i+batch_size]
    batch_tensor = torch.tensor(np.stack(batch_imgs))
    with torch.no_grad():
        output = model(batch_tensor, return_anomaly_map=True)
    if output.anomaly_score is not None:
        trained_scores.extend(output.anomaly_score.cpu().numpy().tolist())
    if output.anomaly_map is not None:
        trained_maps.extend(output.anomaly_map.cpu().numpy())

trained_scores = np.array(trained_scores)
trained_norm = (trained_scores - trained_scores.min()) / (trained_scores.max() - trained_scores.min() + 1e-10)

fpr2, tpr2, thresh2 = roc_curve(all_labels, trained_norm)
roc_auc2 = auc(fpr2, tpr2)
ap2 = average_precision_score(all_labels, trained_norm)

j2 = tpr2 - fpr2
opt2 = np.argmax(j2)
preds2 = (trained_norm >= thresh2[opt2]).astype(int)
acc2 = accuracy_score(all_labels, preds2)
f1_2 = f1_score(all_labels, preds2)
mcc2 = matthews_corrcoef(all_labels, preds2)

print(f"\n  ═══ POST-TRAINING RESULTS ═══")
print(f"  AUROC:    {roc_auc:.4f} → {roc_auc2:.4f}")
print(f"  AP:       {ap:.4f} → {ap2:.4f}")
print(f"  F1:       {f1:.4f} → {f1_2:.4f}")
print(f"  Accuracy: {acc:.4f} → {acc2:.4f}")
print(f"  MCC:      {mcc:.4f} → {mcc2:.4f}")

metrics["post_training"] = {
    "AUROC": float(roc_auc2),
    "AP": float(ap2),
    "F1": float(f1_2),
    "Accuracy": float(acc2),
    "MCC": float(mcc2),
}

with open(f'{OUT}/metrics.json', 'w') as f:
    json.dump({k: v for k, v in metrics.items() if k != 'classification_report'}, f, indent=2)

# ============================================================================
# 13. PNG 09: Parameter Breakdown
# ============================================================================
print("[12/12] Generating parameter breakdown...")

param_groups = {}
for name, param in model.named_parameters():
    group = name.split('.')[0]
    if group not in param_groups:
        param_groups[group] = {'total': 0, 'trainable': 0}
    param_groups[group]['total'] += param.numel()
    if param.requires_grad:
        param_groups[group]['trainable'] += param.numel()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('MSViTFD — Model Parameter Analysis', fontsize=18, fontweight='bold', y=1.02)

# Pie chart
sizes = [v['total'] for v in param_groups.values()]
labels = [f"{k}\n({v['total']/1e6:.2f}M)" for k, v in param_groups.items()]
colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
                                     textprops={'fontsize': 10}, startangle=90)
ax1.set_title('Parameter Distribution by Component', fontsize=13, fontweight='bold')

# Bar chart: trainable vs frozen
groups = list(param_groups.keys())
trainable_vals = [param_groups[g]['trainable']/1e6 for g in groups]
frozen_vals = [(param_groups[g]['total'] - param_groups[g]['trainable'])/1e6 for g in groups]

x = np.arange(len(groups))
width = 0.35
ax2.bar(x - width/2, trainable_vals, width, label='Trainable', color='steelblue')
ax2.bar(x + width/2, frozen_vals, width, label='Frozen', color='lightcoral')
ax2.set_xlabel('Component', fontsize=12)
ax2.set_ylabel('Parameters (Millions)', fontsize=12)
ax2.set_title('Trainable vs Frozen Parameters', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(groups, rotation=45, ha='right', fontsize=10)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT}/09_parameter_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 09_parameter_breakdown.png")

# ============================================================================
# PNG 01: Architecture Overview Diagram
# ============================================================================
print("\nGenerating architecture diagram...")

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')
ax.set_title('MSViTFD — Architecture Overview', fontsize=22, fontweight='bold', pad=20)

def draw_box(ax, x, y, w, h, text, color='lightblue', fontsize=9, text_color='black'):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, 
            fontweight='bold', color=text_color, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Input
draw_box(ax, 7, 12.5, 6, 0.8, 'Input Image (B, 3, 256, 256)', '#FFE0B2', 11)

# Encoder
draw_box(ax, 6, 10.8, 8, 1.2, 'EfficientNet-B2 Encoder (Frozen)\nF1(48ch) → F2(120ch) → F3(352ch)', '#BBDEFB', 10)
draw_arrow(ax, 10, 12.5, 10, 12.0)

# Feature Pyramid  
draw_box(ax, 1, 9.2, 5, 0.8, 'Scale 1: 48→256\n1×1 Conv + BN', '#C8E6C9', 9)
draw_box(ax, 7.5, 9.2, 5, 0.8, 'Scale 2: 120→256\n1×1 Conv + BN', '#C8E6C9', 9)
draw_box(ax, 14, 9.2, 5, 0.8, 'Scale 3: 352→256\n1×1 Conv + BN', '#C8E6C9', 9)
draw_arrow(ax, 7, 10.8, 3.5, 10.0)
draw_arrow(ax, 10, 10.8, 10, 10.0)
draw_arrow(ax, 13, 10.8, 16.5, 10.0)

# LSS Decoder
draw_box(ax, 1, 6.8, 5, 2.0, 'LSS Block ×2\n├ Hilbert Scan 2D→1D\n├ Mamba SSM (global)\n├ DWConv [3,5] (local)\n└ Gated Fusion', '#E1BEE7', 8)
draw_box(ax, 7.5, 6.8, 5, 2.0, 'LSS Block ×3\n├ Hilbert Scan 2D→1D\n├ Mamba SSM (global)\n├ DWConv [3,5,7] (local)\n└ Gated Fusion', '#E1BEE7', 8)
draw_box(ax, 14, 6.8, 5, 2.0, 'LSS Block ×2\n├ Hilbert Scan 2D→1D\n├ Mamba SSM (global)\n├ DWConv [3,5,7] (local)\n└ Gated Fusion', '#E1BEE7', 8)
draw_arrow(ax, 3.5, 9.2, 3.5, 8.8)
draw_arrow(ax, 10, 9.2, 10, 8.8)
draw_arrow(ax, 16.5, 9.2, 16.5, 8.8)

# Cross-scale attention arrows
ax.annotate('', xy=(7.5, 7.8), xytext=(6.0, 7.8),
            arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, ls='--'))
ax.annotate('', xy=(14.0, 7.8), xytext=(12.5, 7.8),
            arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, ls='--'))
ax.text(6.7, 8.1, 'cross-scale\nattn', fontsize=7, color='purple', ha='center')

# Reconstruction + Discriminator
draw_box(ax, 2, 4.5, 7, 1.8, 'Multi-Scale Reconstruction\nMSE(decoded, encoder_features)\n\nFeature-Space Discriminator\nnormal + Gaussian noise → binary score', '#FFCDD2', 9)
draw_box(ax, 11, 4.5, 7, 1.8, 'Bidirectional Feature Transfer\nForward MLP: F_low → F_high\nBackward MLP: F_high → F_low\nCosine Similarity Loss', '#FFF9C4', 9)
draw_arrow(ax, 5, 6.8, 5, 6.3)
draw_arrow(ax, 15, 6.8, 15, 6.3)

# Output
draw_box(ax, 4, 2.5, 12, 1.5, 'OUTPUT\n• Anomaly Map: (B, H, W) — per-pixel anomaly heatmap\n• Anomaly Score: (B,) — image-level max score\n• Loss = 1.0×Recon + 0.5×Discriminator + 0.3×FBFT', '#B2DFDB', 10, 'darkgreen')
draw_arrow(ax, 5.5, 4.5, 8, 4.0)
draw_arrow(ax, 14.5, 4.5, 12, 4.0)

# Stats box
stats = f"Total: {total_params/1e6:.1f}M params | Trainable: {trainable_params/1e6:.1f}M | Frozen Encoder: {frozen_params/1e6:.1f}M"
ax.text(10, 1.8, stats, ha='center', fontsize=11, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(f'{OUT}/01_architecture_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 01_architecture_overview.png")

# ============================================================================
# PNG 10: Per-Class Performance
# ============================================================================
print("Generating per-class performance...")

# Evaluate per fault type
per_class_metrics = {}
for ft in fault_types:
    ft_indices = [N_NORMAL + i for i, l in enumerate(fault_labels_detail) if l == ft]
    normal_indices = list(range(N_NORMAL))
    
    eval_indices = normal_indices + ft_indices
    eval_labels = np.array([0]*len(normal_indices) + [1]*len(ft_indices))
    eval_scores = trained_norm[eval_indices]
    
    fpr_ft, tpr_ft, _ = roc_curve(eval_labels, eval_scores)
    auc_ft = auc(fpr_ft, tpr_ft)
    ap_ft = average_precision_score(eval_labels, eval_scores)
    
    per_class_metrics[ft] = {
        'AUROC': auc_ft,
        'AP': ap_ft,
        'n_samples': len(ft_indices),
        'fpr': fpr_ft,
        'tpr': tpr_ft,
    }

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('MSViTFD — Per-Fault-Type Performance', fontsize=18, fontweight='bold', y=1.02)

# ROC curves per class
colors_ft = {'scratch': '#FF6B6B', 'blob': '#4ECDC4', 'crack': '#45B7D1', 'stain': '#96CEB4'}
for ft, m in per_class_metrics.items():
    ax1.plot(m['fpr'], m['tpr'], linewidth=2, color=colors_ft[ft],
             label=f'{ft.capitalize()} (AUROC={m["AUROC"]:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax1.set_xlabel('False Positive Rate', fontsize=13)
ax1.set_ylabel('True Positive Rate', fontsize=13)
ax1.set_title('ROC per Fault Type', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Bar chart
faults = list(per_class_metrics.keys())
aurocs = [per_class_metrics[ft]['AUROC'] for ft in faults]
aps = [per_class_metrics[ft]['AP'] for ft in faults]

x = np.arange(len(faults))
width = 0.35
bars1 = ax2.bar(x - width/2, aurocs, width, label='AUROC', color='steelblue')
bars2 = ax2.bar(x + width/2, aps, width, label='Average Precision', color='coral')

for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', fontsize=10)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', fontsize=10)

ax2.set_xlabel('Fault Type', fontsize=13)
ax2.set_ylabel('Score', fontsize=13)
ax2.set_title('AUROC & AP per Fault Type', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([ft.capitalize() for ft in faults], fontsize=12)
ax2.legend(fontsize=11)
ax2.set_ylim(0, 1.15)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT}/10_per_class_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 10_per_class_performance.png")

# ============================================================================
# PNG 11: Post-Training Heatmaps Comparison
# ============================================================================
print("Generating post-training heatmaps comparison...")

fig, axes = plt.subplots(2, 6, figsize=(24, 8))
fig.suptitle('MSViTFD — Post-Training Anomaly Heatmaps', fontsize=18, fontweight='bold', y=1.02)

for col, (idx, title) in enumerate(zip(sample_indices, titles)):
    axes[0, col].imshow(all_images[idx].transpose(1, 2, 0))
    axes[0, col].set_title(title, fontsize=11, fontweight='bold',
                           color='green' if idx < N_NORMAL else 'red')
    axes[0, col].axis('off')
    
    if idx < len(trained_maps):
        hmap = trained_maps[idx]
        im = axes[1, col].imshow(hmap, cmap='hot', interpolation='bilinear')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046)
        axes[1, col].set_title(f'Score: {trained_scores[idx]:.2e}', fontsize=10)
    axes[1, col].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=13, fontweight='bold', rotation=0, labelpad=50)
axes[1, 0].set_ylabel('Heatmap', fontsize=13, fontweight='bold', rotation=0, labelpad=50)

plt.tight_layout()
plt.savefig(f'{OUT}/11_post_training_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 11_post_training_heatmaps.png")

# ============================================================================
# PNG 12: Comprehensive Metrics Dashboard
# ============================================================================
print("Generating metrics dashboard...")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('MSViTFD — Complete Evaluation Dashboard', fontsize=20, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35)

# Pre vs Post ROC
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(fpr, tpr, 'b--', linewidth=1.5, alpha=0.5, label=f'Pre-train (AUROC={roc_auc:.3f})')
ax1.plot(fpr2, tpr2, 'b-', linewidth=2.5, label=f'Post-train (AUROC={roc_auc2:.3f})')
ax1.plot([0,1],[0,1],'k--', alpha=0.3)
ax1.set_title('ROC: Pre vs Post Training', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Score distribution post-training
ax2 = fig.add_subplot(gs[0, 2:4])
trained_normal = trained_norm[all_labels == 0]
trained_anomaly = trained_norm[all_labels == 1]
ax2.hist(trained_normal, bins=25, alpha=0.7, color='green', label='Normal', edgecolor='black')
ax2.hist(trained_anomaly, bins=25, alpha=0.7, color='red', label='Anomaly', edgecolor='black')
ax2.set_title('Post-Training Score Distribution', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Training loss
ax3 = fig.add_subplot(gs[1, 0:2])
ax3.plot(epochs, train_losses, 'b-o', markersize=3, linewidth=2)
ax3.set_title('Training Loss Convergence', fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.grid(True, alpha=0.3)

# Per-class AUROC
ax4 = fig.add_subplot(gs[1, 2:4])
faults = list(per_class_metrics.keys())
aurocs = [per_class_metrics[ft]['AUROC'] for ft in faults]
bars = ax4.bar([ft.capitalize() for ft in faults], aurocs, color=[colors_ft[ft] for ft in faults])
for bar, val in zip(bars, aurocs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=10)
ax4.set_title('AUROC per Fault Type', fontweight='bold')
ax4.set_ylim(0, 1.1)
ax4.grid(True, alpha=0.3, axis='y')

# Confusion matrix post-training
ax5 = fig.add_subplot(gs[2, 0:2])
cm2 = confusion_matrix(all_labels, preds2)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
            annot_kws={'size': 16})
ax5.set_title('Post-Training Confusion Matrix', fontweight='bold')

# Metrics summary table
ax6 = fig.add_subplot(gs[2, 2:4])
ax6.axis('off')
table_data = [
    ['Metric', 'Pre-Training', 'Post-Training'],
    ['AUROC', f'{roc_auc:.4f}', f'{roc_auc2:.4f}'],
    ['Avg Precision', f'{ap:.4f}', f'{ap2:.4f}'],
    ['F1 Score', f'{f1:.4f}', f'{f1_2:.4f}'],
    ['Accuracy', f'{acc:.4f}', f'{acc2:.4f}'],
    ['MCC', f'{mcc:.4f}', f'{mcc2:.4f}'],
    ['Parameters', f'{total_params:,}', '-'],
    ['Trainable', f'{trainable_params:,}', '-'],
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

# Save final metrics
with open(f'{OUT}/metrics.json', 'w') as f:
    json.dump({k: v for k, v in metrics.items() if k != 'classification_report'}, f, indent=2)

with open(f'{OUT}/classification_report.txt', 'w') as f:
    f.write("MSViTFD — Classification Report\n")
    f.write("=" * 50 + "\n\n")
    f.write("PRE-TRAINING:\n")
    f.write(classification_report(all_labels, preds, target_names=['Normal', 'Anomaly']))
    f.write("\n\nPOST-TRAINING:\n")
    f.write(classification_report(all_labels, preds2, target_names=['Normal', 'Anomaly']))

print(f"\n✅ MSViTFD evaluation complete! {len(os.listdir(OUT))} files in {OUT}")
print("Files:", sorted(os.listdir(OUT)))
