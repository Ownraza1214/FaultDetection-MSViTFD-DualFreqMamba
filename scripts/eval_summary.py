"""Generate cross-model comparison summary PNG."""
import os, json
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

OUT = "/app/evaluation_results/summary"
sns.set_theme(style="whitegrid", font_scale=1.1)

# Load metrics
with open("/app/evaluation_results/msviTfd/metrics.json") as f:
    img_metrics = json.load(f)
with open("/app/evaluation_results/dualfreqmamba/metrics.json") as f:
    ts_metrics = json.load(f)

# ============================================================================
# PNG: Cross-Model Comparison Dashboard
# ============================================================================
fig = plt.figure(figsize=(28, 20))
fig.suptitle('🔬 Fault Detection Models — Complete Cross-Model Evaluation Summary', 
             fontsize=24, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 4, hspace=0.5, wspace=0.4)

# Model Cards
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.axis('off')
ax1.set_title('📸 MSViTFD — Image Fault Detector', fontsize=16, fontweight='bold', color='#1565C0')
info = (
    f"Architecture: EfficientNet-B2 + Mamba SSM + DWConv\n"
    f"Total Params: {img_metrics['total_params']:,}\n"
    f"Trainable: {img_metrics['trainable_params']:,}\n"
    f"Frozen Encoder: {img_metrics['frozen_params']:,}\n"
    f"Input: Images (B, 3, 256, 256)\n"
    f"Output: Anomaly heatmap + score\n"
    f"Key: Hilbert scanning, Feature-space discriminator,\n"
    f"       Bidirectional feature transfer (L2BT)"
)
ax1.text(0.05, 0.95, info, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.9))

ax2 = fig.add_subplot(gs[0, 2:4])
ax2.axis('off')
ax2.set_title('📈 DualFreqMamba — Time Series Fault Detector', fontsize=16, fontweight='bold', color='#C62828')
info2 = (
    f"Architecture: FFT + CWT + Mamba SSM + Assoc Disc.\n"
    f"Total Params: {ts_metrics['total_params']:,}\n"
    f"Trainable: {ts_metrics['trainable_params']:,}\n"
    f"Input: Multivariate TS (B, {ts_metrics['n_channels']}ch, {ts_metrics['window_size']}t)\n"
    f"Output: Per-timestep anomaly scores + labels\n"
    f"Key: FFT frequency patching, CWT wavelets,\n"
    f"       Gated Mamba decoder, Association discrepancy"
)
ax2.text(0.05, 0.95, info2, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.9))

# Pre-training performance comparison
ax3 = fig.add_subplot(gs[1, 0:2])
metrics_names = ['AUROC', 'AP', 'F1', 'Accuracy', 'MCC']
img_vals = [img_metrics['AUROC'], img_metrics['Average_Precision'], 
            img_metrics['F1_Score'], img_metrics['Accuracy'], img_metrics['MCC']]
ts_vals = [ts_metrics['window_level']['AUROC'], ts_metrics['window_level']['AP'],
           ts_metrics['window_level']['F1'], ts_metrics['window_level']['Accuracy'],
           ts_metrics['window_level']['MCC']]

x = np.arange(len(metrics_names))
width = 0.35
bars1 = ax3.bar(x - width/2, img_vals, width, label='MSViTFD', color='#1976D2', alpha=0.8)
bars2 = ax3.bar(x + width/2, ts_vals, width, label='DualFreqMamba', color='#D32F2F', alpha=0.8)
for b in bars1:
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{b.get_height():.3f}', ha='center', fontsize=9)
for b in bars2:
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{b.get_height():.3f}', ha='center', fontsize=9)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics_names, fontsize=11)
ax3.set_title('Pre-Training Performance Comparison', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.set_ylim(0, 1.15)
ax3.grid(True, alpha=0.3, axis='y')

# Post-training performance comparison  
ax4 = fig.add_subplot(gs[1, 2:4])
img_post = img_metrics['post_training']
ts_post = ts_metrics['post_training']['window']

img_post_vals = [img_post['AUROC'], img_post['AP'], img_post['F1'], img_post['Accuracy'], img_post['MCC']]
ts_post_vals = [ts_post['AUROC'], ts_post['AP'], ts_post['F1'], ts_post['Accuracy'], ts_post['MCC']]

bars1 = ax4.bar(x - width/2, img_post_vals, width, label='MSViTFD', color='#1976D2', alpha=0.8)
bars2 = ax4.bar(x + width/2, ts_post_vals, width, label='DualFreqMamba', color='#D32F2F', alpha=0.8)
for b in bars1:
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{b.get_height():.3f}', ha='center', fontsize=9)
for b in bars2:
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{b.get_height():.3f}', ha='center', fontsize=9)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics_names, fontsize=11)
ax4.set_title('Post-Training Performance Comparison', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.set_ylim(0, 1.15)
ax4.grid(True, alpha=0.3, axis='y')

# Improvement analysis
ax5 = fig.add_subplot(gs[2, 0:2])
img_delta = [img_post_vals[i] - img_vals[i] for i in range(len(metrics_names))]
ts_delta = [ts_post_vals[i] - ts_vals[i] for i in range(len(metrics_names))]

bars1 = ax5.bar(x - width/2, img_delta, width, label='MSViTFD Δ', color='#1976D2', alpha=0.8)
bars2 = ax5.bar(x + width/2, ts_delta, width, label='DualFreqMamba Δ', color='#D32F2F', alpha=0.8)
for b in bars1:
    val = b.get_height()
    ax5.text(b.get_x()+b.get_width()/2, val + (0.01 if val >=0 else -0.03), 
             f'{val:+.3f}', ha='center', fontsize=9)
for b in bars2:
    val = b.get_height()
    ax5.text(b.get_x()+b.get_width()/2, val + (0.01 if val >=0 else -0.03),
             f'{val:+.3f}', ha='center', fontsize=9)
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_names, fontsize=11)
ax5.set_title('Training Improvement (Δ = Post - Pre)', fontsize=14, fontweight='bold')
ax5.legend(fontsize=11)
ax5.axhline(0, color='black', linewidth=0.5)
ax5.grid(True, alpha=0.3, axis='y')

# Parameter efficiency comparison
ax6 = fig.add_subplot(gs[2, 2:4])
models = ['MSViTFD\n(Image)', 'DualFreqMamba\n(Time Series)']
total_params = [img_metrics['total_params']/1e6, ts_metrics['total_params']/1e6]
train_params = [img_metrics['trainable_params']/1e6, ts_metrics['trainable_params']/1e6]

bars1 = ax6.bar(np.arange(2) - 0.2, total_params, 0.35, label='Total', color='#90CAF9')
bars2 = ax6.bar(np.arange(2) + 0.2, train_params, 0.35, label='Trainable', color='#42A5F5')
for b in bars1:
    ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{b.get_height():.2f}M', ha='center', fontsize=11)
for b in bars2:
    ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'{b.get_height():.2f}M', ha='center', fontsize=11)
ax6.set_xticks(np.arange(2))
ax6.set_xticklabels(models, fontsize=12)
ax6.set_ylabel('Parameters (Millions)', fontsize=12)
ax6.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
ax6.legend(fontsize=11)
ax6.grid(True, alpha=0.3, axis='y')

# Full results table
ax7 = fig.add_subplot(gs[3, :])
ax7.axis('off')

table_data = [
    ['Metric', 'MSViTFD\n(Pre)', 'MSViTFD\n(Post)', 'MSViTFD\nΔ', 
     'DualFreqMamba\n(Pre)', 'DualFreqMamba\n(Post)', 'DualFreqMamba\nΔ'],
    ['AUROC', f'{img_vals[0]:.4f}', f'{img_post_vals[0]:.4f}', f'{img_delta[0]:+.4f}',
     f'{ts_vals[0]:.4f}', f'{ts_post_vals[0]:.4f}', f'{ts_delta[0]:+.4f}'],
    ['Avg Precision', f'{img_vals[1]:.4f}', f'{img_post_vals[1]:.4f}', f'{img_delta[1]:+.4f}',
     f'{ts_vals[1]:.4f}', f'{ts_post_vals[1]:.4f}', f'{ts_delta[1]:+.4f}'],
    ['F1 Score', f'{img_vals[2]:.4f}', f'{img_post_vals[2]:.4f}', f'{img_delta[2]:+.4f}',
     f'{ts_vals[2]:.4f}', f'{ts_post_vals[2]:.4f}', f'{ts_delta[2]:+.4f}'],
    ['Accuracy', f'{img_vals[3]:.4f}', f'{img_post_vals[3]:.4f}', f'{img_delta[3]:+.4f}',
     f'{ts_vals[3]:.4f}', f'{ts_post_vals[3]:.4f}', f'{ts_delta[3]:+.4f}'],
    ['MCC', f'{img_vals[4]:.4f}', f'{img_post_vals[4]:.4f}', f'{img_delta[4]:+.4f}',
     f'{ts_vals[4]:.4f}', f'{ts_post_vals[4]:.4f}', f'{ts_delta[4]:+.4f}'],
    ['Total Params', f'{img_metrics["total_params"]:,}', '-', '-',
     f'{ts_metrics["total_params"]:,}', '-', '-'],
    ['Trainable', f'{img_metrics["trainable_params"]:,}', '-', '-',
     f'{ts_metrics["trainable_params"]:,}', '-', '-'],
]

table = ax7.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.0)

# Style header
for i in range(len(table_data[0])):
    table[0, i].set_facecolor('#1565C0')
    table[0, i].set_text_props(color='white', fontweight='bold', fontsize=10)

# Color code improvements
for row in range(1, len(table_data)):
    for col in [3, 6]:  # delta columns
        val = table_data[row][col]
        if val != '-':
            cell = table[row, col]
            if float(val) > 0:
                cell.set_facecolor('#C8E6C9')
            elif float(val) < 0:
                cell.set_facecolor('#FFCDD2')

ax7.set_title('📊 Complete Results Table', fontsize=16, fontweight='bold', pad=25)

plt.savefig(f'{OUT}/00_cross_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 00_cross_model_comparison.png")

# ============================================================================
# Summary README
# ============================================================================
with open(f'/app/evaluation_results/EVALUATION_REPORT.md', 'w') as f:
    f.write("# 🔬 Fault Detection Models — Complete Evaluation Report\n\n")
    f.write("## Models Evaluated\n\n")
    f.write("### 1. MSViTFD — Multi-Scale Vision Transformer Fault Detector (Image)\n")
    f.write(f"- **Parameters:** {img_metrics['total_params']:,} total ({img_metrics['trainable_params']:,} trainable)\n")
    f.write("- **Architecture:** EfficientNet-B2 + Hilbert-Scan Mamba SSM + Multi-Kernel DWConv + Feature Discriminator + L2BT\n")
    f.write("- **Input:** Images (B, 3, 256, 256)\n")
    f.write("- **Innovations:** Hilbert space-filling curve scanning, feature-space anomaly generation, bidirectional feature transfer\n\n")
    
    f.write("### 2. DualFreqMamba — Dual-Branch Frequency-Temporal Fault Detector (Time Series)\n")
    f.write(f"- **Parameters:** {ts_metrics['total_params']:,} total (all trainable)\n")
    f.write("- **Architecture:** FFT Patching + CWT Wavelets + Gated Mamba SSM + Association Discrepancy\n")
    f.write(f"- **Input:** Multivariate TS (B, {ts_metrics['n_channels']}ch, {ts_metrics['window_size']}t)\n")
    f.write("- **Innovations:** Frequency-domain patching, tri-branch adaptive fusion, dual-domain reconstruction, sparse association discrepancy\n\n")
    
    f.write("## Results Summary\n\n")
    f.write("### MSViTFD (Image Fault Detection)\n\n")
    f.write("| Metric | Pre-Training | Post-Training (25 epochs) | Improvement |\n")
    f.write("|--------|-------------|--------------------------|-------------|\n")
    for i, name in enumerate(metrics_names):
        f.write(f"| {name} | {img_vals[i]:.4f} | {img_post_vals[i]:.4f} | {img_delta[i]:+.4f} |\n")
    
    f.write("\n### DualFreqMamba (Time Series Fault Detection)\n\n")
    f.write("| Metric | Pre-Training | Post-Training (30 epochs) | Improvement |\n")
    f.write("|--------|-------------|--------------------------|-------------|\n")
    for i, name in enumerate(metrics_names):
        f.write(f"| {name} | {ts_vals[i]:.4f} | {ts_post_vals[i]:.4f} | {ts_delta[i]:+.4f} |\n")
    
    f.write("\n### DualFreqMamba Pointwise Metrics\n\n")
    f.write("| Metric | Pre-Training | Post-Training |\n")
    f.write("|--------|-------------|---------------|\n")
    pw_pre = ts_metrics['pointwise']
    pw_post = ts_metrics['post_training']['pointwise']
    for k in ['AUROC', 'AP', 'F1']:
        f.write(f"| {k} | {pw_pre[k]:.4f} | {pw_post[k]:.4f} |\n")
    
    f.write("\n## Evaluation Dataset\n\n")
    f.write("- **Image:** 160 synthetic images (80 normal + 80 with faults: scratch, blob, crack, stain)\n")
    f.write("- **Time Series:** 200 windows of 10-channel, 100-timestep data (100 normal + 100 with faults: spike, drift, frequency shift, dropout, noise burst)\n")
    f.write("- **Anomaly rate:** ~11.1% of all timesteps in time series\n\n")
    
    f.write("## Generated Visualizations\n\n")
    f.write("### MSViTFD (12 PNGs)\n")
    for fname in sorted(os.listdir('/app/evaluation_results/msviTfd')):
        if fname.endswith('.png'):
            f.write(f"- `msviTfd/{fname}`\n")
    
    f.write("\n### DualFreqMamba (12 PNGs)\n")
    for fname in sorted(os.listdir('/app/evaluation_results/dualfreqmamba')):
        if fname.endswith('.png'):
            f.write(f"- `dualfreqmamba/{fname}`\n")
    
    f.write("\n### Summary (1 PNG)\n")
    f.write("- `summary/00_cross_model_comparison.png`\n\n")
    
    f.write("## Key Findings\n\n")
    f.write("1. **MSViTFD** shows dramatic improvement with training: AUROC 0.63 → 0.96 (+0.33) in just 25 epochs\n")
    f.write("2. **DualFreqMamba** achieves strong pre-training baselines (0.85 AUROC, 0.92 AP) demonstrating the model architecture itself captures meaningful anomaly patterns\n")
    f.write("3. Both models are **lightweight**: MSViTFD at 7.1M trainable params, DualFreqMamba at just 267K total params\n")
    f.write("4. The **pointwise** AUROC of 0.92 for DualFreqMamba confirms precise temporal fault localization\n")
    f.write("5. Per-fault-type analysis shows varying detection difficulty across anomaly categories\n\n")
    
    f.write("## References\n\n")
    f.write("- MambaAD (arXiv:2404.06564) — Hilbert scanning + Mamba decoder\n")
    f.write("- SimpleNet (CVPR 2023, arXiv:2303.15140) — Feature-space anomaly generation\n")
    f.write("- L2BT (arXiv:2407.04092) — Bidirectional MLP transfer\n")
    f.write("- CATCH (arXiv:2410.12261) — Frequency patching + channel correlation\n")
    f.write("- MAAT (arXiv:2502.07858) — Mamba + sparse attention\n")
    f.write("- TSCMamba (arXiv:2406.04419) — CWT multi-view + Mamba\n")
    f.write("- Anomaly Transformer (ICLR 2022) — Association discrepancy\n")

print("✓ EVALUATION_REPORT.md")
print(f"\n✅ All evaluation files generated!")
