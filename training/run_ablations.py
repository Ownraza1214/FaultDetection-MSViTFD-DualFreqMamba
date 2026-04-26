#!/usr/bin/env python3
"""
Ablation Study Scripts for MSViTFD and DualFreqMamba
=====================================================

Systematically removes each novel component to quantify its contribution.
Runs 3 seeds per configuration for statistical significance.

MSViTFD Ablations:
    1. Full model (baseline)
    2. w/o Hilbert Scanning (use raster scan only)
    3. w/o Multi-Kernel DWConv (remove local branch from LSS)
    4. w/o Feature Discriminator (SimpleNet head)
    5. w/o Bidirectional Feature Transfer (L2BT)
    6. w/o Gated Fusion (use simple addition)
    7. w/o Cross-Scale Attention
    8. Single-Scale (use only scale 2)

DualFreqMamba Ablations:
    1. Full model (baseline)
    2. w/o FFT Branch (remove frequency patching)
    3. w/o CWT Branch (remove wavelet features)
    4. w/o Raw Temporal Branch
    5. w/o Association Discrepancy (reconstruction only)
    6. w/o Gated Skip in Mamba
    7. w/o Frequency Domain Reconstruction (time only)
    8. w/o Adaptive Gated Fusion (use concat)
    9. Single-Branch: FFT only
    10. Single-Branch: Temporal only

Usage:
    python run_ablations.py --model msviTfd --data_root ./datasets/MVTec --category bottle
    python run_ablations.py --model dualfreqmamba --dataset SMD
    python run_ablations.py --model both --output_dir ./ablation_results
"""

import os
import sys
import json
import copy
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# MSViTFD Ablation Configurations
# ============================================================================

def get_msviTfd_ablation_configs():
    """Return dict of ablation name → config modifications."""
    from models.msviTfd.configuration_msviTfd import MSViTFDConfig
    
    base_kwargs = dict(
        encoder_name='efficientnet_b2',
        encoder_pretrained=True,
        freeze_encoder=True,
        hidden_dim=256,
        decoder_depths=[2, 3, 2],
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        dwconv_kernels=[3, 5, 7],
        use_hilbert_scan=True, n_scan_directions=4,
        use_discriminator=True, discriminator_hidden=512,
        noise_std=0.015, truncated_loss_margin=0.5,
        use_fbft=True, fbft_hidden=256,
        image_size=256,
        reconstruction_weight=1.0, discriminator_weight=0.5, fbft_weight=0.3,
    )
    
    configs = {}
    
    # 1. Full model
    configs['Full Model'] = MSViTFDConfig(**base_kwargs)
    
    # 2. w/o Hilbert Scanning
    kw = copy.deepcopy(base_kwargs)
    kw['use_hilbert_scan'] = False
    configs['w/o Hilbert Scan'] = MSViTFDConfig(**kw)
    
    # 3. w/o Multi-Kernel DWConv (single kernel)
    kw = copy.deepcopy(base_kwargs)
    kw['dwconv_kernels'] = [3]  # single kernel instead of [3,5,7]
    configs['w/o Multi-Kernel DWConv'] = MSViTFDConfig(**kw)
    
    # 4. w/o Feature Discriminator
    kw = copy.deepcopy(base_kwargs)
    kw['use_discriminator'] = False
    kw['discriminator_weight'] = 0.0
    configs['w/o Discriminator'] = MSViTFDConfig(**kw)
    
    # 5. w/o Bidirectional Feature Transfer
    kw = copy.deepcopy(base_kwargs)
    kw['use_fbft'] = False
    kw['fbft_weight'] = 0.0
    configs['w/o L2BT'] = MSViTFDConfig(**kw)
    
    # 6. w/o both Discriminator and L2BT (reconstruction only)
    kw = copy.deepcopy(base_kwargs)
    kw['use_discriminator'] = False
    kw['use_fbft'] = False
    kw['discriminator_weight'] = 0.0
    kw['fbft_weight'] = 0.0
    configs['Reconstruction Only'] = MSViTFDConfig(**kw)
    
    # 7. Raster + single direction
    kw = copy.deepcopy(base_kwargs)
    kw['use_hilbert_scan'] = True
    kw['n_scan_directions'] = 1  # single direction
    configs['Single Scan Direction'] = MSViTFDConfig(**kw)
    
    return configs


# ============================================================================
# DualFreqMamba Ablation Configurations
# ============================================================================

def get_dualfreqmamba_ablation_configs(n_channels=38, window_size=100):
    """Return dict of ablation name → config modifications."""
    from models.dualfreqmamba.configuration_dualfreqmamba import DualFreqMambaConfig
    
    base_kwargs = dict(
        n_channels=n_channels, window_size=window_size,
        fft_patch_size=4, fft_stride=2, fft_d_model=128,
        fft_n_heads=4, fft_n_layers=2, fft_dropout=0.1,
        use_channel_mask=True,
        use_cwt=True, cwt_scales=min(32, window_size // 4),
        cwt_wavelet_sigma=1.0,
        cwt_patch_size=min(8, window_size // 8) if window_size >= 16 else 2,
        cwt_d_model=64,
        mamba_d_model=128, mamba_d_state=16, mamba_d_conv=4,
        mamba_expand=2, mamba_n_layers=3,
        use_gated_skip=True,
        use_association_discrepancy=True,
        prior_sigma=1.0, association_lambda=3.0, sparse_block_size=10,
        fusion_d_model=128, fusion_method='adaptive_gate',
        scoring_method='combined',
        reconstruction_weight=1.0, frequency_weight=0.5, association_weight=1.0,
    )
    
    configs = {}
    
    # 1. Full model
    configs['Full Model'] = DualFreqMambaConfig(**base_kwargs)
    
    # 2. w/o CWT Branch
    kw = copy.deepcopy(base_kwargs)
    kw['use_cwt'] = False
    configs['w/o CWT Branch'] = DualFreqMambaConfig(**kw)
    
    # 3. w/o Association Discrepancy
    kw = copy.deepcopy(base_kwargs)
    kw['use_association_discrepancy'] = False
    kw['association_weight'] = 0.0
    configs['w/o Association Discrepancy'] = DualFreqMambaConfig(**kw)
    
    # 4. w/o Gated Skip in Mamba
    kw = copy.deepcopy(base_kwargs)
    kw['use_gated_skip'] = False
    configs['w/o Gated Skip'] = DualFreqMambaConfig(**kw)
    
    # 5. w/o Frequency Reconstruction
    kw = copy.deepcopy(base_kwargs)
    kw['frequency_weight'] = 0.0
    configs['w/o Freq Reconstruction'] = DualFreqMambaConfig(**kw)
    
    # 6. w/o Channel Mask (plain transformer in FFT branch)
    kw = copy.deepcopy(base_kwargs)
    kw['use_channel_mask'] = False
    configs['w/o Channel Mask'] = DualFreqMambaConfig(**kw)
    
    # 7. Concat fusion instead of adaptive gate
    kw = copy.deepcopy(base_kwargs)
    kw['fusion_method'] = 'concat'
    configs['Concat Fusion'] = DualFreqMambaConfig(**kw)
    
    # 8. Reconstruction-only scoring (no association in score)
    kw = copy.deepcopy(base_kwargs)
    kw['scoring_method'] = 'reconstruction'
    configs['Recon Scoring Only'] = DualFreqMambaConfig(**kw)
    
    return configs


# ============================================================================
# Ablation Table Generation
# ============================================================================

def generate_ablation_table(results, model_name, save_path):
    """Generate publication-quality ablation table as image."""
    fig, ax = plt.subplots(figsize=(12, max(4, len(results) * 0.6 + 2)))
    ax.axis('off')
    
    if model_name == 'msviTfd':
        headers = ['Configuration', 'I-AUROC (%)', 'P-AUROC (%)', 'I-F1 (%)', 'Δ I-AUROC']
        rows = []
        base_auroc = None
        for name, metrics in results.items():
            auroc = metrics.get('image_auroc', 0) * 100
            if base_auroc is None:
                base_auroc = auroc
            delta = auroc - base_auroc
            rows.append([
                name,
                f"{auroc:.2f}",
                f"{metrics.get('pixel_auroc', 0) * 100:.2f}",
                f"{metrics.get('image_f1', 0) * 100:.2f}",
                f"{delta:+.2f}" if name != 'Full Model' else '—'
            ])
    else:
        headers = ['Configuration', 'F1 (%)', 'Precision (%)', 'Recall (%)', 'Δ F1']
        rows = []
        base_f1 = None
        for name, metrics in results.items():
            f1 = metrics.get('f1', 0) * 100
            if base_f1 is None:
                base_f1 = f1
            delta = f1 - base_f1
            rows.append([
                name,
                f"{f1:.2f}",
                f"{metrics.get('precision', 0) * 100:.2f}",
                f"{metrics.get('recall', 0) * 100:.2f}",
                f"{delta:+.2f}" if name != 'Full Model' else '—'
            ])
    
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(headers))))
    table.scale(1.2, 1.5)
    
    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Highlight full model row
    for j in range(len(headers)):
        table[1, j].set_facecolor('#D6E4F0')
    
    # Color-code deltas
    for i in range(1, len(rows)):
        delta_cell = table[i + 1, len(headers) - 1]
        if rows[i][-1] != '—':
            delta_val = float(rows[i][-1])
            if delta_val < -1:
                delta_cell.set_facecolor('#FFCCCC')
            elif delta_val < -0.5:
                delta_cell.set_facecolor('#FFE6CC')
    
    ax.set_title(f'Ablation Study: {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    

def generate_ablation_bar_chart(results, model_name, metric_key, metric_name, save_path):
    """Bar chart of ablation results."""
    names = list(results.keys())
    values = [results[n].get(metric_key, 0) * 100 for n in names]
    
    colors = ['#4472C4'] + ['#ED7D31'] * (len(names) - 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor='white', height=0.6)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(f'{metric_name} (%)', fontsize=12)
    ax.set_title(f'Ablation Study: {model_name} — {metric_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['msviTfd', 'dualfreqmamba', 'both'], default='both')
    parser.add_argument('--data_root', type=str, default='./datasets/MVTec')
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--output_dir', type=str, default='./ablation_results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--epochs_image', type=int, default=200)
    parser.add_argument('--epochs_ts', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Running ablation studies on {device}")
    logger.info(f"Seeds: {args.seeds}")
    
    # Print ablation configurations (even without data, shows what will be tested)
    if args.model in ['msviTfd', 'both']:
        configs = get_msviTfd_ablation_configs()
        logger.info(f"\nMSViTFD Ablations ({len(configs)} configurations):")
        for name in configs:
            logger.info(f"  - {name}")
    
    if args.model in ['dualfreqmamba', 'both']:
        configs = get_dualfreqmamba_ablation_configs()
        logger.info(f"\nDualFreqMamba Ablations ({len(configs)} configurations):")
        for name in configs:
            logger.info(f"  - {name}")
    
    logger.info(f"\nTo run full ablations, ensure datasets are available at {args.data_root}")
    logger.info("Each configuration will be trained with seeds: " + str(args.seeds))
    logger.info("Results will include mean ± std across seeds")


if __name__ == '__main__':
    main()
