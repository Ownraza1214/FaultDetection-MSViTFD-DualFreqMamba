#!/usr/bin/env python3
"""
Complexity Analysis — FLOPs, Inference Time, Parameter Counts
==============================================================

Computes and compares:
1. Total parameters (trainable + frozen)
2. FLOPs (multiply-accumulate operations)
3. Inference time (ms/sample on CPU and GPU)
4. Memory footprint (MB)
5. Throughput (samples/second)

For both MSViTFD and DualFreqMamba, plus comparison with published baselines.

Usage:
    python complexity_analysis.py --output_dir ./complexity_results
"""

import os
import sys
import json
import time
import argparse
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.msviTfd.configuration_msviTfd import MSViTFDConfig
from models.msviTfd.modeling_msviTfd import MSViTFDModel
from models.dualfreqmamba.configuration_dualfreqmamba import DualFreqMambaConfig
from models.dualfreqmamba.modeling_dualfreqmamba import DualFreqMambaModel


def count_parameters(model):
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {'total': total, 'trainable': trainable, 'frozen': frozen}


def estimate_flops_msviTfd(config, batch_size=1):
    """Estimate FLOPs for MSViTFD.
    
    Approximate calculation based on architecture components:
    - Encoder (EfficientNet-B2): ~1B FLOPs
    - Decoder (Mamba + DWConv): calculated from dim sizes
    - Discriminator: small MLP
    - FBFT: small MLP
    """
    H = config.image_size
    D = config.hidden_dim
    
    # Encoder FLOPs (EfficientNet-B2 ~1.0 GFLOPs at 256x256)
    encoder_flops = 1.0e9
    
    # Decoder FLOPs per scale
    decoder_flops = 0
    spatial_sizes = [H // 4, H // 8, H // 16]
    for i, (depth, ss) in enumerate(zip(config.decoder_depths, spatial_sizes)):
        L = ss * ss  # sequence length
        # Input projection (Conv2d 1x1): C_in * D * L
        decoder_flops += config.feature_dims[i] * D * L
        
        for _ in range(depth):
            # SSM: in_proj + conv1d + x_proj + scan + out_proj
            d_inner = D * config.mamba_expand
            decoder_flops += L * D * d_inner * 2  # in_proj
            decoder_flops += L * d_inner * config.mamba_d_conv  # conv1d
            decoder_flops += L * d_inner * config.mamba_d_state * 3  # scan
            decoder_flops += L * d_inner * D  # out_proj
            
            # Multi-kernel DWConv
            for k in config.dwconv_kernels:
                decoder_flops += L * D * k  # depthwise
            decoder_flops += L * D * len(config.dwconv_kernels) * D  # fusion
            
            # Gated fusion
            decoder_flops += L * D * 2 * D  # gate
        
        # Reconstruction head
        decoder_flops += L * D * D + L * D * config.feature_dims[i]
    
    # Discriminator
    if config.use_discriminator:
        total_L = sum(s * s for s in spatial_sizes)
        disc_flops = total_L * (D * config.discriminator_hidden + 
                                config.discriminator_hidden * config.discriminator_hidden // 2 +
                                config.discriminator_hidden // 2)
        decoder_flops += disc_flops
    
    # FBFT
    if config.use_fbft:
        fbft_flops = 2 * D * config.fbft_hidden * 2  # forward + backward MLPs
        decoder_flops += fbft_flops * (len(config.feature_dims) - 1)
    
    total_flops = (encoder_flops + decoder_flops) * batch_size
    return {
        'encoder_gflops': encoder_flops / 1e9,
        'decoder_gflops': decoder_flops / 1e9,
        'total_gflops': total_flops / 1e9,
    }


def estimate_flops_dualfreqmamba(config, batch_size=1):
    """Estimate FLOPs for DualFreqMamba."""
    T = config.window_size
    C = config.n_channels
    D = config.mamba_d_model
    
    # FFT Branch
    fft_flops = T * C * np.log2(T)  # FFT itself
    freq_len = T // 2 + 1
    n_patches = max(1, (freq_len - config.fft_patch_size) // config.fft_stride + 1)
    patch_dim = C * 2 * config.fft_patch_size
    fft_flops += n_patches * patch_dim * config.fft_d_model * 2  # projection
    
    # Transformer layers in FFT branch
    for _ in range(config.fft_n_layers):
        fft_flops += n_patches * config.fft_d_model * config.fft_d_model * 4  # QKV + out
        fft_flops += n_patches * n_patches * config.fft_d_model  # attention
        fft_flops += n_patches * config.fft_d_model * config.fft_d_model * 8  # FFN
    
    # CWT Branch
    cwt_flops = 0
    if config.use_cwt:
        cwt_flops = config.cwt_scales * C * T * 256  # wavelet convolution
        cwt_flops += C * config.cwt_d_model * (config.cwt_patch_size ** 2) * (
            (config.cwt_scales // config.cwt_patch_size) * (T // config.cwt_patch_size)
        )
    
    # Temporal Branch
    temp_flops = T * C * config.fusion_d_model * 2
    
    # Fusion
    fusion_flops = T * config.fusion_d_model * 3 * config.fusion_d_model  # gate
    fusion_flops += T * config.fusion_d_model * D  # output
    
    # Mamba Decoder
    mamba_flops = 0
    L = n_patches  # approximate sequence length after fusion
    for _ in range(config.mamba_n_layers):
        d_inner = D * config.mamba_expand
        mamba_flops += L * D * d_inner * 2
        mamba_flops += L * d_inner * config.mamba_d_conv
        mamba_flops += L * d_inner * config.mamba_d_state * 3
        mamba_flops += L * d_inner * D
        # Gated skip + FFN
        mamba_flops += L * D * D * 2  # skip gate
        mamba_flops += L * D * D * 4  # FFN
    
    # Association
    assoc_flops = 0
    if config.use_association_discrepancy:
        assoc_flops = L * D * D * 2  # Q, K projections
        assoc_flops += L * L * D  # attention
    
    # Reconstruction
    recon_flops = L * D * D + L * D * C  # time recon
    recon_flops += L * D * D + L * D * C * 2  # freq recon
    
    total = (fft_flops + cwt_flops + temp_flops + fusion_flops + mamba_flops + assoc_flops + recon_flops) * batch_size
    
    return {
        'fft_branch_mflops': fft_flops / 1e6,
        'cwt_branch_mflops': cwt_flops / 1e6,
        'temporal_branch_mflops': temp_flops / 1e6,
        'mamba_decoder_mflops': mamba_flops / 1e6,
        'total_mflops': total / 1e6,
        'total_gflops': total / 1e9,
    }


def measure_inference_time(model, input_tensor, n_warmup=10, n_runs=50, device='cpu'):
    """Measure inference time."""
    model.eval()
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor) if hasattr(model, 'pixel_values') else model(input_tensor)
    
    # Timed runs
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            if str(device) == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps': 1000.0 / np.mean(times),
    }


def get_model_memory(model):
    """Get model memory footprint in MB."""
    mem = 0
    for p in model.parameters():
        mem += p.numel() * p.element_size()
    for b in model.buffers():
        mem += b.numel() * b.element_size()
    return mem / (1024 * 1024)  # MB


# ============================================================================
# Published Baseline Complexity
# ============================================================================

PUBLISHED_COMPLEXITY = OrderedDict({
    # Image baselines
    'PatchCore': {'params_M': 43.0, 'gflops': 3.8, 'fps': 5.6, 'type': 'image',
                  'note': 'WideResNet50 + memory bank, ~3.8 GB memory bank'},
    'PaDiM': {'params_M': 23.5, 'gflops': 3.8, 'fps': 1.1, 'type': 'image',
              'note': 'WideResNet50, Mahalanobis distance per patch'},
    'Reverse Distillation': {'params_M': 51.0, 'gflops': 5.2, 'fps': 3.2, 'type': 'image',
                              'note': 'Teacher WR50 + Student decoder, 352 MB total'},
    'SimpleNet': {'params_M': 25.0, 'gflops': 4.0, 'fps': 77.0, 'type': 'image',
                  'note': 'WideResNet50 + small discriminator, fastest'},
    
    # TS baselines  
    'Anomaly Transformer': {'params_M': 7.2, 'gflops': 0.8, 'fps': 156, 'type': 'ts',
                            'note': '3-layer Transformer, d=512, 8 heads'},
    'TimesNet': {'params_M': 1.5, 'gflops': 0.3, 'fps': 89, 'type': 'ts',
                 'note': 'TimesBlock + Inception, d_model auto-sized'},
    'USAD': {'params_M': 0.5, 'gflops': 0.05, 'fps': 5000, 'type': 'ts',
             'note': 'Two FC autoencoders, very lightweight'},
    'OmniAnomaly': {'params_M': 3.0, 'gflops': 0.4, 'fps': 200, 'type': 'ts',
                    'note': 'GRU + Normalizing Flow, 500-d hidden'},
})


def generate_complexity_table(results, save_path):
    """Generate comprehensive complexity comparison table."""
    fig, ax = plt.subplots(figsize=(16, max(6, (len(results) + 2) * 0.6)))
    ax.axis('off')
    
    headers = ['Method', 'Type', 'Params (M)', 'GFLOPs', 'FPS', 'Memory (MB)', 'Notes']
    rows = []
    
    for name, info in results.items():
        rows.append([
            name,
            info.get('type', '—'),
            f"{info.get('params_M', 0):.1f}",
            f"{info.get('gflops', 0):.2f}",
            f"{info.get('fps', 0):.1f}",
            f"{info.get('memory_mb', 0):.1f}" if info.get('memory_mb', 0) > 0 else '—',
            info.get('note', '')[:40],
        ])
    
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(headers))))
    table.scale(1.1, 1.5)
    
    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Highlight our models
    for i, name in enumerate(results.keys()):
        if '(Ours)' in name:
            for j in range(len(headers)):
                table[i + 1, j].set_facecolor('#E2EFDA')
                table[i + 1, j].set_text_props(fontweight='bold')
    
    ax.set_title('Complexity Analysis: Parameters, FLOPs, and Inference Speed', 
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_params_vs_performance(save_path):
    """Scatter plot: Parameters vs Performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Image methods
    image_methods = {
        'PatchCore': (43.0, 99.1),
        'SimpleNet': (25.0, 99.6),
        'RevDist': (51.0, 98.5),
        'PaDiM': (23.5, 95.3),
        'MSViTFD\n(Ours)': (14.3, 0),  # Will be filled with actual results
    }
    
    for name, (params, auroc) in image_methods.items():
        color = '#C00000' if 'Ours' in name else '#4472C4'
        marker = '*' if 'Ours' in name else 'o'
        size = 200 if 'Ours' in name else 100
        ax1.scatter(params, auroc, c=color, marker=marker, s=size, zorder=5)
        ax1.annotate(name, (params, auroc), textcoords="offset points", 
                    xytext=(5, 5), fontsize=9)
    
    ax1.set_xlabel('Parameters (M)', fontsize=12)
    ax1.set_ylabel('Image AUROC (%)', fontsize=12)
    ax1.set_title('Image AD: Params vs Performance', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # TS methods
    ts_methods = {
        'AnomalyTrans': (7.2, 94.91),
        'TimesNet': (1.5, 86.34),
        'OmniAnomaly': (3.0, 84.69),
        'USAD': (0.5, 85.14),
        'DualFreqMamba\n(Ours)': (1.5, 0),  # Will be filled
    }
    
    for name, (params, f1) in ts_methods.items():
        color = '#C00000' if 'Ours' in name else '#4472C4'
        marker = '*' if 'Ours' in name else 'o'
        size = 200 if 'Ours' in name else 100
        ax2.scatter(params, f1, c=color, marker=marker, s=size, zorder=5)
        ax2.annotate(name, (params, f1), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    
    ax2.set_xlabel('Parameters (M)', fontsize=12)
    ax2.set_ylabel('Avg F1 Score (%)', fontsize=12)
    ax2.set_title('TS AD: Params vs Performance', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./complexity_results')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_runs', type=int, default=20)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    results = OrderedDict()
    
    # ---- MSViTFD ----
    logger.info("Analyzing MSViTFD...")
    msviTfd_config = MSViTFDConfig(
        encoder_name='efficientnet_b2', hidden_dim=256,
        decoder_depths=[2, 3, 2], mamba_d_state=16,
        image_size=256,
    )
    
    msviTfd_model = MSViTFDModel(msviTfd_config)
    msviTfd_params = count_parameters(msviTfd_model)
    msviTfd_flops = estimate_flops_msviTfd(msviTfd_config)
    msviTfd_mem = get_model_memory(msviTfd_model)
    
    # Inference time
    dummy_img = torch.randn(1, 3, 256, 256)
    msviTfd_time = measure_inference_time(msviTfd_model, dummy_img, n_runs=args.n_runs, device=args.device)
    
    results['MSViTFD (Ours)'] = {
        'type': 'image',
        'params_M': msviTfd_params['total'] / 1e6,
        'trainable_M': msviTfd_params['trainable'] / 1e6,
        'frozen_M': msviTfd_params['frozen'] / 1e6,
        'gflops': msviTfd_flops['total_gflops'],
        'fps': msviTfd_time['fps'],
        'memory_mb': msviTfd_mem,
        'inference_ms': msviTfd_time['mean_ms'],
        'note': f"EfficientNet-B2 + Mamba LSS Decoder",
    }
    
    logger.info(f"  Params: {msviTfd_params['total']/1e6:.1f}M "
                f"(trainable: {msviTfd_params['trainable']/1e6:.1f}M)")
    logger.info(f"  FLOPs: {msviTfd_flops['total_gflops']:.2f} GFLOPs")
    logger.info(f"  Inference: {msviTfd_time['mean_ms']:.1f}ms ({msviTfd_time['fps']:.1f} FPS)")
    logger.info(f"  Memory: {msviTfd_mem:.1f} MB")
    
    # ---- DualFreqMamba ----
    logger.info("\nAnalyzing DualFreqMamba...")
    dfm_config = DualFreqMambaConfig(
        n_channels=38, window_size=100,
        fft_d_model=128, mamba_d_model=128, mamba_n_layers=3,
    )
    
    dfm_model = DualFreqMambaModel(dfm_config)
    dfm_params = count_parameters(dfm_model)
    dfm_flops = estimate_flops_dualfreqmamba(dfm_config)
    dfm_mem = get_model_memory(dfm_model)
    
    dummy_ts = torch.randn(1, 38, 100)
    dfm_time = measure_inference_time(dfm_model, dummy_ts, n_runs=args.n_runs, device=args.device)
    
    results['DualFreqMamba (Ours)'] = {
        'type': 'ts',
        'params_M': dfm_params['total'] / 1e6,
        'trainable_M': dfm_params['trainable'] / 1e6,
        'gflops': dfm_flops['total_gflops'],
        'fps': dfm_time['fps'],
        'memory_mb': dfm_mem,
        'inference_ms': dfm_time['mean_ms'],
        'note': f"FFT+CWT+Temporal → Mamba Decoder",
    }
    
    logger.info(f"  Params: {dfm_params['total']/1e6:.2f}M")
    logger.info(f"  FLOPs: {dfm_flops['total_gflops']:.4f} GFLOPs ({dfm_flops['total_mflops']:.1f} MFLOPs)")
    logger.info(f"  Inference: {dfm_time['mean_ms']:.1f}ms ({dfm_time['fps']:.1f} FPS)")
    logger.info(f"  Memory: {dfm_mem:.1f} MB")
    
    # Add published baselines
    for name, info in PUBLISHED_COMPLEXITY.items():
        results[name] = info
    
    # Generate outputs
    generate_complexity_table(results, os.path.join(args.output_dir, 'complexity_table.png'))
    generate_params_vs_performance(os.path.join(args.output_dir, 'params_vs_performance.png'))
    
    # Save JSON
    with open(os.path.join(args.output_dir, 'complexity_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
