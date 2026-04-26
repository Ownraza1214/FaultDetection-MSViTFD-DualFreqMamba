#!/usr/bin/env python3
"""
Multi-Seed Evaluation for Statistical Significance
====================================================

Runs training with multiple seeds and reports mean ± std.
This is required by reviewers to demonstrate reproducibility.

Usage:
    python multi_seed_eval.py --model msviTfd --data_root ./datasets/MVTec --category bottle
    python multi_seed_eval.py --model dualfreqmamba --dataset SMD
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SEEDS = [42, 123, 456]


def run_multi_seed_msviTfd(args):
    """Run MSViTFD training with multiple seeds."""
    results_per_seed = {}
    
    for seed in args.seeds:
        output_dir = os.path.join(args.output_dir, f'seed_{seed}')
        
        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), 'train_msviTfd.py'),
            '--data_root', args.data_root,
            '--category', args.category,
            '--output_dir', output_dir,
            '--seed', str(seed),
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
        ]
        
        logging.info(f"Running seed {seed}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            summary_path = os.path.join(output_dir, 'summary.json')
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    results_per_seed[seed] = json.load(f)
        else:
            logging.error(f"Seed {seed} failed: {result.stderr[:500]}")
    
    return results_per_seed


def run_multi_seed_dualfreqmamba(args):
    """Run DualFreqMamba training with multiple seeds."""
    results_per_seed = {}
    
    for seed in args.seeds:
        output_dir = os.path.join(args.output_dir, f'seed_{seed}')
        
        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), 'train_dualfreqmamba.py'),
            '--dataset', args.dataset,
            '--output_dir', output_dir,
            '--seed', str(seed),
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
        ]
        
        logging.info(f"Running seed {seed}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            summary_path = os.path.join(output_dir, 'summary.json')
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    results_per_seed[seed] = json.load(f)
        else:
            logging.error(f"Seed {seed} failed: {result.stderr[:500]}")
    
    return results_per_seed


def generate_multi_seed_table(results, model_name, save_path):
    """Generate mean ± std table from multi-seed results."""
    if not results:
        return
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    
    if model_name == 'msviTfd':
        metrics_keys = ['mean_image_auroc', 'mean_image_ap', 'mean_image_f1', 'mean_pixel_auroc']
        metric_names = ['Image AUROC', 'Image AP', 'Image F1', 'Pixel AUROC']
        
        headers = ['Seed'] + metric_names
        rows = []
        
        for seed, result in sorted(results.items()):
            row = [str(seed)]
            for key in metrics_keys:
                val = result.get(key, 0) * 100
                row.append(f"{val:.2f}")
            rows.append(row)
        
        # Mean ± std row
        mean_row = ['Mean ± Std']
        for key in metrics_keys:
            vals = [r.get(key, 0) * 100 for r in results.values()]
            mean_row.append(f"{np.mean(vals):.2f} ± {np.std(vals):.2f}")
        rows.append(mean_row)
    
    else:
        metrics_keys = ['mean_f1']
        metric_names = ['F1 Score']
        
        # Get per-dataset results
        first_result = next(iter(results.values()))
        datasets = list(first_result.get('per_dataset', {}).keys())
        
        headers = ['Seed'] + [f'{d} F1%' for d in datasets] + ['Mean F1%']
        rows = []
        
        for seed, result in sorted(results.items()):
            row = [str(seed)]
            per_ds = result.get('per_dataset', {})
            for d in datasets:
                f1 = per_ds.get(d, {}).get('f1', 0) * 100
                row.append(f"{f1:.2f}")
            row.append(f"{result.get('mean_f1', 0) * 100:.2f}")
            rows.append(row)
        
        # Mean ± std
        mean_row = ['Mean ± Std']
        for d in datasets:
            vals = [r.get('per_dataset', {}).get(d, {}).get('f1', 0) * 100 for r in results.values()]
            mean_row.append(f"{np.mean(vals):.2f} ± {np.std(vals):.2f}")
        mean_f1s = [r.get('mean_f1', 0) * 100 for r in results.values()]
        mean_row.append(f"{np.mean(mean_f1s):.2f} ± {np.std(mean_f1s):.2f}")
        rows.append(mean_row)
    
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(headers))))
    table.scale(1.2, 1.5)
    
    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Highlight mean row
    last_row = len(rows)
    for j in range(len(headers)):
        table[last_row, j].set_facecolor('#E2EFDA')
        table[last_row, j].set_text_props(fontweight='bold')
    
    ax.set_title(f'{model_name} — Multi-Seed Results (Mean ± Std)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['msviTfd', 'dualfreqmamba'], required=True)
    parser.add_argument('--data_root', type=str, default='./datasets/MVTec')
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--output_dir', type=str, default='./multi_seed_results')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info(f"Multi-seed evaluation: {args.model}")
    logging.info(f"Seeds: {args.seeds}")
    
    if args.model == 'msviTfd':
        results = run_multi_seed_msviTfd(args)
    else:
        results = run_multi_seed_dualfreqmamba(args)
    
    if results:
        generate_multi_seed_table(
            results, args.model,
            os.path.join(args.output_dir, f'{args.model}_multi_seed_table.png')
        )
        
        with open(os.path.join(args.output_dir, f'{args.model}_multi_seed_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
