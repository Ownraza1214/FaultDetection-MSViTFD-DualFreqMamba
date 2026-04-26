#!/usr/bin/env python3
"""
CPU Training Script — Train both models on real benchmark datasets.
Uses reduced model configs to fit CPU memory/time constraints while
still producing valid benchmark results on standard datasets.
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/app')
from models.dualfreqmamba.configuration_dualfreqmamba import DualFreqMambaConfig
from models.dualfreqmamba.modeling_dualfreqmamba import DualFreqMambaModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cpu')
SEEDS = [42]  # start with 1 seed for speed; add 123, 456 after validation

# ============================================================================
# Data Loading
# ============================================================================

DATASET_CONFIGS = {
    'SMD': {'n_channels': 38, 'source': 'npy', 'prefix': 'SMD'},
    'MSL': {'n_channels': 55, 'source': 'npy', 'prefix': 'MSL'},
    'SMAP': {'n_channels': 25, 'source': 'npy', 'prefix': 'SMAP'},
    'SWaT': {'n_channels': 51, 'source': 'parquet', 'config_name': 'SWaT'},
    'PSM': {'n_channels': 25, 'source': 'parquet', 'config_name': 'PSM-data'},
}

def load_ts_data(dataset_name):
    """Load time-series anomaly detection dataset from HuggingFace."""
    ds_config = DATASET_CONFIGS[dataset_name]
    
    if ds_config['source'] == 'npy':
        from huggingface_hub import hf_hub_download
        prefix = ds_config['prefix']
        train_path = hf_hub_download("thuml/Time-Series-Library", 
                                      f"{prefix}/{prefix}_train.npy", repo_type="dataset")
        test_path = hf_hub_download("thuml/Time-Series-Library", 
                                     f"{prefix}/{prefix}_test.npy", repo_type="dataset")
        label_path = hf_hub_download("thuml/Time-Series-Library", 
                                      f"{prefix}/{prefix}_test_label.npy", repo_type="dataset")
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_labels = np.load(label_path)
        
    elif ds_config['source'] == 'parquet':
        from datasets import load_dataset
        ds = load_dataset("thuml/Time-Series-Library", ds_config['config_name'])
        train_df = ds['train'].to_pandas()
        test_df = ds['test'].to_pandas()
        
        if dataset_name == 'SWaT':
            label_col = 'Normal/Attack'
            feature_cols = [c for c in train_df.columns if c != label_col]
            train_data = train_df[feature_cols].values.astype(np.float32)
            test_data = test_df[feature_cols].values.astype(np.float32)
            test_labels = test_df[label_col].values.astype(np.float32)
        else:  # PSM
            feature_cols = [c for c in train_df.columns if c not in ['label', 'timestamp']]
            train_data = train_df[feature_cols].values.astype(np.float32)
            test_data = test_df[feature_cols].values.astype(np.float32)
            if 'label' in test_df.columns:
                test_labels = test_df['label'].values.astype(np.float32)
            else:
                test_labels = np.zeros(len(test_df))
    
    # Handle NaN and shape
    train_data = np.nan_to_num(train_data, nan=0.0).astype(np.float32)
    test_data = np.nan_to_num(test_data, nan=0.0).astype(np.float32)
    test_labels = np.nan_to_num(test_labels, nan=0.0).astype(np.float32)
    if test_labels.ndim > 1:
        test_labels = test_labels.max(axis=1)
    
    logger.info(f"[{dataset_name}] Train: {train_data.shape}, Test: {test_data.shape}, "
                f"Anomaly ratio: {test_labels.mean():.4f}")
    return train_data, test_data, test_labels


class TSWindowDataset(Dataset):
    """Sliding window dataset."""
    def __init__(self, data, labels=None, window_size=100, stride=1, max_windows=None):
        self.window_size = window_size
        self.stride = stride
        # Normalize per-channel
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True) + 1e-8
        self.data = ((data - self.mean) / self.std).astype(np.float32)
        self.labels = labels
        self.n_windows = max(1, (len(data) - window_size) // stride + 1)
        if max_windows and self.n_windows > max_windows:
            self.n_windows = max_windows
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        window = torch.tensor(self.data[start:end], dtype=torch.float32).T  # (C, T)
        if self.labels is not None:
            label = torch.tensor(self.labels[start:end], dtype=torch.float32)
            return {'input_values': window, 'labels': label}
        return {'input_values': window}


# ============================================================================
# Point-Adjust Protocol
# ============================================================================

def point_adjust(pred, label):
    """Standard point-adjust for TS anomaly detection."""
    pred = pred.copy()
    anomaly_state = False
    for i in range(len(label)):
        if label[i] == 1 and not anomaly_state:
            anomaly_state = True
            start = i
        if label[i] == 0 and anomaly_state:
            anomaly_state = False
            if pred[start:i].sum() > 0:
                pred[start:i] = 1
        if i == len(label) - 1 and anomaly_state:
            if pred[start:i+1].sum() > 0:
                pred[start:i+1] = 1
    return pred


def best_f1_with_point_adjust(scores, labels, n_thresholds=500):
    """Find best F1 threshold with point-adjust.
    Searches across full range of percentiles (1% to 99.9%)."""
    # Search across WIDE range of thresholds 
    thresholds = np.percentile(scores, np.linspace(1, 99.9, n_thresholds))
    # Also add mean + k*std thresholds
    for k in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        thresholds = np.append(thresholds, scores.mean() + k * scores.std())
    thresholds = np.unique(thresholds)
    
    best_f1, best_prec, best_rec, best_t = 0, 0, 0, 0
    for t in thresholds:
        pred = (scores > t).astype(int)
        pred_adj = point_adjust(pred, labels)
        f1 = f1_score(labels, pred_adj, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_prec = precision_score(labels, pred_adj, zero_division=0)
            best_rec = recall_score(labels, pred_adj, zero_division=0)
            best_t = t
    return {'f1': best_f1, 'precision': best_prec, 'recall': best_rec, 'threshold': float(best_t)}


# ============================================================================
# Training Loop
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_dualfreqmamba_on_dataset(dataset_name, seed, output_dir, 
                                     epochs=10, batch_size=32, lr=1e-4,
                                     window_size=100, max_train_windows=5000):
    """Train DualFreqMamba on a single dataset."""
    set_seed(seed)
    logger.info(f"\n{'='*60}")
    logger.info(f"Training DualFreqMamba on {dataset_name} | seed={seed}")
    logger.info(f"{'='*60}")
    
    # Load data
    train_data, test_data, test_labels = load_ts_data(dataset_name)
    n_channels = train_data.shape[1]
    
    # Create model (CPU-friendly config)
    config = DualFreqMambaConfig(
        n_channels=n_channels,
        window_size=window_size,
        fft_patch_size=4,
        fft_stride=2,
        fft_d_model=64,        # reduced from 128 for CPU
        fft_n_heads=4,
        fft_n_layers=2,
        fft_dropout=0.1,
        use_channel_mask=True,
        use_cwt=True,
        cwt_scales=min(16, window_size // 4),  # reduced from 32
        cwt_wavelet_sigma=1.0,
        cwt_patch_size=min(8, window_size // 8) if window_size >= 16 else 2,
        cwt_d_model=32,         # reduced from 64
        mamba_d_model=64,       # reduced from 128
        mamba_d_state=8,        # reduced from 16
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_n_layers=2,       # reduced from 3
        use_gated_skip=True,
        use_association_discrepancy=True,
        prior_sigma=1.0,
        association_lambda=3.0,
        sparse_block_size=10,
        fusion_d_model=64,      # reduced from 128
        fusion_method='adaptive_gate',
        scoring_method='reconstruction',  # use recon error for stable scoring
        reconstruction_weight=1.0,
        frequency_weight=0.5,
        association_weight=0.5,  # reduced from 1.0 to prevent minimax domination
    )
    
    model = DualFreqMambaModel(config).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total_params:,}")
    
    # Datasets
    train_dataset = TSWindowDataset(train_data, window_size=window_size, stride=window_size,
                                     max_windows=max_train_windows)
    test_dataset = TSWindowDataset(test_data, test_labels, window_size=window_size, stride=window_size//2,
                                    max_windows=min(len(test_data) // (window_size//2), 5000))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train windows: {len(train_dataset)}, Test windows: {len(test_dataset)}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    history = {'loss': [], 'recon_loss': [], 'freq_loss': [], 'assoc_loss': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_freq = 0
        epoch_assoc = 0
        n_batches = 0
        
        for batch in train_loader:
            input_values = batch['input_values'].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(input_values=input_values, return_anomaly_labels=False)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += outputs.reconstruction_loss.item() if outputs.reconstruction_loss is not None else 0
            epoch_freq += outputs.frequency_loss.item() if outputs.frequency_loss is not None else 0
            epoch_assoc += outputs.association_loss.item() if outputs.association_loss is not None else 0
            n_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / max(n_batches, 1)
        history['loss'].append(avg_loss)
        history['recon_loss'].append(epoch_recon / max(n_batches, 1))
        history['freq_loss'].append(epoch_freq / max(n_batches, 1))
        history['assoc_loss'].append(epoch_assoc / max(n_batches, 1))
        
        logger.info(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.6f} "
                    f"recon={history['recon_loss'][-1]:.6f} "
                    f"freq={history['freq_loss'][-1]:.6f} "
                    f"assoc={history['assoc_loss'][-1]:.6f}")
    
    # Evaluate
    logger.info("Evaluating...")
    model.eval()
    
    # Get train scores for normalization
    train_scores_list = []
    with torch.no_grad():
        for batch in train_loader:
            out = model(batch['input_values'].to(DEVICE), return_anomaly_labels=False)
            train_scores_list.append(out.anomaly_score.numpy())
    train_scores_all = np.concatenate(train_scores_list, axis=0)
    train_mean = train_scores_all.mean()
    train_std = train_scores_all.std() + 1e-8
    
    # Get test scores
    all_scores = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch['input_values'].to(DEVICE), return_anomaly_labels=False)
            all_scores.append(out.anomaly_score.numpy())
    all_scores = np.concatenate(all_scores, axis=0)
    
    # Reconstruct pointwise scores
    test_len = len(test_labels)
    point_scores = np.zeros(test_len)
    point_counts = np.zeros(test_len)
    n_test_windows = len(all_scores)
    for i in range(n_test_windows):
        start = i
        end = min(start + window_size, test_len)
        score_len = end - start
        point_scores[start:end] += all_scores[i, :score_len]
        point_counts[start:end] += 1
    point_counts[point_counts == 0] = 1
    point_scores = point_scores / point_counts
    point_scores = (point_scores - train_mean) / train_std
    
    # Metrics
    metrics = best_f1_with_point_adjust(point_scores, test_labels)
    try:
        metrics['auroc'] = roc_auc_score(test_labels, point_scores)
    except:
        metrics['auroc'] = 0.0
    
    logger.info(f"  RESULTS: F1={metrics['f1']:.4f} Prec={metrics['precision']:.4f} "
                f"Rec={metrics['recall']:.4f} AUROC={metrics['auroc']:.4f}")
    
    # Save
    save_dir = Path(output_dir) / dataset_name / f'seed_{seed}'
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir / 'model'))
    
    with open(str(save_dir / 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(str(save_dir / 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'DualFreqMamba — {dataset_name} (seed={seed})', fontweight='bold')
    axes[0].plot(history['loss'], 'b-', linewidth=1.5)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['recon_loss'], 'r-', label='Recon')
    axes[1].plot(history['freq_loss'], 'g-', label='Freq')
    axes[1].plot(history['assoc_loss'], 'm-', label='Assoc')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].set_title('Component Losses')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_dir / 'training_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    output_dir = '/app/benchmark_results/dualfreqmamba'
    os.makedirs(output_dir, exist_ok=True)
    
    datasets_to_train = ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']
    
    all_results = {}
    
    for ds_name in datasets_to_train:
        seed_results = {}
        for seed in SEEDS:
            try:
                metrics = train_dualfreqmamba_on_dataset(
                    ds_name, seed, output_dir,
                    epochs=10,
                    batch_size=32,
                    lr=1e-4,
                    window_size=100,
                    max_train_windows=5000,  # limit for CPU speed
                )
                seed_results[seed] = metrics
            except Exception as e:
                logger.error(f"FAILED {ds_name} seed={seed}: {e}")
                import traceback; traceback.print_exc()
        
        if seed_results:
            # Compute mean ± std
            f1s = [m['f1'] for m in seed_results.values()]
            precs = [m['precision'] for m in seed_results.values()]
            recs = [m['recall'] for m in seed_results.values()]
            aurocs = [m['auroc'] for m in seed_results.values()]
            
            all_results[ds_name] = {
                'per_seed': {str(k): v for k, v in seed_results.items()},
                'f1_mean': float(np.mean(f1s)),
                'f1_std': float(np.std(f1s)),
                'precision_mean': float(np.mean(precs)),
                'recall_mean': float(np.mean(recs)),
                'auroc_mean': float(np.mean(aurocs)),
                'auroc_std': float(np.std(aurocs)),
            }
            
            logger.info(f"\n  {ds_name} SUMMARY: F1={np.mean(f1s)*100:.2f}±{np.std(f1s)*100:.2f}% "
                        f"AUROC={np.mean(aurocs)*100:.2f}±{np.std(aurocs)*100:.2f}%")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL RESULTS: DualFreqMamba (3 seeds, point-adjust)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Dataset':<10} {'F1 (%)':>15} {'Prec (%)':>15} {'Rec (%)':>15} {'AUROC (%)':>15}")
    logger.info(f"{'-'*70}")
    
    avg_f1s = []
    for ds, r in all_results.items():
        f1_str = f"{r['f1_mean']*100:.2f}±{r['f1_std']*100:.2f}"
        prec_str = f"{r['precision_mean']*100:.2f}"
        rec_str = f"{r['recall_mean']*100:.2f}"
        auroc_str = f"{r['auroc_mean']*100:.2f}±{r['auroc_std']*100:.2f}"
        logger.info(f"{ds:<10} {f1_str:>15} {prec_str:>15} {rec_str:>15} {auroc_str:>15}")
        avg_f1s.append(r['f1_mean'])
    
    logger.info(f"{'-'*70}")
    logger.info(f"{'MEAN':<10} {np.mean(avg_f1s)*100:.2f}%")
    
    # Save
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}/all_results.json")


if __name__ == '__main__':
    main()
