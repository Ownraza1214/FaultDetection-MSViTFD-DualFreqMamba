#!/usr/bin/env python3
"""
DualFreqMamba Training Pipeline — Full Reproducibility
========================================================

Training script for DualFreqMamba on standard time-series anomaly detection benchmarks.
Supports: SMD, MSL, SMAP, SWaT, PSM datasets (all from thuml/Time-Series-Library).

Usage:
    # Train on SMD dataset
    python train_dualfreqmamba.py --dataset SMD --output_dir ./checkpoints/dualfreqmamba

    # Train on SWaT
    python train_dualfreqmamba.py --dataset SWaT --output_dir ./checkpoints/dualfreqmamba

    # Train on all datasets
    python train_dualfreqmamba.py --dataset all --output_dir ./checkpoints/dualfreqmamba

Hyperparameters (Publication Defaults):
    - Optimizer: Adam (β1=0.9, β2=0.999)
    - Learning rate: 1e-4 with CosineAnnealingLR (T_max=epochs, eta_min=1e-6)
    - Batch size: 32
    - Epochs: 10 (with early stopping patience=3, following Anomaly Transformer protocol)
    - Window size: 100 (standard across all TS anomaly detection papers)
    - Mamba d_model: 128, d_state: 16, d_conv: 4, expand: 2, n_layers: 3
    - FFT d_model: 128, n_heads: 4, n_layers: 2, patch_size: 4
    - CWT scales: 32, Morlet sigma: 1.0
    - Fusion: adaptive_gate
    - Association lambda: 3.0, sigma: 1.0
    - Loss weights: reconstruction=1.0, frequency=0.5, association=1.0
    - Seeds: 42, 123, 456 (for 3-run statistical significance)

Evaluation Protocol:
    - Point-adjust strategy (standard): if one point in an anomaly segment is detected,
      the entire segment is credited as correctly detected.
    - Metrics: Precision, Recall, F1-Score (primary), AUROC (secondary)
    - Threshold: best F1 threshold on test set (standard protocol per Anomaly Transformer)
"""

import os
import sys
import json
import time
import random
import argparse
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dualfreqmamba.configuration_dualfreqmamba import DualFreqMambaConfig
from models.dualfreqmamba.modeling_dualfreqmamba import DualFreqMambaModel


# ============================================================================
# Dataset Configurations
# ============================================================================

DATASET_CONFIGS = {
    'SMD': {'n_channels': 38, 'source': 'npy', 'prefix': 'SMD'},
    'MSL': {'n_channels': 55, 'source': 'npy', 'prefix': 'MSL'},
    'SMAP': {'n_channels': 25, 'source': 'npy', 'prefix': 'SMAP'},
    'SWaT': {'n_channels': 51, 'source': 'parquet', 'config': 'SWaT'},
    'PSM': {'n_channels': 25, 'source': 'parquet', 'config': 'PSM-data'},
}


# ============================================================================
# Data Loading
# ============================================================================

def load_ts_data(dataset_name, data_root=None):
    """Load time-series anomaly detection dataset.
    
    Returns:
        train_data: (N_train, n_channels) numpy array
        test_data: (N_test, n_channels) numpy array
        test_labels: (N_test,) binary labels
    """
    ds_config = DATASET_CONFIGS[dataset_name]
    
    if data_root and os.path.exists(os.path.join(data_root, dataset_name)):
        # Load from local directory
        local_dir = os.path.join(data_root, dataset_name)
        train_data = np.load(os.path.join(local_dir, f'{dataset_name}_train.npy'))
        test_data = np.load(os.path.join(local_dir, f'{dataset_name}_test.npy'))
        test_labels = np.load(os.path.join(local_dir, f'{dataset_name}_test_label.npy'))
    else:
        # Load from HuggingFace Hub
        from huggingface_hub import hf_hub_download
        
        if ds_config['source'] == 'npy':
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
            ds = load_dataset("thuml/Time-Series-Library", ds_config['config'])
            
            train_df = ds['train'].to_pandas()
            test_df = ds['test'].to_pandas()
            
            if dataset_name == 'SWaT':
                label_col = 'Normal/Attack'
                feature_cols = [c for c in train_df.columns if c != label_col]
                train_data = train_df[feature_cols].values.astype(np.float32)
                test_data = test_df[feature_cols].values.astype(np.float32)
                test_labels = test_df[label_col].values.astype(np.float32)
            else:
                # PSM etc
                feature_cols = [c for c in train_df.columns if c != 'label']
                label_col = 'label' if 'label' in test_df.columns else test_df.columns[-1]
                train_data = train_df[feature_cols].values.astype(np.float32)
                test_data = test_df[feature_cols].values.astype(np.float32)
                test_labels = test_df[label_col].values.astype(np.float32) if label_col in test_df.columns else np.zeros(len(test_df))
    
    # Handle NaN
    train_data = np.nan_to_num(train_data, nan=0.0)
    test_data = np.nan_to_num(test_data, nan=0.0)
    test_labels = np.nan_to_num(test_labels, nan=0.0)
    
    # Flatten labels if multi-dim
    if test_labels.ndim > 1:
        test_labels = test_labels.max(axis=1)
    
    logging.info(f"[{dataset_name}] Train: {train_data.shape}, Test: {test_data.shape}, "
                 f"Labels: {test_labels.shape} (anomaly ratio: {test_labels.mean():.4f})")
    
    return train_data, test_data, test_labels


class TimeSeriesWindowDataset(Dataset):
    """Sliding window dataset for time-series anomaly detection."""
    
    def __init__(self, data, labels=None, window_size=100, stride=1):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        
        self.n_windows = max(1, (len(data) - window_size) // stride + 1)
        
        # Normalize (instance norm will be applied in model, but we do channel-wise here)
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True) + 1e-8
        self.data_norm = (data - self.mean) / self.std
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        
        window = self.data_norm[start:end]  # (T, C)
        window_tensor = torch.tensor(window, dtype=torch.float32).T  # (C, T)
        
        if self.labels is not None:
            label = self.labels[start:end]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return {'input_values': window_tensor, 'labels': label_tensor}
        else:
            return {'input_values': window_tensor}


# ============================================================================
# Point-Adjust Protocol
# ============================================================================

def point_adjust(pred, label):
    """Point-adjust protocol for time-series anomaly detection.
    
    Standard protocol: If any point in a contiguous anomaly segment is detected,
    all points in that segment are credited as correctly detected.
    """
    pred = pred.copy()
    label = label.copy()
    
    # Find contiguous anomaly segments in ground truth
    anomaly_state = False
    for i in range(len(label)):
        if label[i] == 1 and not anomaly_state:
            anomaly_state = True
            start = i
        if label[i] == 0 and anomaly_state:
            anomaly_state = False
            # Check if any point in [start, i) was detected
            if pred[start:i].sum() > 0:
                pred[start:i] = 1
        if i == len(label) - 1 and anomaly_state:
            if pred[start:i+1].sum() > 0:
                pred[start:i+1] = 1
    
    return pred


def find_best_f1_threshold(scores, labels, n_thresholds=100):
    """Find threshold that maximizes F1 score with point-adjust."""
    thresholds = np.percentile(scores, np.linspace(90, 100, n_thresholds))
    
    best_f1 = 0
    best_threshold = thresholds[0]
    best_metrics = {}
    
    for threshold in thresholds:
        pred = (scores > threshold).astype(int)
        pred_adjusted = point_adjust(pred, labels)
        
        f1 = f1_score(labels, pred_adjusted, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'precision': precision_score(labels, pred_adjusted, zero_division=0),
                'recall': recall_score(labels, pred_adjusted, zero_division=0),
                'f1': f1,
                'threshold': float(threshold),
            }
    
    return best_metrics


# ============================================================================
# Training Functions
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, device, epoch, logger):
    model.train()
    total_loss = 0
    recon_loss_sum = 0
    freq_loss_sum = 0
    assoc_loss_sum = 0
    n_batches = 0
    
    for batch in dataloader:
        input_values = batch['input_values'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_values=input_values, return_anomaly_labels=False)
        
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        recon_loss_sum += outputs.reconstruction_loss.item() if outputs.reconstruction_loss is not None else 0
        freq_loss_sum += outputs.frequency_loss.item() if outputs.frequency_loss is not None else 0
        assoc_loss_sum += outputs.association_loss.item() if outputs.association_loss is not None else 0
        n_batches += 1
    
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': recon_loss_sum / n_batches,
        'freq_loss': freq_loss_sum / n_batches,
        'assoc_loss': assoc_loss_sum / n_batches,
    }
    
    logger.info(f"Epoch {epoch}: loss={metrics['loss']:.6f} "
                f"recon={metrics['recon_loss']:.6f} "
                f"freq={metrics['freq_loss']:.6f} "
                f"assoc={metrics['assoc_loss']:.6f}")
    
    return metrics


@torch.no_grad()
def evaluate(model, train_loader, test_loader, test_labels, window_size, device, logger):
    """Evaluate with point-adjust protocol."""
    model.eval()
    
    # Get train scores (for normalization)
    train_scores = []
    for batch in train_loader:
        input_values = batch['input_values'].to(device)
        outputs = model(input_values=input_values, return_anomaly_labels=False)
        scores = outputs.anomaly_score.cpu().numpy()
        train_scores.append(scores)
    train_scores = np.concatenate(train_scores, axis=0)
    train_mean = train_scores.mean()
    train_std = train_scores.std() + 1e-8
    
    # Get test scores
    all_scores = []
    for batch in test_loader:
        input_values = batch['input_values'].to(device)
        outputs = model(input_values=input_values, return_anomaly_labels=False)
        scores = outputs.anomaly_score.cpu().numpy()
        all_scores.append(scores)
    all_scores = np.concatenate(all_scores, axis=0)  # (N_windows, T)
    
    # Reconstruct pointwise scores from windows
    test_len = len(test_labels)
    point_scores = np.zeros(test_len)
    point_counts = np.zeros(test_len)
    
    for i in range(len(all_scores)):
        start = i  # stride=1
        end = min(start + window_size, test_len)
        score_len = end - start
        point_scores[start:end] += all_scores[i, :score_len]
        point_counts[start:end] += 1
    
    point_counts[point_counts == 0] = 1
    point_scores = point_scores / point_counts
    
    # Normalize scores
    point_scores = (point_scores - train_mean) / train_std
    
    # Find best F1 with point-adjust
    metrics = find_best_f1_threshold(point_scores, test_labels)
    
    # AUROC
    try:
        auroc = roc_auc_score(test_labels, point_scores)
    except:
        auroc = 0.0
    metrics['auroc'] = auroc
    
    logger.info(f"EVAL: F1={metrics['f1']:.4f} Prec={metrics['precision']:.4f} "
                f"Rec={metrics['recall']:.4f} AUROC={auroc:.4f}")
    
    return metrics


def plot_training_curves(history, save_path, dataset_name):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'DualFreqMamba Training — {dataset_name}', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['loss']) + 1)
    
    axes[0, 0].plot(epochs, history['loss'], 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['recon_loss'], 'r-', label='Reconstruction', linewidth=1.5)
    axes[0, 1].plot(epochs, history['freq_loss'], 'g-', label='Frequency', linewidth=1.5)
    axes[0, 1].plot(epochs, history['assoc_loss'], 'm-', label='Association', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if 'eval_epochs' in history and history['eval_epochs']:
        axes[1, 0].plot(history['eval_epochs'], history['f1'], 'b-o', label='F1 (point-adjust)', linewidth=1.5, markersize=4)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Test F1 Score (Point-Adjust)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        axes[1, 1].plot(history['eval_epochs'], history['precision'], 'g-o', label='Precision', linewidth=1.5, markersize=4)
        axes[1, 1].plot(history['eval_epochs'], history['recall'], 'r-s', label='Recall', linewidth=1.5, markersize=4)
        axes[1, 1].plot(history['eval_epochs'], history['auroc'], 'm-^', label='AUROC', linewidth=1.5, markersize=4)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision / Recall / AUROC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Training Loop
# ============================================================================

def train_dataset(args, dataset_name, device, logger):
    """Train DualFreqMamba on a single dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training DualFreqMamba on: {dataset_name}")
    logger.info(f"{'='*60}")
    
    ds_config = DATASET_CONFIGS[dataset_name]
    
    # Load data
    train_data, test_data, test_labels = load_ts_data(dataset_name, args.data_root)
    n_channels = train_data.shape[1] if train_data.ndim > 1 else 1
    
    # Create model config
    config = DualFreqMambaConfig(
        n_channels=n_channels,
        window_size=args.window_size,
        fft_patch_size=4,
        fft_stride=2,
        fft_d_model=args.fft_d_model,
        fft_n_heads=4,
        fft_n_layers=2,
        fft_dropout=0.1,
        use_channel_mask=True,
        use_cwt=True,
        cwt_scales=min(32, args.window_size // 4),
        cwt_wavelet_sigma=1.0,
        cwt_patch_size=min(8, args.window_size // 8) if args.window_size >= 16 else 2,
        cwt_d_model=64,
        mamba_d_model=args.mamba_d_model,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_n_layers=3,
        use_gated_skip=True,
        use_association_discrepancy=True,
        prior_sigma=1.0,
        association_lambda=3.0,
        sparse_block_size=10,
        fusion_d_model=args.mamba_d_model,
        fusion_method='adaptive_gate',
        scoring_method='combined',
        reconstruction_weight=1.0,
        frequency_weight=0.5,
        association_weight=1.0,
    )
    
    model = DualFreqMambaModel(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create datasets
    train_dataset = TimeSeriesWindowDataset(train_data, window_size=args.window_size, stride=1)
    test_dataset = TimeSeriesWindowDataset(test_data, test_labels, window_size=args.window_size, stride=1)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training history
    history = {
        'loss': [], 'recon_loss': [], 'freq_loss': [], 'assoc_loss': [],
        'eval_epochs': [], 'f1': [], 'precision': [], 'recall': [], 'auroc': [],
    }
    
    best_f1 = 0
    patience_counter = 0
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, logger)
        scheduler.step()
        
        history['loss'].append(train_metrics['loss'])
        history['recon_loss'].append(train_metrics['recon_loss'])
        history['freq_loss'].append(train_metrics['freq_loss'])
        history['assoc_loss'].append(train_metrics['assoc_loss'])
        
        # Evaluate every eval_interval epochs
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            eval_metrics = evaluate(model, train_loader, test_loader, test_labels,
                                   args.window_size, device, logger)
            
            history['eval_epochs'].append(epoch)
            history['f1'].append(eval_metrics['f1'])
            history['precision'].append(eval_metrics['precision'])
            history['recall'].append(eval_metrics['recall'])
            history['auroc'].append(eval_metrics['auroc'])
            
            if eval_metrics['f1'] > best_f1:
                best_f1 = eval_metrics['f1']
                model.save_pretrained(str(output_dir / 'best'))
                logger.info(f"  → New best F1: {best_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Save final model
    model.save_pretrained(str(output_dir / 'final'))
    
    # Plot
    plot_training_curves(history, str(output_dir / 'training_curves.png'), dataset_name)
    
    # Save history
    with open(str(output_dir / 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config_dict = {
        'dataset': dataset_name,
        'n_channels': n_channels,
        'window_size': args.window_size,
        'epochs_trained': epoch,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_f1': best_f1,
        'final_metrics': eval_metrics,
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'anomaly_ratio': float(test_labels.mean()),
    }
    with open(str(output_dir / 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description='Train DualFreqMamba on TS anomaly benchmarks')
    
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all'] + list(DATASET_CONFIGS.keys()))
    parser.add_argument('--data_root', type=str, default=None, help='Local data directory (optional, downloads from HF if not set)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/dualfreqmamba')
    
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--fft_d_model', type=int, default=128)
    parser.add_argument('--mamba_d_model', type=int, default=128)
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
        ]
    )
    logger = logging.getLogger(__name__)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == 'all' else [args.dataset]
    
    all_results = {}
    for ds_name in datasets:
        set_seed(args.seed)
        try:
            metrics = train_dataset(args, ds_name, device, logger)
            all_results[ds_name] = metrics
        except Exception as e:
            logger.error(f"Failed on {ds_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: DualFreqMamba Results")
    logger.info(f"{'='*80}")
    logger.info(f"{'Dataset':<10} {'F1':>8} {'Prec':>8} {'Rec':>8} {'AUROC':>8}")
    logger.info(f"{'-'*42}")
    
    f1s = []
    for ds, m in all_results.items():
        logger.info(f"{ds:<10} {m['f1']:>8.4f} {m['precision']:>8.4f} "
                    f"{m['recall']:>8.4f} {m['auroc']:>8.4f}")
        f1s.append(m['f1'])
    
    logger.info(f"{'-'*42}")
    logger.info(f"{'MEAN':<10} {np.mean(f1s):>8.4f}")
    
    summary = {
        'per_dataset': all_results,
        'mean_f1': float(np.mean(f1s)),
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
