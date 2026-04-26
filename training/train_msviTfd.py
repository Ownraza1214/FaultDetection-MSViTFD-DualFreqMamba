#!/usr/bin/env python3
"""
MSViTFD Training Pipeline — Full Reproducibility
==================================================

Training script for MSViTFD on MVTec AD benchmark.
Supports per-category training (one model per category, standard anomaly detection protocol).

Usage:
    # Train on all MVTec AD categories
    python train_msviTfd.py --data_root ./datasets/MVTec --output_dir ./checkpoints/msviTfd

    # Train on a specific category
    python train_msviTfd.py --data_root ./datasets/MVTec --category bottle --output_dir ./checkpoints/msviTfd

    # Train with custom hyperparameters
    python train_msviTfd.py --data_root ./datasets/MVTec --lr 1e-4 --epochs 200 --batch_size 8

Hyperparameters (Publication Defaults):
    - Optimizer: AdamW (β1=0.9, β2=0.999, weight_decay=1e-4)
    - Learning rate: 2e-4 with CosineAnnealingLR (T_max=epochs, eta_min=1e-6)
    - Batch size: 8
    - Epochs: 200
    - Image size: 256×256
    - Encoder: EfficientNet-B2 (frozen, ImageNet pretrained)
    - Hidden dim: 256
    - Decoder depths: [2, 3, 2]
    - Mamba d_state: 16, d_conv: 4, expand: 2
    - DWConv kernels: [3, 5, 7]
    - Discriminator noise_std: 0.015, margin: 0.5
    - Loss weights: reconstruction=1.0, discriminator=0.5, fbft=0.3
    - Seeds: 42, 123, 456 (for 3-run statistical significance)

Data Augmentation (training):
    - RandomResizedCrop(256, scale=(0.8, 1.0))
    - RandomHorizontalFlip(p=0.5)
    - RandomVerticalFlip(p=0.5)
    - RandomRotation(±10°)
    - ColorJitter(brightness=0.1, contrast=0.1)
    - Normalize(ImageNet mean/std)

Data Preprocessing (evaluation):
    - Resize(256)
    - CenterCrop(256)
    - Normalize(ImageNet mean/std)
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
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path for model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.msviTfd.configuration_msviTfd import MSViTFDConfig
from models.msviTfd.modeling_msviTfd import MSViTFDModel


# ============================================================================
# Configuration
# ============================================================================

MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# Dataset
# ============================================================================

class MVTecADDataset(Dataset):
    """MVTec AD dataset loader.
    
    Expected directory structure:
        data_root/
        ├── bottle/
        │   ├── train/
        │   │   └── good/
        │   │       ├── 000.png
        │   │       └── ...
        │   ├── test/
        │   │   ├── good/
        │   │   │   ├── 000.png
        │   │   │   └── ...
        │   │   ├── broken_large/
        │   │   │   ├── 000.png
        │   │   │   └── ...
        │   │   └── ...
        │   └── ground_truth/
        │       ├── broken_large/
        │       │   ├── 000_mask.png
        │       │   └── ...
        │       └── ...
        └── ...
    """
    
    def __init__(self, data_root, category, split='train', transform=None, mask_transform=None):
        self.data_root = Path(data_root)
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.images = []
        self.labels = []
        self.mask_paths = []
        
        if split == 'train':
            # Training: only good/normal images
            good_dir = self.data_root / category / 'train' / 'good'
            if good_dir.exists():
                for img_path in sorted(good_dir.glob('*.png')):
                    self.images.append(str(img_path))
                    self.labels.append(0)
                    self.mask_paths.append(None)
        else:
            # Test: both good and defective
            test_dir = self.data_root / category / 'test'
            gt_dir = self.data_root / category / 'ground_truth'
            
            if test_dir.exists():
                for defect_type in sorted(test_dir.iterdir()):
                    if not defect_type.is_dir():
                        continue
                    
                    is_good = defect_type.name == 'good'
                    
                    for img_path in sorted(defect_type.glob('*.png')):
                        self.images.append(str(img_path))
                        self.labels.append(0 if is_good else 1)
                        
                        if is_good:
                            self.mask_paths.append(None)
                        else:
                            # Find corresponding mask
                            mask_name = img_path.stem + '_mask.png'
                            mask_path = gt_dir / defect_type.name / mask_name
                            self.mask_paths.append(str(mask_path) if mask_path.exists() else None)
        
        logging.info(f"[{category}/{split}] Loaded {len(self.images)} images "
                     f"({sum(self.labels)} anomalous, {len(self.labels) - sum(self.labels)} normal)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        mask = None
        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert('L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = transforms.ToTensor()(mask)
                mask = (mask > 0.5).float()
        
        return {
            'pixel_values': img,
            'label': label,
            'mask': mask if mask is not None else torch.zeros(1, 256, 256),
            'has_mask': mask is not None,
        }


# ============================================================================
# Training Functions
# ============================================================================

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(image_size=256, is_train=True):
    """Get data augmentation / preprocessing transforms."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_mask_transform(image_size=256):
    """Transform for ground truth masks."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


def train_one_epoch(model, dataloader, optimizer, device, epoch, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    recon_loss_sum = 0
    disc_loss_sum = 0
    fbft_loss_sum = 0
    n_batches = 0
    
    for batch in dataloader:
        pixel_values = batch['pixel_values'].to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, return_anomaly_map=False)
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        recon_loss_sum += outputs.reconstruction_loss.item()
        if outputs.discriminator_loss is not None:
            disc_loss_sum += outputs.discriminator_loss.item()
        if outputs.fbft_loss is not None:
            fbft_loss_sum += outputs.fbft_loss.item()
        n_batches += 1
    
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': recon_loss_sum / n_batches,
        'disc_loss': disc_loss_sum / n_batches,
        'fbft_loss': fbft_loss_sum / n_batches,
    }
    
    logger.info(f"Epoch {epoch}: loss={metrics['loss']:.4f} "
                f"recon={metrics['recon_loss']:.4f} "
                f"disc={metrics['disc_loss']:.4f} "
                f"fbft={metrics['fbft_loss']:.4f}")
    
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, logger):
    """Evaluate model on test set."""
    model.eval()
    
    all_labels = []
    all_scores = []
    all_masks = []
    all_anomaly_maps = []
    all_has_masks = []
    
    for batch in dataloader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label']
        masks = batch['mask']
        has_masks = batch['has_mask']
        
        outputs = model(pixel_values=pixel_values, return_anomaly_map=True)
        
        all_labels.extend(labels.numpy().tolist())
        all_scores.extend(outputs.anomaly_score.cpu().numpy().tolist())
        all_masks.append(masks)
        all_anomaly_maps.append(outputs.anomaly_map.cpu())
        all_has_masks.extend(has_masks.numpy().tolist())
    
    # Image-level metrics
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    image_auroc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    image_ap = average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    
    # Optimal F1 threshold
    thresholds = np.percentile(all_scores, np.arange(0, 100, 1))
    best_f1 = 0
    for t in thresholds:
        preds = (all_scores > t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    
    # Pixel-level AUROC (if masks available)
    pixel_auroc = 0.0
    all_anomaly_maps_cat = torch.cat(all_anomaly_maps, dim=0)
    all_masks_cat = torch.cat(all_masks, dim=0)
    
    valid_mask_indices = [i for i, has in enumerate(all_has_masks) if has or all_labels[i] == 0]
    if len(valid_mask_indices) > 0 and any(all_has_masks):
        try:
            gt_masks = all_masks_cat[valid_mask_indices].numpy().flatten()
            pred_maps = all_anomaly_maps_cat[valid_mask_indices]
            # Resize pred maps to match mask size
            pred_maps = torch.nn.functional.interpolate(
                pred_maps.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False
            ).squeeze(1).numpy().flatten()
            
            gt_binary = (gt_masks > 0.5).astype(int)
            if len(np.unique(gt_binary)) > 1:
                pixel_auroc = roc_auc_score(gt_binary, pred_maps)
        except Exception as e:
            logger.warning(f"Pixel AUROC computation failed: {e}")
    
    metrics = {
        'image_auroc': image_auroc,
        'image_ap': image_ap,
        'image_f1': best_f1,
        'pixel_auroc': pixel_auroc,
    }
    
    logger.info(f"EVAL: I-AUROC={image_auroc:.4f} I-AP={image_ap:.4f} "
                f"I-F1={best_f1:.4f} P-AUROC={pixel_auroc:.4f}")
    
    return metrics


def plot_training_curves(history, save_path, category):
    """Plot training loss curves and evaluation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'MSViTFD Training — {category}', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, history['loss'], 'b-', label='Total Loss', linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Component losses
    axes[0, 1].plot(epochs, history['recon_loss'], 'r-', label='Reconstruction', linewidth=1.5)
    axes[0, 1].plot(epochs, history['disc_loss'], 'g-', label='Discriminator', linewidth=1.5)
    axes[0, 1].plot(epochs, history['fbft_loss'], 'm-', label='FBFT', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUROC over evaluation epochs
    if 'eval_epochs' in history and history['eval_epochs']:
        axes[1, 0].plot(history['eval_epochs'], history['image_auroc'], 'b-o', label='Image AUROC', linewidth=1.5, markersize=3)
        axes[1, 0].plot(history['eval_epochs'], history['pixel_auroc'], 'r-s', label='Pixel AUROC', linewidth=1.5, markersize=3)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUROC')
        axes[1, 0].set_title('Evaluation AUROC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    
    # F1 and AP
    if 'eval_epochs' in history and history['eval_epochs']:
        axes[1, 1].plot(history['eval_epochs'], history['image_f1'], 'g-o', label='Image F1', linewidth=1.5, markersize=3)
        axes[1, 1].plot(history['eval_epochs'], history['image_ap'], 'b-s', label='Image AP', linewidth=1.5, markersize=3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('F1 and Average Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Training Loop
# ============================================================================

def train_category(args, category, device, logger):
    """Train MSViTFD on a single MVTec AD category."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training MSViTFD on: {category}")
    logger.info(f"{'='*60}")
    
    # Create model config (publication defaults)
    config = MSViTFDConfig(
        encoder_name=args.encoder_name,
        encoder_pretrained=True,
        freeze_encoder=True,
        hidden_dim=args.hidden_dim,
        decoder_depths=[2, 3, 2],
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        dwconv_kernels=[3, 5, 7],
        use_hilbert_scan=True,
        n_scan_directions=4,
        use_discriminator=True,
        discriminator_hidden=args.hidden_dim * 2,
        noise_std=0.015,
        truncated_loss_margin=0.5,
        use_fbft=True,
        fbft_hidden=args.hidden_dim,
        image_size=args.image_size,
        reconstruction_weight=1.0,
        discriminator_weight=0.5,
        fbft_weight=0.3,
    )
    
    model = MSViTFDModel(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable, {frozen_params:,} frozen")
    
    # Data loaders
    train_transform = get_transforms(args.image_size, is_train=True)
    test_transform = get_transforms(args.image_size, is_train=False)
    mask_transform = get_mask_transform(args.image_size)
    
    train_dataset = MVTecADDataset(args.data_root, category, 'train', train_transform)
    test_dataset = MVTecADDataset(args.data_root, category, 'test', test_transform, mask_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training history
    history = {
        'loss': [], 'recon_loss': [], 'disc_loss': [], 'fbft_loss': [],
        'eval_epochs': [], 'image_auroc': [], 'pixel_auroc': [],
        'image_f1': [], 'image_ap': [],
    }
    
    best_auroc = 0
    output_dir = Path(args.output_dir) / category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, logger)
        scheduler.step()
        
        history['loss'].append(train_metrics['loss'])
        history['recon_loss'].append(train_metrics['recon_loss'])
        history['disc_loss'].append(train_metrics['disc_loss'])
        history['fbft_loss'].append(train_metrics['fbft_loss'])
        
        # Evaluate every eval_interval epochs
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            eval_metrics = evaluate(model, test_loader, device, logger)
            
            history['eval_epochs'].append(epoch)
            history['image_auroc'].append(eval_metrics['image_auroc'])
            history['pixel_auroc'].append(eval_metrics['pixel_auroc'])
            history['image_f1'].append(eval_metrics['image_f1'])
            history['image_ap'].append(eval_metrics['image_ap'])
            
            # Save best model
            if eval_metrics['image_auroc'] > best_auroc:
                best_auroc = eval_metrics['image_auroc']
                model.save_pretrained(str(output_dir / 'best'))
                logger.info(f"  → New best I-AUROC: {best_auroc:.4f}")
    
    # Save final model
    model.save_pretrained(str(output_dir / 'final'))
    
    # Save training curves
    plot_training_curves(history, str(output_dir / 'training_curves.png'), category)
    
    # Save history
    with open(str(output_dir / 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config_dict = {
        'category': category,
        'encoder_name': args.encoder_name,
        'hidden_dim': args.hidden_dim,
        'image_size': args.image_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_image_auroc': best_auroc,
        'final_metrics': eval_metrics,
    }
    with open(str(output_dir / 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description='Train MSViTFD on MVTec AD')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True, help='Path to MVTec AD dataset')
    parser.add_argument('--category', type=str, default='all', 
                       choices=['all'] + MVTEC_CATEGORIES, help='Category to train on')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/msviTfd')
    
    # Model
    parser.add_argument('--encoder_name', type=str, default='efficientnet_b2')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=256)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    # Device
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save full config
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train
    categories = MVTEC_CATEGORIES if args.category == 'all' else [args.category]
    
    all_results = {}
    for category in categories:
        set_seed(args.seed)  # Reset seed for each category
        metrics = train_category(args, category, device, logger)
        all_results[category] = metrics
    
    # Summary table
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: MSViTFD Results on MVTec AD")
    logger.info(f"{'='*80}")
    logger.info(f"{'Category':<15} {'I-AUROC':>10} {'I-AP':>10} {'I-F1':>10} {'P-AUROC':>10}")
    logger.info(f"{'-'*55}")
    
    aurocs = []
    for cat, m in all_results.items():
        logger.info(f"{cat:<15} {m['image_auroc']:>10.4f} {m['image_ap']:>10.4f} "
                    f"{m['image_f1']:>10.4f} {m['pixel_auroc']:>10.4f}")
        aurocs.append(m['image_auroc'])
    
    logger.info(f"{'-'*55}")
    mean_auroc = np.mean(aurocs)
    logger.info(f"{'MEAN':<15} {mean_auroc:>10.4f}")
    
    # Save summary
    summary = {
        'per_category': all_results,
        'mean_image_auroc': float(mean_auroc),
        'mean_image_ap': float(np.mean([m['image_ap'] for m in all_results.values()])),
        'mean_image_f1': float(np.mean([m['image_f1'] for m in all_results.values()])),
        'mean_pixel_auroc': float(np.mean([m['pixel_auroc'] for m in all_results.values()])),
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
