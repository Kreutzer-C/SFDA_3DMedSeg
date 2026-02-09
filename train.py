"""
Training entry point for SFDA 3D Medical Image Segmentation.

Usage examples:
    # Source domain pretraining
    python train.py --method source_pretrain --dataset ABDOMINAL --domain CHAOST2 --exp_name src_chaost2_v1
    
    # Oracle adaptation (upper bound)
    python train.py --method oracle_adapt --dataset ABDOMINAL --domain BTCV --exp_name oracle_btcv_v1 \
        --pretrained_path ./results/ABDOMINAL/CHAOST2/src_chaost2_v1/checkpoints/best_checkpoint.pth

Note: Data preprocessing should be done via the dataset-specific preprocessor script:
    python datasets/ABDOMINAL/abdominal_preprocessor.py
"""

import os
import argparse
import json
import torch
import numpy as np
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='SFDA 3D Medical Image Segmentation - Training')
    
    # Method
    parser.add_argument('--method', type=str, default='source_pretrain',
                        choices=['source_pretrain', 'oracle_adapt'],
                        help='Training method')
    
    # Dataset and domain
    parser.add_argument('--dataset', type=str, default='ABDOMINAL',
                        help='Dataset name')
    parser.add_argument('--domain', type=str, default=None,
                        help='Domain name (e.g., CHAOST2, BTCV)')
    
    # Experiment
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Base directory for datasets')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained model checkpoint (for oracle_adapt)')
    
    # Model (default configuration aligned with SFDA-DDFP project)
    parser.add_argument('--model', type=str, default='UNet3D',
                        choices=['UNet3D', 'ResidualUNet3D'],
                        help='Model architecture')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of segmentation classes')
    parser.add_argument('--f_maps', type=int, nargs='+', default=[64,128,256,512,512],
                        help='Feature maps per level')
    parser.add_argument('--num_levels', type=int, default=5,
                        help='Number of levels in U-Net')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[64, 96, 96],
                        help='Patch size for training (D H W or single value)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    
    # Data loading
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--patches_per_volume', type=int, default=4,
                        help='Number of patches per volume')
    parser.add_argument('--foreground_ratio', type=float, default=0.7,
                        help='Ratio of foreground patches')
    
    # Logging
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='SFDA_3DMedSeg',
                        help='W&B project name')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_patch_size(patch_size):
    """Parse patch_size argument to tuple of 3 integers."""
    if isinstance(patch_size, int):
        return (patch_size, patch_size, patch_size)
    elif isinstance(patch_size, (list, tuple)):
        if len(patch_size) == 1:
            return (patch_size[0], patch_size[0], patch_size[0])
        elif len(patch_size) == 3:
            return tuple(patch_size)
        else:
            raise ValueError(f"patch_size must have 1 or 3 values, got {len(patch_size)}")
    else:
        raise ValueError(f"Invalid patch_size type: {type(patch_size)}")


def create_model(args):
    """Create model based on arguments."""
    from models import UNet3D, ResidualUNet3D
    
    model_classes = {
        'UNet3D': UNet3D,
        'ResidualUNet3D': ResidualUNet3D,
    }
    
    model_class = model_classes[args.model]
    
    # Handle f_maps: None uses DDFP-aligned default [64,128,256,512,512]
    f_maps = args.f_maps
    if f_maps is not None and len(f_maps) == 1:
        # If single value provided, use it as init_channel_number
        f_maps = f_maps[0]
    
    model = model_class(
        in_channels=args.in_channels,
        out_channels=args.num_classes,
        f_maps=f_maps,
        num_levels=args.num_levels,
        final_sigmoid=False,
        is_segmentation=True,
    )
    
    return model


def get_data_paths(args):
    """Get data paths based on arguments."""
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    preprocessed_dir = os.path.join(dataset_dir, 'preprocessed_new')
    metadata_path = os.path.join(dataset_dir, 'metadata.json')
    
    return {
        'dataset_dir': dataset_dir,
        'preprocessed_dir': preprocessed_dir,
        'metadata_path': metadata_path,
    }


def main():
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 60)
    print("SFDA 3D Medical Image Segmentation - Training")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Domain: {args.domain}")
    print(f"Device: {args.device}")
    print("=" * 60 + "\n")
    
    # Set seed
    set_seed(args.seed)
    
    # Parse patch_size
    patch_size = parse_patch_size(args.patch_size)
    print(f"Patch size: {patch_size}")
    
    # Get paths
    paths = get_data_paths(args)
    
    # Check if preprocessed data exists
    if not os.path.exists(paths['preprocessed_dir']):
        raise FileNotFoundError(
            f"Preprocessed data not found at {paths['preprocessed_dir']}. "
            f"Please run the preprocessor first:\n"
            f"  python datasets/{args.dataset}/abdominal_preprocessor.py"
        )
    
    # Create model
    model = create_model(args)
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    from dataloader import get_dataloader
    
    train_loader = get_dataloader(
        data_dir=paths['preprocessed_dir'],
        split='train',
        batch_size=args.batch_size,
        patch_size=patch_size,
        num_workers=args.num_workers,
        domain=args.domain,
        metadata_path=paths['metadata_path'],
        use_patch_dataset=True,
        patches_per_volume=args.patches_per_volume,
        foreground_ratio=args.foreground_ratio,
    )
    
    test_loader = get_dataloader(
        data_dir=paths['preprocessed_dir'],
        split='test',
        batch_size=1,
        patch_size=patch_size,
        num_workers=args.num_workers,
        domain=args.domain,
        metadata_path=paths['metadata_path'],
        use_patch_dataset=False,
    )
    
    # Generate experiment name if not provided
    if args.exp_name is None:
        args.exp_name = f"{args.method}_{args.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create config
    config = {
        'dataset_name': args.dataset,
        'domain_name': args.domain or 'all',
        'exp_name': args.exp_name,
        'results_dir': args.results_dir,
        'model': args.model,
        'num_classes': args.num_classes,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'patch_size': patch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'device': args.device,
        'use_wandb': not args.disable_wandb,
        'wandb_project': args.wandb_project,
        'save_every': args.save_every,
        'scheduler': {'name': 'CosineAnnealingLR'},
    }
    
    # Create trainer based on method
    if args.method == 'source_pretrain':
        from trainer.source_trainer import create_source_trainer
        
        trainer = create_source_trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
        )
    
    elif args.method == 'oracle_adapt':
        from trainer.target_trainer import create_oracle_trainer
        
        if args.pretrained_path is None:
            print("Warning: No pretrained_path provided for oracle_adapt. Training from scratch.")
        
        trainer = create_oracle_trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            pretrained_path=args.pretrained_path,
        )
    
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Run training
    print(f"\nStarting training for {args.num_epochs} epochs...")
    trainer.train(num_epochs=args.num_epochs)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
