"""
Prediction/Evaluation entry point for SFDA 3D Medical Image Segmentation.

Usage examples:
    # Evaluate on test set only (default)
    python predict.py --checkpoint_path ./results/ABDOMINAL/CHAOST2/src_chaost2_v1/checkpoints/best_checkpoint.pth \
        --dataset ABDOMINAL --domain CHAOST2
    
    # Evaluate on all data (train + test)
    python predict.py --checkpoint_path ./results/ABDOMINAL/CHAOST2/src_chaost2_v1/checkpoints/best_checkpoint.pth \
        --dataset ABDOMINAL --domain CHAOST2 --eval_mode all
    
    # Evaluate and save predictions
    python predict.py --checkpoint_path ./results/ABDOMINAL/CHAOST2/src_chaost2_v1/checkpoints/best_checkpoint.pth \
        --dataset ABDOMINAL --domain CHAOST2 --save_predictions
    
    # Evaluate with custom patch size and stride
    python predict.py --checkpoint_path ./path/to/checkpoint.pth \
        --dataset ABDOMINAL --domain BTCV \
        --patch_size 64 64 64 --stride 32 32 32
"""

import os
import argparse
import json
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='SFDA 3D Medical Image Segmentation - Prediction/Evaluation')
    
    # Dataset and domain
    parser.add_argument('--dataset', type=str, default='ABDOMINAL',
                        help='Dataset name')
    parser.add_argument('--domain', type=str, default=None,
                        help='Domain name (e.g., CHAOST2, BTCV)')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='Base directory for datasets')
    parser.add_argument('--output_dir', type=str, default=None, required=True,
                        help='Directory to save predictions (default: auto-generated)')
    parser.add_argument('--checkpoint_path', '-ckpt', type=str,
                        help='Path to model checkpoint (default best)')
    
    # Model (should match the trained model)
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
    
    # Inference settings
    parser.add_argument('--patch_size', type=int, default=None,
                        help='Patch size for sliding window inference (Auto load from config.json)')
    parser.add_argument('--stride', type=int, nargs='+', default=None,
                        help='Stride for sliding window inference (D H W or single value). If not specified, auto-calculated for 50%% overlap')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--eval_mode', type=str, default='testset',
                        choices=['testset', 'all'],
                        help='Evaluation mode: testset (only test split) or all (all data)')
    
    # Output options
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction results as numpy files')
    parser.add_argument('--save_nifti', action='store_true',
                        help='Save predictions as NIfTI files for visualization')
    
    # Data loading
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
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


def parse_size_arg(size_arg):
    """Parse size argument (patch_size or stride) to tuple of 3 integers."""
    if isinstance(size_arg, int):
        return (size_arg, size_arg, size_arg)
    elif isinstance(size_arg, (list, tuple)):
        if len(size_arg) == 1:
            return (size_arg[0], size_arg[0], size_arg[0])
        elif len(size_arg) == 3:
            return tuple(size_arg)
        else:
            raise ValueError(f"Size argument must have 1 or 3 values, got {len(size_arg)}")
    else:
        raise ValueError(f"Invalid size argument type: {type(size_arg)}")


def create_model(args):
    """Create model based on arguments."""
    from models import UNet3D, ResidualUNet3D
    
    model_classes = {
        'UNet3D': UNet3D,
        'ResidualUNet3D': ResidualUNet3D,
    }
    
    model_class = model_classes[args.model]
    model = model_class(
        in_channels=args.in_channels,
        out_channels=args.num_classes,
        f_maps=args.f_maps,
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


def save_as_nifti(prediction, case_id, output_dir, spacing=(1.0, 1.0, 1.0)):
    """Save prediction as NIfTI file."""
    try:
        import nibabel as nib
        
        # Create affine matrix
        affine = np.diag([-spacing[2], -spacing[1], spacing[0], 1])
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(prediction.astype(np.int16), affine)
        
        # Save
        output_path = os.path.join(output_dir, f'{case_id}_pred.nii.gz')
        nib.save(nifti_img, output_path)
        print(f"  Saved NIfTI: {output_path}")
        
    except ImportError:
        print("Warning: nibabel not installed. Cannot save NIfTI files.")


def main():
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 60)
    print("SFDA 3D Medical Image Segmentation - Prediction/Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Domain: {args.domain}")
    print(f"Eval Mode: {args.eval_mode}")
    print(f"Device: {args.device}")
    print("=" * 60 + "\n")

    # Setup output directory
    output_dir = args.output_dir
    if output_dir is None and (args.save_predictions or args.save_nifti):
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        output_dir = os.path.join(os.path.dirname(checkpoint_dir), 'predictions')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Validate checkpoint path
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(output_dir, 'checkpoints', 'best_checkpoint.pth')
    else:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        
    # Set seed
    set_seed(args.seed)
    
    # Parse sizes
    if args.patch_size is None:
        with open(os.path.join(output_dir, 'config.json'), 'r') as f:
            config = json.load(f)
            patch_size = config.get('patch_size', [64, 96, 96])
    else:
        patch_size = parse_size_arg(args.patch_size)
    
    # Auto-calculate stride for 50% overlap if not specified
    if args.stride is None:
        stride = tuple(p // 2 for p in patch_size)
        print(f"Patch size: {patch_size}")
        print(f"Stride: {stride} (auto-calculated for 50% overlap)")
    else:
        stride = parse_size_arg(args.stride)
        print(f"Patch size: {patch_size}")
        print(f"Stride: {stride}")
    
    # Get paths
    paths = get_data_paths(args)
    
    # Check if preprocessed data exists
    if not os.path.exists(paths['preprocessed_dir']):
        raise FileNotFoundError(
            f"Preprocessed data not found at {paths['preprocessed_dir']}. "
            f"Please run the preprocessor first."
        )
    
    # Create model
    model = create_model(args)
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loader
    from dataloader import get_inference_dataloader
    
    # Determine split based on eval_mode
    split = 'test' if args.eval_mode == 'testset' else None
    
    test_loader = get_inference_dataloader(
        data_dir=paths['preprocessed_dir'],
        patch_size=patch_size,
        stride=stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        domain=args.domain,
        metadata_path=paths['metadata_path'],
        split=split,
    )
    
    print(f"Number of cases to evaluate: {len(test_loader.dataset)}")
    
    # Load metadata for class names and spacing
    class_names = None
    target_spacing = (1.0, 1.0, 1.0)
    if os.path.exists(paths['metadata_path']):
        with open(paths['metadata_path'], 'r') as f:
            metadata = json.load(f)
            class_names = metadata.get('class_names', None)
            target_spacing = tuple(metadata.get('target_spacing', [1.0, 1.0, 1.0]))
    
    # Run evaluation
    from trainer.evaluator import Evaluator
    
    evaluator = Evaluator(
        model=model,
        device=args.device,
        num_classes=args.num_classes,
        patch_size=patch_size,
        stride=stride,
        save_predictions=args.save_predictions,
        output_dir=output_dir,
    )
    
    # Load checkpoint
    evaluator.load_checkpoint(args.checkpoint_path)
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate(test_loader, class_names=class_names)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    if output_dir:
        mode_suffix = 'all' if args.eval_mode == 'all' else 'test'
        results_path = os.path.join(output_dir, f'evaluation_results_{args.domain}_{mode_suffix}.json')
        evaluator.save_results(results, results_path)
    
    # Save as NIfTI if requested
    if args.save_nifti and output_dir:
        print("\nSaving predictions as NIfTI files...")
        for case_result in results['per_case_results']:
            case_id = case_result['case_id']
            pred_path = os.path.join(output_dir, f'{case_id}_pred.npy')
            if os.path.exists(pred_path):
                prediction = np.load(pred_path)
                save_as_nifti(prediction, case_id, output_dir, spacing=target_spacing)
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    
    # Return overall dice for scripting purposes
    overall_dice = results['summary'].get('Overall', {}).get('dice_mean', 0.0)
    print(f"\nOverall Dice: {overall_dice:.4f}")
    
    return results


if __name__ == '__main__':
    main()
