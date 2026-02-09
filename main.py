"""
Main entry point for SFDA 3D Medical Image Segmentation.

This file provides backward compatibility. For new usage, please use:
    - train.py: For training models
    - predict.py: For evaluation and prediction

Usage examples:
    # Training (recommended to use train.py directly)
    python main.py --mode train --method source_pretrain --dataset ABDOMINAL --domain CHAOST2
    
    # Evaluation (recommended to use predict.py directly)
    python main.py --mode eval --checkpoint_path ./path/to/checkpoint.pth --dataset ABDOMINAL --domain CHAOST2

Note: Data preprocessing should be done via the dataset-specific preprocessor script:
    python datasets/ABDOMINAL/abdominal_preprocessor.py
"""

import sys
import os


def main():
    # Check if --mode is provided
    if '--mode' not in sys.argv:
        print("=" * 60)
        print("SFDA 3D Medical Image Segmentation")
        print("=" * 60)
        print("\nPlease use the dedicated entry points:")
        print("  - python train.py    : For training models")
        print("  - python predict.py  : For evaluation and prediction")
        print("\nOr use this script with --mode argument:")
        print("  - python main.py --mode train ...")
        print("  - python main.py --mode eval ...")
        print("=" * 60)
        return
    
    # Find mode argument
    mode_idx = sys.argv.index('--mode')
    if mode_idx + 1 >= len(sys.argv):
        print("Error: --mode requires an argument (train or eval)")
        return
    
    mode = sys.argv[mode_idx + 1]
    
    # Remove --mode and its value from argv for the sub-script
    new_argv = [sys.argv[0]] + sys.argv[1:mode_idx] + sys.argv[mode_idx + 2:]
    sys.argv = new_argv
    
    if mode == 'train':
        print("Redirecting to train.py...")
        print("(For future use, please run: python train.py ...)")
        print()
        
        # Import and run train
        from train import main as train_main
        train_main()
        
    elif mode == 'eval':
        print("Redirecting to predict.py...")
        print("(For future use, please run: python predict.py ...)")
        print()
        
        # Import and run predict
        from predict import main as predict_main
        predict_main()
        
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'train' or 'eval'.")


if __name__ == '__main__':
    main()
