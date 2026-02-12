"""
Utility functions for data loading.
"""

import os
import json
from torch.utils.data import DataLoader

from .dataset import MedicalSegDataset3D, PatchDataset3D, InferenceDataset3D
from .transforms import get_train_transforms, get_val_transforms


def get_dataloader(
    data_dir,
    split='train',
    batch_size=2,
    patch_size=96,
    num_workers=4,
    domain=None,
    metadata_path=None,
    use_patch_dataset=True,
    target_coverage=5.0,
    foreground_ratio=0.5,
    pin_memory=True,
):
    """
    Create a dataloader for training or validation.
    
    Args:
        data_dir: path to preprocessed data directory
        split: 'train', 'val', or 'test'
        batch_size: batch size
        patch_size: size of patches (int or tuple)
        num_workers: number of data loading workers
        domain: specific domain to load
        metadata_path: path to metadata.json
        use_patch_dataset: whether to use patch-based dataset
        target_coverage: percentage of volume to cover with patches
        foreground_ratio: ratio of foreground patches (for patch dataset)
        pin_memory: whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    
    if split == 'train':
        transform = get_train_transforms(crop_size=patch_size, normalize=True)
        shuffle = True
        
        if use_patch_dataset:
            dataset = PatchDataset3D(
                data_dir=data_dir,
                patch_size=patch_size,
                target_coverage=target_coverage,
                split=split,
                transform=transform,
                domain=domain,
                metadata_path=metadata_path,
                foreground_ratio=foreground_ratio,
            )
        else:
            dataset = MedicalSegDataset3D(
                data_dir=data_dir,
                split=split,
                transform=transform,
                domain=domain,
                metadata_path=metadata_path,
            )
    else:
        transform = get_val_transforms(crop_size=None, normalize=True)
        shuffle = False
        
        dataset = MedicalSegDataset3D(
            data_dir=data_dir,
            split=split,
            transform=transform,
            domain=domain,
            metadata_path=metadata_path,
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train'),
    )
    
    return dataloader


def get_inference_dataloader(
    data_dir,
    patch_size=(96, 96, 96),
    stride=(48, 48, 48),
    batch_size=1,
    num_workers=4,
    domain=None,
    metadata_path=None,
    split=None,
):
    """
    Create a dataloader for inference with sliding window.
    
    Args:
        data_dir: path to preprocessed data directory
        patch_size: size of patches for sliding window
        stride: stride for sliding window
        batch_size: batch size (usually 1 for inference)
        num_workers: number of data loading workers
        domain: specific domain to load
        metadata_path: path to metadata.json
        split: 'train', 'test', or None (all data)
    
    Returns:
        DataLoader instance
    """
    dataset = InferenceDataset3D(
        data_dir=data_dir,
        patch_size=patch_size,
        stride=stride,
        transform=None,
        domain=domain,
        metadata_path=metadata_path,
        split=split,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


def sliding_window_inference(image, patch_size, stride, model, device, num_classes):
    """
    Perform sliding window inference on a 3D volume.
    
    Args:
        image: (1, C, D, H, W) input tensor
        patch_size: (D, H, W) patch size
        stride: (D, H, W) stride
        model: segmentation model
        device: torch device
        num_classes: number of output classes
    
    Returns:
        (D, H, W) predicted segmentation
    """
    import torch
    import torch.nn.functional as F
    
    model.eval()
    
    _, C, D, H, W = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    # Calculate output shape and create accumulation tensors
    output = torch.zeros((num_classes, D, H, W), device=device)
    count = torch.zeros((D, H, W), device=device)
    
    def _forward_patch(patch):
        """Forward pass with logits extraction for consistency with training."""
        # Use return_logits=True to get raw logits (consistent with trainer.test())
        pred = model(patch, return_logits=True)
        if isinstance(pred, tuple):
            # pred[1] is logits, pred[0] is softmax probabilities
            return pred[1]
        return pred
    
    # Sliding window
    with torch.no_grad():
        for d in range(0, D - pd + 1, sd):
            for h in range(0, H - ph + 1, sh):
                for w in range(0, W - pw + 1, sw):
                    # Extract patch
                    patch = image[:, :, d:d+pd, h:h+ph, w:w+pw].to(device)
                    
                    # Forward pass - use logits for accumulation
                    logits = _forward_patch(patch)
                    
                    # Accumulate predictions
                    output[:, d:d+pd, h:h+ph, w:w+pw] += logits[0]
                    count[d:d+pd, h:h+ph, w:w+pw] += 1
        
        # Handle remaining regions (edges)
        # Right edge
        if (D - pd) % sd != 0:
            d = D - pd
            for h in range(0, H - ph + 1, sh):
                for w in range(0, W - pw + 1, sw):
                    patch = image[:, :, d:d+pd, h:h+ph, w:w+pw].to(device)
                    logits = _forward_patch(patch)
                    output[:, d:d+pd, h:h+ph, w:w+pw] += logits[0]
                    count[d:d+pd, h:h+ph, w:w+pw] += 1
        
        # Bottom edge
        if (H - ph) % sh != 0:
            h = H - ph
            for d in range(0, D - pd + 1, sd):
                for w in range(0, W - pw + 1, sw):
                    patch = image[:, :, d:d+pd, h:h+ph, w:w+pw].to(device)
                    logits = _forward_patch(patch)
                    output[:, d:d+pd, h:h+ph, w:w+pw] += logits[0]
                    count[d:d+pd, h:h+ph, w:w+pw] += 1
        
        # Back edge
        if (W - pw) % sw != 0:
            w = W - pw
            for d in range(0, D - pd + 1, sd):
                for h in range(0, H - ph + 1, sh):
                    patch = image[:, :, d:d+pd, h:h+ph, w:w+pw].to(device)
                    logits = _forward_patch(patch)
                    output[:, d:d+pd, h:h+ph, w:w+pw] += logits[0]
                    count[d:d+pd, h:h+ph, w:w+pw] += 1
    
    # Average overlapping predictions
    count = count.clamp(min=1)
    output = output / count.unsqueeze(0)
    
    # Get final segmentation
    segmentation = output.argmax(dim=0)
    
    return segmentation.cpu().numpy()
