"""
Dataset classes for 3D medical image segmentation.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class MedicalSegDataset3D(Dataset):
    """
    Dataset for 3D medical image segmentation.
    Loads preprocessed data from .npz files.
    """
    
    def __init__(
        self,
        data_dir,
        split='train',
        transform=None,
        domain=None,
        metadata_path=None,
    ):
        """
        Args:
            data_dir: path to preprocessed data directory
            split: 'train', 'val', or 'test'
            transform: data transforms to apply
            domain: specific domain to load (e.g., 'CHAOST2', 'BTCV')
            metadata_path: path to metadata.json file
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.domain = domain
        
        # Load metadata if available
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Get list of data files
        self.data_files = self._get_data_files()
        
        if len(self.data_files) == 0:
            raise ValueError(f"No data files found in {data_dir} for split '{split}'")
    
    def _get_data_files(self):
        """Get list of data files for the specified split."""
        files = []
        
        # Check if metadata specifies split
        if self.metadata and 'splits' in self.metadata:
            if self.split in self.metadata['splits']:
                case_ids = self.metadata['splits'][self.split]
                if self.domain:
                    case_ids = [c for c in case_ids if self.domain in c]
                for case_id in case_ids:
                    file_path = os.path.join(self.data_dir, f"{case_id}.npz")
                    if os.path.exists(file_path):
                        files.append(file_path)
        else:
            # Fall back to directory listing
            if os.path.exists(self.data_dir):
                for f in os.listdir(self.data_dir):
                    if f.endswith('.npz'):
                        if self.domain is None or self.domain in f:
                            files.append(os.path.join(self.data_dir, f))
        
        return sorted(files)
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load data
        data = np.load(self.data_files[idx])
        image = data['image'].astype(np.float32)
        label = data['label'].astype(np.int64)
        
        # Apply transforms
        if self.transform:
            image, label = self.transform(image, label)
        else:
            # Default: convert to tensor
            if image.ndim == 3:
                image = image[np.newaxis, ...]
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)
        
        # Get case info
        case_id = os.path.basename(self.data_files[idx]).replace('.npz', '')
        
        return {
            'image': image,
            'label': label,
            'case_id': case_id,
            'file_path': self.data_files[idx]
        }
    
    def get_num_classes(self):
        """Get number of classes from metadata or data."""
        if self.metadata and 'num_classes' in self.metadata:
            return self.metadata['num_classes']
        
        # Infer from first sample
        data = np.load(self.data_files[0])
        return int(data['label'].max()) + 1
    
    def get_class_names(self):
        """Get class names from metadata."""
        if self.metadata and 'class_names' in self.metadata:
            return self.metadata['class_names']
        return None


class PatchDataset3D(Dataset):
    """
    Dataset that extracts patches from 3D volumes.
    Useful for training when full volumes don't fit in memory.
    """
    
    def __init__(
        self,
        data_dir,
        patch_size=(96, 96, 96),
        target_coverage=5.0,
        split='train',
        transform=None,
        domain=None,
        metadata_path=None,
        foreground_ratio=0.5,
    ):
        """
        Args:
            data_dir: path to preprocessed data directory
            patch_size: size of patches to extract (D, H, W)
            target_coverage: percentage of volume to cover with patches
            split: 'train', 'val', or 'test'
            transform: data transforms to apply
            domain: specific domain to load
            metadata_path: path to metadata.json file
            foreground_ratio: ratio of patches that should contain foreground
        """
        self.data_dir = data_dir
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size,) * 3
        self.target_coverage = target_coverage
        self.split = split
        self.transform = transform
        self.domain = domain
        self.foreground_ratio = foreground_ratio
        
        # Load metadata
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Get data files
        self.data_files = self._get_data_files()
        
        # Pre-compute patch locations for each volume
        self._precompute_patch_locations()
    
    def _get_data_files(self):
        """Get list of data files."""
        files = []
        
        if self.metadata and 'splits' in self.metadata:
            if self.split in self.metadata['splits']:
                case_ids = self.metadata['splits'][self.split]
                if self.domain:
                    case_ids = [c for c in case_ids if self.domain in c]
                for case_id in case_ids:
                    file_path = os.path.join(self.data_dir, f"{case_id}.npz")
                    if os.path.exists(file_path):
                        files.append(file_path)
        else:
            if os.path.exists(self.data_dir):
                for f in os.listdir(self.data_dir):
                    if f.endswith('.npz'):
                        if self.domain is None or self.domain in f:
                            files.append(os.path.join(self.data_dir, f))
        
        return sorted(files)
    
    def _adaptive_patch_config(self, volume_size, target_coverage=5.0):
        volume_voxels = volume_size[0] * volume_size[1] * volume_size[2]
        patch_voxels = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        required_patches = int((volume_voxels * target_coverage) / patch_voxels)
        patches_per_volume = min(required_patches, 500)

        return patches_per_volume
    
    def _precompute_patch_locations(self):
        """Pre-compute valid patch locations for each volume."""
        self.patch_locations = []
        self.patches_per_volume = []
        
        for file_path in self.data_files:
            data = np.load(file_path)
            label = data['label']
            shape = label.shape

            for i in range(3):
                if self.patch_size[i] > shape[i]:
                    print(f"Warning: patch_size={self.patch_size} > volume_size={shape} for case {file_path}")
            
            # Get foreground locations
            foreground_mask = label > 0
            foreground_coords = np.argwhere(foreground_mask)

            # Get adaptive patch configuration
            patches_this_volume = self._adaptive_patch_config(shape, self.target_coverage)
            self.patches_per_volume.append(patches_this_volume)
            
            volume_patches = []
            
            for _ in range(patches_this_volume):
                if len(foreground_coords) > 0 and np.random.random() < self.foreground_ratio:
                    # Sample patch centered on foreground
                    idx = np.random.randint(len(foreground_coords))
                    center = foreground_coords[idx]
                else:
                    # Random patch
                    center = np.array([
                        np.random.randint(self.patch_size[0] // 2, shape[0] - self.patch_size[0] // 2),
                        np.random.randint(self.patch_size[1] // 2, shape[1] - self.patch_size[1] // 2),
                        np.random.randint(self.patch_size[2] // 2, shape[2] - self.patch_size[2] // 2),
                    ])
                
                # Compute patch bounds
                start = np.maximum(center - np.array(self.patch_size) // 2, 0)
                end = start + np.array(self.patch_size)
                
                # Adjust if patch exceeds volume bounds
                for i in range(3):
                    if end[i] > shape[i]:
                        end[i] = shape[i]
                        start[i] = max(0, end[i] - self.patch_size[i])
                
                volume_patches.append((start, end))
            
            self.patch_locations.append(volume_patches)
    
    def __len__(self):
        return sum(self.patches_per_volume)
    
    def __getitem__(self, idx):
        volume_idx = np.argmax(np.cumsum(self.patches_per_volume) > idx)
        patch_idx = idx - np.sum(self.patches_per_volume[:volume_idx])
        
        # Load volume
        data = np.load(self.data_files[volume_idx])
        image = data['image'].astype(np.float32)
        label = data['label'].astype(np.int64)
        
        # Extract patch
        start, end = self.patch_locations[volume_idx][patch_idx]
        image_patch = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        label_patch = label[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        # Apply transforms
        if self.transform:
            image_patch, label_patch = self.transform(image_patch, label_patch)
        else:
            if image_patch.ndim == 3:
                image_patch = image_patch[np.newaxis, ...]
            image_patch = torch.from_numpy(image_patch)
            label_patch = torch.from_numpy(label_patch)
        
        case_id = os.path.basename(self.data_files[volume_idx]).replace('.npz', '')
        
        return {
            'image': image_patch,
            'label': label_patch,
            'case_id': case_id,
            'patch_idx': patch_idx,
            'patch_location': (start.tolist(), end.tolist())
        }


class InferenceDataset3D(Dataset):
    """
    Dataset for inference that returns full volumes with sliding window info.
    """
    
    def __init__(
        self,
        data_dir,
        patch_size=(96, 96, 96),
        stride=(48, 48, 48),
        transform=None,
        domain=None,
        metadata_path=None,
        split=None,
    ):
        """
        Args:
            data_dir: path to preprocessed data directory
            patch_size: size of patches for sliding window
            stride: stride for sliding window
            transform: data transforms (should not include random augmentation)
            domain: specific domain to load
            metadata_path: path to metadata.json file
            split: 'train', 'test', or None (all data)
        """
        self.data_dir = data_dir
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.transform = transform
        self.domain = domain
        self.split = split
        
        # Load metadata
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Get data files
        self.data_files = self._get_data_files()
    
    def _get_data_files(self):
        """Get list of data files."""
        files = []
        
        # If split is specified and metadata has splits info, use it
        if self.split and self.metadata and 'splits' in self.metadata:
            if self.split in self.metadata['splits']:
                case_ids = self.metadata['splits'][self.split]
                if self.domain:
                    case_ids = [c for c in case_ids if self.domain in c]
                for case_id in case_ids:
                    file_path = os.path.join(self.data_dir, f"{case_id}.npz")
                    if os.path.exists(file_path):
                        files.append(file_path)
        else:
            # Fall back to directory listing (all files)
            if os.path.exists(self.data_dir):
                for f in os.listdir(self.data_dir):
                    if f.endswith('.npz'):
                        if self.domain is None or self.domain in f:
                            files.append(os.path.join(self.data_dir, f))
        
        return sorted(files)
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load full volume
        data = np.load(self.data_files[idx])
        image = data['image'].astype(np.float32)
        label = data['label'].astype(np.int64)
        
        # Get spacing if available
        spacing = data['spacing'] if 'spacing' in data else np.array([1.0, 1.0, 1.0])
        
        # Normalize image
        if self.transform:
            # Only apply normalization, not cropping
            from .transforms import Normalize
            norm = Normalize()
            image, label = norm(image, label)
        
        # Add channel dimension
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        case_id = os.path.basename(self.data_files[idx]).replace('.npz', '')
        
        return {
            'image': image,
            'label': label,
            'case_id': case_id,
            'spacing': spacing,
            'original_shape': label.shape,
            'patch_size': self.patch_size,
            'stride': self.stride,
        }
