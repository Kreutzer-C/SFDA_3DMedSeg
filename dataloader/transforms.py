"""
Data augmentation transforms for 3D medical image segmentation.
All transforms maintain spatial consistency between image and label.
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import rotate, zoom, map_coordinates, gaussian_filter


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""
    
    def __call__(self, image, label):
        # Add channel dimension if needed
        if image.ndim == 3:
            image = image[np.newaxis, ...]  # (1, D, H, W)
        
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.int64))
        
        return image, label


class Normalize:
    """
    Normalize image.
    
    Supports two modes:
    - 'zscore': zero mean and unit variance (default for raw data)
    - 'minmax': already normalized to [0, 1] (for preprocessed data)
    """
    
    def __init__(self, mode='minmax', mean=None, std=None, clip_range=None):
        """
        Args:
            mode: 'zscore' or 'minmax'
            mean: mean for zscore normalization (auto if None)
            std: std for zscore normalization (auto if None)
            clip_range: optional clipping range
        """
        self.mode = mode
        self.mean = mean
        self.std = std
        self.clip_range = clip_range
    
    def __call__(self, image, label):
        if self.clip_range is not None:
            image = np.clip(image, self.clip_range[0], self.clip_range[1])
        
        if self.mode == 'zscore':
            if self.mean is None:
                mean = np.mean(image)
            else:
                mean = self.mean
                
            if self.std is None:
                std = np.std(image)
                std = std if std > 0 else 1.0
            else:
                std = self.std
            
            image = (image - mean) / std
        # For 'minmax' mode, data is already in [0, 1], no transformation needed
        
        return image, label


class RandomCrop3D:
    """Random crop for 3D volumes."""
    
    def __init__(self, crop_size):
        """
        Args:
            crop_size: tuple (D, H, W) or int for cubic crop
        """
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size, crop_size)
        else:
            self.crop_size = crop_size
    
    def __call__(self, image, label):
        d, h, w = image.shape[-3:]
        cd, ch, cw = self.crop_size
        
        # Ensure crop size doesn't exceed image size
        cd = min(cd, d)
        ch = min(ch, h)
        cw = min(cw, w)
        
        # Random start positions
        d_start = np.random.randint(0, d - cd + 1) if d > cd else 0
        h_start = np.random.randint(0, h - ch + 1) if h > ch else 0
        w_start = np.random.randint(0, w - cw + 1) if w > cw else 0
        
        if image.ndim == 4:  # (C, D, H, W)
            image = image[:, d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]
        else:  # (D, H, W)
            image = image[d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]
        
        label = label[d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]
        
        return image, label


class CenterCrop3D:
    """Center crop for 3D volumes."""
    
    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size, crop_size)
        else:
            self.crop_size = crop_size
    
    def __call__(self, image, label):
        d, h, w = image.shape[-3:]
        cd, ch, cw = self.crop_size
        
        cd = min(cd, d)
        ch = min(ch, h)
        cw = min(cw, w)
        
        d_start = (d - cd) // 2
        h_start = (h - ch) // 2
        w_start = (w - cw) // 2
        
        if image.ndim == 4:
            image = image[:, d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]
        else:
            image = image[d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]
        
        label = label[d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]
        
        return image, label


class RandomFlip3D:
    """Random flip along each axis."""
    
    def __init__(self, p=0.5, axes=(0, 1, 2)):
        """
        Args:
            p: probability of flip for each axis
            axes: axes to potentially flip (0=D, 1=H, 2=W)
        """
        self.p = p
        self.axes = axes
    
    def __call__(self, image, label):
        for axis in self.axes:
            if np.random.random() < self.p:
                if image.ndim == 4:
                    image = np.flip(image, axis=axis+1).copy()
                else:
                    image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        
        return image, label


class RandomRotation3D:
    """Random rotation in 3D."""
    
    def __init__(self, angle_range=(-15, 15), axes=((0, 1), (0, 2), (1, 2)), p=0.5):
        """
        Args:
            angle_range: (min_angle, max_angle) in degrees
            axes: rotation planes
            p: probability of rotation
        """
        self.angle_range = angle_range
        self.axes = axes
        self.p = p
    
    def __call__(self, image, label):
        if np.random.random() > self.p:
            return image, label
        
        # Choose random rotation plane
        plane = self.axes[np.random.randint(len(self.axes))]
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        
        if image.ndim == 4:
            # Rotate each channel
            rotated_image = np.zeros_like(image)
            for c in range(image.shape[0]):
                rotated_image[c] = rotate(image[c], angle, axes=plane, reshape=False, order=1, mode='constant')
        else:
            rotated_image = rotate(image, angle, axes=plane, reshape=False, order=1, mode='constant')
        
        rotated_label = rotate(label, angle, axes=plane, reshape=False, order=0, mode='constant')
        
        return rotated_image, rotated_label


class RandomScale3D:
    """Random scaling in 3D."""
    
    def __init__(self, scale_range=(0.85, 1.15), p=0.5):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, image, label):
        if np.random.random() > self.p:
            return image, label
        
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        if image.ndim == 4:
            scaled_image = np.zeros_like(image)
            for c in range(image.shape[0]):
                scaled_image[c] = zoom(image[c], scale, order=1, mode='constant')
        else:
            scaled_image = zoom(image, scale, order=1, mode='constant')
        
        scaled_label = zoom(label, scale, order=0, mode='constant')
        
        # Crop or pad to original size
        original_shape = image.shape[-3:]
        scaled_image, scaled_label = self._adjust_size(scaled_image, scaled_label, original_shape)
        
        return scaled_image, scaled_label
    
    def _adjust_size(self, image, label, target_shape):
        """Crop or pad to target shape."""
        current_shape = image.shape[-3:]
        
        # Calculate padding/cropping for each dimension
        result_image = np.zeros(image.shape[:-3] + target_shape if image.ndim == 4 else target_shape, dtype=image.dtype)
        result_label = np.zeros(target_shape, dtype=label.dtype)
        
        slices_src = []
        slices_dst = []
        
        for i in range(3):
            if current_shape[i] > target_shape[i]:
                # Crop
                start = (current_shape[i] - target_shape[i]) // 2
                slices_src.append(slice(start, start + target_shape[i]))
                slices_dst.append(slice(None))
            else:
                # Pad
                start = (target_shape[i] - current_shape[i]) // 2
                slices_src.append(slice(None))
                slices_dst.append(slice(start, start + current_shape[i]))
        
        if image.ndim == 4:
            result_image[:, slices_dst[0], slices_dst[1], slices_dst[2]] = \
                image[:, slices_src[0], slices_src[1], slices_src[2]]
        else:
            result_image[slices_dst[0], slices_dst[1], slices_dst[2]] = \
                image[slices_src[0], slices_src[1], slices_src[2]]
        
        result_label[slices_dst[0], slices_dst[1], slices_dst[2]] = \
            label[slices_src[0], slices_src[1], slices_src[2]]
        
        return result_image, result_label


class RandomGaussianNoise:
    """Add random Gaussian noise to image."""
    
    def __init__(self, mean=0, std_range=(0, 0.1), p=0.5):
        self.mean = mean
        self.std_range = std_range
        self.p = p
    
    def __call__(self, image, label):
        if np.random.random() > self.p:
            return image, label
        
        std = np.random.uniform(self.std_range[0], self.std_range[1])
        noise = np.random.normal(self.mean, std, image.shape)
        image = image + noise
        
        return image.astype(np.float32), label


class RandomGaussianBlur:
    """Apply random Gaussian blur to image."""
    
    def __init__(self, sigma_range=(0.5, 1.5), p=0.5):
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, image, label):
        if np.random.random() > self.p:
            return image, label
        
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        
        if image.ndim == 4:
            for c in range(image.shape[0]):
                image[c] = gaussian_filter(image[c], sigma=sigma)
        else:
            image = gaussian_filter(image, sigma=sigma)
        
        return image, label


class RandomBrightnessContrast:
    """Random brightness and contrast adjustment."""
    
    def __init__(self, brightness_range=(-0.2, 0.2), contrast_range=(0.8, 1.2), p=0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, image, label):
        if np.random.random() > self.p:
            return image, label
        
        brightness = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        image = image * contrast + brightness
        
        return image.astype(np.float32), label


class ElasticDeformation3D:
    """Elastic deformation for 3D volumes."""
    
    def __init__(self, alpha=100, sigma=10, p=0.5):
        """
        Args:
            alpha: deformation intensity
            sigma: smoothness of deformation
            p: probability of applying deformation
        """
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, image, label):
        if np.random.random() > self.p:
            return image, label
        
        shape = image.shape[-3:]
        
        # Generate random displacement fields
        dx = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        dy = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        dz = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        
        # Create coordinate grids
        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        
        indices = [
            np.clip(z + dz, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(x + dx, 0, shape[2] - 1)
        ]
        
        if image.ndim == 4:
            deformed_image = np.zeros_like(image)
            for c in range(image.shape[0]):
                deformed_image[c] = map_coordinates(image[c], indices, order=1, mode='constant')
        else:
            deformed_image = map_coordinates(image, indices, order=1, mode='constant')
        
        deformed_label = map_coordinates(label, indices, order=0, mode='constant')
        
        return deformed_image.astype(np.float32), deformed_label.astype(label.dtype)


def get_train_transforms(crop_size=96, normalize=True):
    """Get training transforms with augmentation."""
    transforms = []
    
    if normalize:
        transforms.append(Normalize())
    
    transforms.extend([
        RandomCrop3D(crop_size),
        RandomFlip3D(p=0.5),
        RandomRotation3D(angle_range=(-15, 15), p=0.3),
        RandomScale3D(scale_range=(0.9, 1.1), p=0.3),
        RandomGaussianNoise(std_range=(0, 0.05), p=0.2),
        RandomGaussianBlur(sigma_range=(0.5, 1.0), p=0.2),
        RandomBrightnessContrast(p=0.2),
        ToTensor(),
    ])
    
    return Compose(transforms)


def get_val_transforms(crop_size=None, normalize=True):
    """Get validation/test transforms (no augmentation)."""
    transforms = []
    
    if normalize:
        transforms.append(Normalize())
    
    if crop_size is not None:
        transforms.append(CenterCrop3D(crop_size))
    
    transforms.append(ToTensor())
    
    return Compose(transforms)
