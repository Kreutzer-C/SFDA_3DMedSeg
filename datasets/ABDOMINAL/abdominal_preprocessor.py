"""
Preprocessor for ABDOMINAL dataset.
Handles both CHAOST2 (MRI T2-SPIR) and BTCV (CT) data.

CHAOST2: MRI T2-SPIR images with DICOM format and PNG labels
BTCV: CT images with NIfTI format

Both datasets contain abdominal organ segmentation labels.

Preprocessing steps:
1. N4 bias field correction for MRI, window clipping [-125, 275] for CT
2. Reorient to RAS (Right-Anterior-Superior) coordinate system
3. Resample to target spacing (x,y: median, z: min), trilinear for images, nearest for labels
4. ROI cropping: crop non-body region, then crop foreground with 20% margin
5. Intensity normalization: 0.5%-99.5% percentile clipping, then case-wise min-max normalization

Test cases (fixed):
- MRI (CHAOST2): [1, 31, 32, 38]
- CT (BTCV): [1, 6, 30, 32, 33, 39] -> mapped to [0001, 0006, 0030, 0032, 0033, 0039]
"""

import os
import json
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict

import SimpleITK as sitk
import pydicom
import nibabel as nib
from PIL import Image

from scipy.ndimage import binary_fill_holes
from scipy import ndimage


# Fixed test cases
CHAOST2_TEST_CASES = ['1', '31', '32', '38']
BTCV_TEST_CASES = ['0001', '0006', '0030', '0032', '0033', '0039']

# Foreground classes (excluding background)
FOREGROUND_CLASSES = [1, 2, 3, 4]  # Liver, Right Kidney, Left Kidney, Spleen


def convert_to_native_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: object to convert (can be dict, list, numpy array, numpy scalar, etc.)
    
    Returns:
        object with all numpy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def n4_bias_field_correction(image_sitk, mask_sitk=None, shrink_factor=4, num_iterations=50):
    """
    Apply N4 bias field correction to MRI image.
    
    Args:
        image_sitk: SimpleITK image
        mask_sitk: optional mask for correction
        shrink_factor: downsampling factor for speed
        num_iterations: number of fitting iterations
    
    Returns:
        corrected SimpleITK image
    """
    # Cast to float32
    image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
    
    # Create mask if not provided
    if mask_sitk is None:
        # Use Otsu threshold to create mask
        mask_sitk = sitk.OtsuThreshold(image_sitk, 0, 1, 200)
    
    # Shrink for faster processing
    shrink_filter = sitk.ShrinkImageFilter()
    shrink_filter.SetShrinkFactors([shrink_factor] * 3)
    shrunk_image = shrink_filter.Execute(image_sitk)
    shrunk_mask = shrink_filter.Execute(mask_sitk)
    
    # N4 correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iterations] * 4)
    corrector.SetConvergenceThreshold(0.001)
    
    try:
        corrected_shrunk = corrector.Execute(shrunk_image, shrunk_mask)
        
        # Get the bias field
        log_bias_field = corrector.GetLogBiasFieldAsImage(image_sitk)
        bias_field = sitk.Exp(log_bias_field)
        
        # Apply correction to original resolution
        corrected_image = image_sitk / bias_field
        
        return corrected_image
    except Exception as e:
        print(f"  Warning: N4 correction failed ({e}), returning original image")
        return image_sitk


def reorient_to_ras(image_sitk, label_sitk=None):
    """
    Reorient image to RAS (Right-Anterior-Superior) coordinate system.
    
    Args:
        image_sitk: SimpleITK image
        label_sitk: optional SimpleITK label image
    
    Returns:
        reoriented image (and label if provided)
    """
    # Get current direction
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation('RAS')
    
    image_ras = orienter.Execute(image_sitk)
    
    if label_sitk is not None:
        label_ras = orienter.Execute(label_sitk)
        return image_ras, label_ras
    
    return image_ras, None


def resample_image(image_sitk, target_spacing, is_label=False):
    """
    Resample image to target spacing.
    
    Args:
        image_sitk: SimpleITK image
        target_spacing: target spacing (x, y, z)
        is_label: if True, use nearest neighbor interpolation
    
    Returns:
        resampled SimpleITK image
    """
    original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()
    
    # Calculate new size
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    
    # Set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(float(sitk.GetArrayFromImage(image_sitk).min()))
    
    return resampler.Execute(image_sitk)


def crop_to_body(image_np, label_np=None, threshold=None):
    """
    Crop image to body region (remove background).
    
    Args:
        image_np: 3D numpy array
        label_np: optional 3D label array
        threshold: threshold for body detection (auto if None)
    
    Returns:
        cropped_image, cropped_label (if provided), crop_bbox
    """
    if threshold is None:
        # Use Otsu-like threshold
        threshold = np.percentile(image_np, 10)
    
    # Create body mask
    body_mask = image_np > threshold
    
    # Fill holes and clean up
    for i in range(body_mask.shape[0]):
        body_mask[i] = binary_fill_holes(body_mask[i])
    
    # Find bounding box
    nonzero = np.where(body_mask)
    if len(nonzero[0]) == 0:
        # No body found, return original
        return image_np, label_np, None
    
    z_min, z_max = nonzero[0].min(), nonzero[0].max()
    y_min, y_max = nonzero[1].min(), nonzero[1].max()
    x_min, x_max = nonzero[2].min(), nonzero[2].max()
    
    bbox = [z_min, z_max + 1, y_min, y_max + 1, x_min, x_max + 1]
    
    cropped_image = image_np[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    
    if label_np is not None:
        cropped_label = label_np[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        return cropped_image, cropped_label, bbox
    
    return cropped_image, None, bbox


def crop_to_foreground_with_margin(image_np, label_np, foreground_classes, margin_ratio=0.2):
    """
    Crop to foreground region with specified margin.
    
    Args:
        image_np: 3D numpy array (D, H, W)
        label_np: 3D label array (D, H, W)
        foreground_classes: list of foreground class indices
        margin_ratio: margin ratio (0.2 = 20%)
    
    Returns:
        cropped_image, cropped_label, crop_bbox
    """
    # Create foreground mask
    fg_mask = np.zeros_like(label_np, dtype=bool)
    for cls in foreground_classes:
        fg_mask |= (label_np == cls)
    
    # Find bounding box of foreground
    nonzero = np.where(fg_mask)
    if len(nonzero[0]) == 0:
        # No foreground found, return original
        return image_np, label_np, None
    
    z_min, z_max = nonzero[0].min(), nonzero[0].max()
    y_min, y_max = nonzero[1].min(), nonzero[1].max()
    x_min, x_max = nonzero[2].min(), nonzero[2].max()
    
    # Calculate margin
    z_range = z_max - z_min
    y_range = y_max - y_min
    x_range = x_max - x_min
    
    z_margin = int(z_range * margin_ratio)
    y_margin = int(y_range * margin_ratio)
    x_margin = int(x_range * margin_ratio)
    
    # Apply margin
    z_min = max(0, z_min - z_margin)
    z_max = min(image_np.shape[0], z_max + z_margin + 1)
    y_min = max(0, y_min - y_margin)
    y_max = min(image_np.shape[1], y_max + y_margin + 1)
    x_min = max(0, x_min - x_margin)
    x_max = min(image_np.shape[2], x_max + x_margin + 1)
    
    bbox = [z_min, z_max, y_min, y_max, x_min, x_max]
    
    cropped_image = image_np[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped_label = label_np[z_min:z_max, y_min:y_max, x_min:x_max]
    
    return cropped_image, cropped_label, bbox


def intensity_normalization(image_np, lower_percentile=0.5, upper_percentile=99.5):
    """
    Apply intensity normalization:
    1. Clip to [lower_percentile, upper_percentile]
    2. Min-max normalization to [0, 1]
    
    Args:
        image_np: 3D numpy array
        lower_percentile: lower percentile for clipping
        upper_percentile: upper percentile for clipping
    
    Returns:
        normalized image
    """
    # Calculate percentile values
    p_low = np.percentile(image_np, lower_percentile)
    p_high = np.percentile(image_np, upper_percentile)
    
    # Clip
    image_np = np.clip(image_np, p_low, p_high)
    
    # Min-max normalization
    img_min = image_np.min()
    img_max = image_np.max()
    
    if img_max - img_min > 1e-8:
        image_np = (image_np - img_min) / (img_max - img_min)
    else:
        image_np = np.zeros_like(image_np)
    
    return image_np.astype(np.float32)


class CHAOST2Preprocessor:
    """
    Preprocessor for CHAOS T2-SPIR MRI dataset.
    
    Data structure:
    - raw/CHAOST2/{case_id}/T2SPIR/DICOM_anon/*.dcm (images)
    - raw/CHAOST2/{case_id}/T2SPIR/Ground/*.png (labels)
    
    Label mapping (from PNG values):
    - 0: Background
    - 63: Liver
    - 126: Right Kidney  
    - 189: Left Kidney
    - 252: Spleen
    """
    
    LABEL_MAPPING = {
        0: 0,      # Background
        63: 1,     # Liver
        126: 2,    # Right Kidney
        189: 3,    # Left Kidney
        252: 4,    # Spleen
    }
    
    CLASS_NAMES = ['Background', 'Liver', 'Right Kidney', 'Left Kidney', 'Spleen']
    
    def __init__(self, raw_dir):
        """
        Args:
            raw_dir: path to raw CHAOST2 data
        """
        self.raw_dir = raw_dir
    
    def get_case_ids(self):
        """Get list of case IDs."""
        cases = []
        for d in os.listdir(self.raw_dir):
            case_path = os.path.join(self.raw_dir, d, 'T2SPIR')
            if os.path.isdir(case_path):
                cases.append(d)
        return sorted(cases, key=lambda x: int(x))
    
    def load_case_sitk(self, case_id):
        """
        Load a single case as SimpleITK image.
        
        Returns:
            image_sitk: SimpleITK image
            label_sitk: SimpleITK label image
        """
        case_dir = os.path.join(self.raw_dir, case_id, 'T2SPIR')
        dicom_dir = os.path.join(case_dir, 'DICOM_anon')
        label_dir = os.path.join(case_dir, 'Ground')
        
        # Load DICOM series using SimpleITK
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found for case {case_id}")
        
        reader.SetFileNames(dicom_files)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image_sitk = reader.Execute()
        
        # Load PNG labels
        label_files = sorted(glob(os.path.join(label_dir, '*.png')))
        
        # Read labels and stack
        labels = []
        for label_file in label_files:
            lbl = np.array(Image.open(label_file))
            labels.append(lbl)
        
        # Need to sort labels same as DICOM slices
        # Read DICOM files manually to get slice positions
        dicom_slices = []
        for dcm_file in dicom_files:
            dcm = pydicom.dcmread(dcm_file)
            if hasattr(dcm, 'SliceLocation'):
                pos = float(dcm.SliceLocation)
            elif hasattr(dcm, 'ImagePositionPatient'):
                pos = float(dcm.ImagePositionPatient[2])
            else:
                pos = float(dcm.InstanceNumber)
            dicom_slices.append((dcm_file, pos))
        
        # Sort by position
        dicom_slices_sorted = sorted(dicom_slices, key=lambda x: x[1])
        original_dicom_files = [d[0] for d in dicom_slices]
        sorted_indices = [original_dicom_files.index(d[0]) for d in dicom_slices_sorted]
        
        # Reorder labels to match DICOM order
        if len(labels) == len(sorted_indices):
            labels = [labels[i] for i in sorted_indices]
        
        label_np = np.stack(labels, axis=0)  # (D, H, W)
        
        # Map label values to class indices
        mapped_label = np.zeros_like(label_np, dtype=np.uint8)
        for orig_val, new_val in self.LABEL_MAPPING.items():
            mapped_label[label_np == orig_val] = new_val
        
        # Convert label to SimpleITK image
        # Note: SimpleITK uses (x, y, z) ordering, numpy uses (z, y, x)
        label_sitk = sitk.GetImageFromArray(mapped_label)
        label_sitk.SetSpacing(image_sitk.GetSpacing())
        label_sitk.SetOrigin(image_sitk.GetOrigin())
        label_sitk.SetDirection(image_sitk.GetDirection())
        
        return image_sitk, label_sitk


class BTCVPreprocessor:
    """
    Preprocessor for BTCV CT dataset.
    
    Data structure:
    - raw/BTCV/img/imgXXXX.nii.gz (images)
    - raw/BTCV/label/labelXXXX.nii.gz (labels)
    
    Note: BTCV has 13 organ classes, but for cross-domain experiments with CHAOST2,
    we use only the 4 overlapping organs (liver, kidneys, spleen).
    """
    
    # Mapping to match CHAOST2 classes (4 organs)
    CHAOST2_COMPATIBLE_MAPPING = {
        0: 0,   # Background
        6: 1,   # Liver -> 1
        2: 2,   # Right Kidney -> 2
        3: 3,   # Left Kidney -> 3
        1: 4,   # Spleen -> 4
    }
    
    CLASS_NAMES = ['Background', 'Liver', 'Right Kidney', 'Left Kidney', 'Spleen']
    
    # CT window
    WINDOW_MIN = -125
    WINDOW_MAX = 275
    
    def __init__(self, raw_dir):
        """
        Args:
            raw_dir: path to raw BTCV data
        """
        self.raw_dir = raw_dir
    
    def get_case_ids(self):
        """Get list of case IDs."""
        img_dir = os.path.join(self.raw_dir, 'img')
        cases = []
        for f in os.listdir(img_dir):
            if f.endswith('.nii.gz'):
                case_id = f.replace('img', '').replace('.nii.gz', '')
                cases.append(case_id)
        return sorted(cases)
    
    def load_case_sitk(self, case_id):
        """
        Load a single case as SimpleITK image.
        
        Returns:
            image_sitk: SimpleITK image (with CT windowing applied)
            label_sitk: SimpleITK label image
        """
        img_path = os.path.join(self.raw_dir, 'img', f'img{case_id}.nii.gz')
        label_path = os.path.join(self.raw_dir, 'label', f'label{case_id}.nii.gz')
        
        # Load using SimpleITK
        image_sitk = sitk.ReadImage(img_path)
        label_sitk = sitk.ReadImage(label_path)
        
        # Apply CT window clipping
        image_sitk = sitk.Clamp(image_sitk, sitk.sitkFloat32, 
                                self.WINDOW_MIN, self.WINDOW_MAX)
        
        # Map labels to CHAOST2 compatible classes
        label_np = sitk.GetArrayFromImage(label_sitk)
        mapped_label = np.zeros_like(label_np, dtype=np.uint8)
        for orig_val, new_val in self.CHAOST2_COMPATIBLE_MAPPING.items():
            mapped_label[label_np == orig_val] = new_val
        
        label_sitk_mapped = sitk.GetImageFromArray(mapped_label)
        label_sitk_mapped.CopyInformation(label_sitk)
        
        return image_sitk, label_sitk_mapped


def compute_target_spacing(chaost2_preprocessor, btcv_preprocessor):
    """
    Compute target spacing from all cases in both domains.
    
    Strategy:
    - x, y axes: use median spacing
    - z axis: use minimum spacing (highest resolution)
    
    Returns:
        target_spacing: tuple (x, y, z) in mm
        all_spacings: dict with all spacings for metadata
    """
    all_spacings = {'CHAOST2': [], 'BTCV': []}
    
    print("Computing target spacing from all cases...")
    
    # Get CHAOST2 spacings
    if chaost2_preprocessor is not None:
        for case_id in tqdm(chaost2_preprocessor.get_case_ids(), desc='Reading CHAOST2 spacings'):
            try:
                image_sitk, _ = chaost2_preprocessor.load_case_sitk(case_id)
                # Reorient to RAS first to get consistent spacing direction
                image_ras, _ = reorient_to_ras(image_sitk)
                spacing = image_ras.GetSpacing()  # (x, y, z)
                all_spacings['CHAOST2'].append(spacing)
            except Exception as e:
                print(f"  Warning: Could not read spacing for CHAOST2 case {case_id}: {e}")
    
    # Get BTCV spacings
    if btcv_preprocessor is not None:
        for case_id in tqdm(btcv_preprocessor.get_case_ids(), desc='Reading BTCV spacings'):
            try:
                image_sitk, _ = btcv_preprocessor.load_case_sitk(case_id)
                # Reorient to RAS first to get consistent spacing direction
                image_ras, _ = reorient_to_ras(image_sitk)
                spacing = image_ras.GetSpacing()  # (x, y, z)
                all_spacings['BTCV'].append(spacing)
            except Exception as e:
                print(f"  Warning: Could not read spacing for BTCV case {case_id}: {e}")
    
    # Combine all spacings
    combined_spacings = all_spacings['CHAOST2'] + all_spacings['BTCV']
    
    if not combined_spacings:
        raise ValueError("No valid spacings found!")
    
    # Compute target spacing:
    # - x, y: median
    # - z: minimum (highest resolution)
    spacings_array = np.array(combined_spacings)
    median_spacing_xy = np.median(spacings_array[:, :2], axis=0)  # x, y
    min_spacing_z = np.min(spacings_array[:, 2])  # z - use minimum for highest resolution
    
    target_spacing = (float(median_spacing_xy[0]), float(median_spacing_xy[1]), float(min_spacing_z))
    
    print(f"  CHAOST2 spacings range: {np.min(all_spacings['CHAOST2'], axis=0) if all_spacings['CHAOST2'] else 'N/A'} - {np.max(all_spacings['CHAOST2'], axis=0) if all_spacings['CHAOST2'] else 'N/A'}")
    print(f"  BTCV spacings range: {np.min(all_spacings['BTCV'], axis=0) if all_spacings['BTCV'] else 'N/A'} - {np.max(all_spacings['BTCV'], axis=0) if all_spacings['BTCV'] else 'N/A'}")
    print(f"  Target spacing (x, y: median, z: min): {target_spacing}")
    
    return target_spacing, all_spacings


def preprocess_single_case(image_sitk, label_sitk, target_spacing, is_mri=False, 
                           ct_window=None, case_id=""):
    """
    Preprocess a single case following all steps.
    
    Args:
        image_sitk: SimpleITK image
        label_sitk: SimpleITK label image
        target_spacing: target spacing (x, y, z)
        is_mri: if True, apply N4 bias field correction
        ct_window: tuple (min, max) for CT windowing (already applied in loader)
        case_id: case identifier for logging
    
    Returns:
        image_np: preprocessed image numpy array (D, H, W)
        label_np: preprocessed label numpy array (D, H, W)
        metadata: dict with preprocessing metadata
    """
    metadata = {}
    
    # Record original size and spacing
    original_size = image_sitk.GetSize()  # (x, y, z)
    original_spacing = image_sitk.GetSpacing()  # (x, y, z)
    metadata['origin_size'] = list(original_size)
    metadata['origin_spacing'] = list(original_spacing)
    
    # Step 1: N4 bias field correction for MRI (CT windowing already done in loader)
    if is_mri:
        print(f"    Applying N4 bias field correction...")
        image_sitk = n4_bias_field_correction(image_sitk)
    
    # Step 2: Reorient to RAS
    print(f"    Reorienting to RAS...")
    image_sitk, label_sitk = reorient_to_ras(image_sitk, label_sitk)
    
    # Update spacing after reorientation (direction may change)
    ras_spacing = image_sitk.GetSpacing()
    ras_size = image_sitk.GetSize()
    metadata['ras_size'] = list(ras_size)
    metadata['ras_spacing'] = list(ras_spacing)
    
    # Step 3: Resample to target spacing
    print(f"    Resampling to spacing {target_spacing}...")
    image_sitk = resample_image(image_sitk, target_spacing, is_label=False)
    label_sitk = resample_image(label_sitk, target_spacing, is_label=True)
    
    new_size = image_sitk.GetSize()  # (x, y, z)
    new_spacing = image_sitk.GetSpacing()  # (x, y, z)
    metadata['new_size'] = list(new_size)
    metadata['new_spacing'] = list(new_spacing)
    
    # Convert to numpy for further processing
    # SimpleITK array is (z, y, x), we keep it as (D, H, W) = (z, y, x)
    image_np = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    label_np = sitk.GetArrayFromImage(label_sitk).astype(np.int64)
    
    # Step 4a: Crop to body region
    print(f"    Cropping to body region...")
    if is_mri:
        body_threshold = None  # Auto
    else:
        # For CT, use a threshold based on the window
        body_threshold = -100  # After windowing, this is roughly air/tissue boundary
    
    image_np, label_np, body_bbox = crop_to_body(image_np, label_np, threshold=body_threshold)
    metadata['body_crop_bbox'] = body_bbox
    
    # Step 4b: Crop to foreground with 20% margin
    print(f"    Cropping to foreground with 20% margin...")
    image_np, label_np, fg_bbox = crop_to_foreground_with_margin(
        image_np, label_np, FOREGROUND_CLASSES, margin_ratio=0.2
    )
    metadata['foreground_crop_bbox'] = fg_bbox
    metadata['crop_size'] = list(image_np.shape)  # Final size (D, H, W)
    
    # Step 5: Intensity normalization (0.5%-99.5% percentile + min-max)
    print(f"    Applying intensity normalization...")
    image_np = intensity_normalization(image_np, lower_percentile=0.5, upper_percentile=99.5)
    
    return image_np, label_np, metadata


def preprocess_abdominal_dataset(raw_base_dir, output_dir):
    """
    Preprocess both CHAOST2 and BTCV datasets.
    
    Args:
        raw_base_dir: base directory containing raw/CHAOST2 and raw/BTCV
        output_dir: directory to save preprocessed data
    
    Returns:
        metadata: dictionary containing dataset information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize preprocessors
    chaost2_raw = os.path.join(raw_base_dir, 'CHAOST2')
    btcv_raw = os.path.join(raw_base_dir, 'BTCV')
    
    chaost2_preprocessor = None
    btcv_preprocessor = None
    
    if os.path.exists(chaost2_raw):
        chaost2_preprocessor = CHAOST2Preprocessor(chaost2_raw)
    
    if os.path.exists(btcv_raw):
        btcv_preprocessor = BTCVPreprocessor(btcv_raw)
    
    if chaost2_preprocessor is None and btcv_preprocessor is None:
        raise ValueError("No valid datasets found!")
    
    # Step: Compute median spacing from all cases
    print("\n" + "=" * 60)
    print("Step 1: Computing median spacing from all cases...")
    print("=" * 60)
    
    target_spacing, all_spacings = compute_target_spacing(chaost2_preprocessor, btcv_preprocessor)
    
    all_splits = {'train': [], 'test': []}
    case_metadata = {}
    
    # Process CHAOST2
    if chaost2_preprocessor is not None:
        print("\n" + "=" * 60)
        print("Processing CHAOST2 dataset (MRI T2-SPIR)...")
        print("=" * 60)
        
        case_ids = chaost2_preprocessor.get_case_ids()
        print(f"Found {len(case_ids)} CHAOST2 cases")
        
        # Split into train/test based on fixed test cases
        train_ids = [c for c in case_ids if c not in CHAOST2_TEST_CASES]
        test_ids = [c for c in case_ids if c in CHAOST2_TEST_CASES]
        
        print(f"Train cases: {len(train_ids)}, Test cases: {len(test_ids)}")
        print(f"Test case IDs: {test_ids}")
        
        for case_id in tqdm(case_ids, desc='Processing CHAOST2'):
            try:
                full_case_id = f'CHAOST2_{case_id}'
                print(f"\n  Processing {full_case_id}...")
                
                # Load case
                image_sitk, label_sitk = chaost2_preprocessor.load_case_sitk(case_id)
                
                # Preprocess
                image_np, label_np, meta = preprocess_single_case(
                    image_sitk, label_sitk, 
                    target_spacing=target_spacing,
                    is_mri=True,
                    case_id=full_case_id
                )
                
                # Verify spatial consistency
                assert image_np.shape == label_np.shape, f"Shape mismatch: image {image_np.shape} vs label {label_np.shape}"
                
                # Save
                output_path = os.path.join(output_dir, f'{full_case_id}.npz')
                np.savez_compressed(
                    output_path,
                    image=image_np,
                    label=label_np,
                    spacing=np.array(target_spacing),
                    case_id=full_case_id,
                )
                
                # Update metadata
                meta['case_id'] = full_case_id
                meta['domain'] = 'CHAOST2'
                meta['modality'] = 'MRI'
                meta['final_shape'] = list(image_np.shape)
                meta['unique_labels'] = np.unique(label_np).tolist()
                case_metadata[full_case_id] = meta
                
                print(f"    Final shape: {image_np.shape}, labels: {np.unique(label_np)}")
                
                # Add to splits
                if case_id in CHAOST2_TEST_CASES:
                    all_splits['test'].append(full_case_id)
                else:
                    all_splits['train'].append(full_case_id)
                
            except Exception as e:
                print(f"Error processing case {case_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # Process BTCV
    if btcv_preprocessor is not None:
        print("\n" + "=" * 60)
        print("Processing BTCV dataset (CT)...")
        print("=" * 60)
        
        case_ids = btcv_preprocessor.get_case_ids()
        print(f"Found {len(case_ids)} BTCV cases")
        
        # Split into train/test based on fixed test cases
        train_ids = [c for c in case_ids if c not in BTCV_TEST_CASES]
        test_ids = [c for c in case_ids if c in BTCV_TEST_CASES]
        
        print(f"Train cases: {len(train_ids)}, Test cases: {len(test_ids)}")
        print(f"Test case IDs: {test_ids}")
        
        for case_id in tqdm(case_ids, desc='Processing BTCV'):
            try:
                full_case_id = f'BTCV_{case_id}'
                print(f"\n  Processing {full_case_id}...")
                
                # Load case
                image_sitk, label_sitk = btcv_preprocessor.load_case_sitk(case_id)
                
                # Preprocess
                image_np, label_np, meta = preprocess_single_case(
                    image_sitk, label_sitk,
                    target_spacing=target_spacing,
                    is_mri=False,
                    ct_window=(BTCVPreprocessor.WINDOW_MIN, BTCVPreprocessor.WINDOW_MAX),
                    case_id=full_case_id
                )
                
                # Verify spatial consistency
                assert image_np.shape == label_np.shape, f"Shape mismatch: image {image_np.shape} vs label {label_np.shape}"
                
                # Save
                output_path = os.path.join(output_dir, f'{full_case_id}.npz')
                np.savez_compressed(
                    output_path,
                    image=image_np,
                    label=label_np,
                    spacing=np.array(target_spacing),
                    case_id=full_case_id,
                )
                
                # Update metadata
                meta['case_id'] = full_case_id
                meta['domain'] = 'BTCV'
                meta['modality'] = 'CT'
                meta['final_shape'] = list(image_np.shape)
                meta['unique_labels'] = np.unique(label_np).tolist()
                case_metadata[full_case_id] = meta
                
                print(f"    Final shape: {image_np.shape}, labels: {np.unique(label_np)}")
                
                # Add to splits
                if case_id in BTCV_TEST_CASES:
                    all_splits['test'].append(full_case_id)
                else:
                    all_splits['train'].append(full_case_id)
                
            except Exception as e:
                print(f"Error processing case {case_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # Create dataset metadata
    metadata = {
        'name': 'ABDOMINAL',
        'description': 'Abdominal organ segmentation dataset (CHAOST2 + BTCV)',
        'domains': ['CHAOST2', 'BTCV'],
        'modalities': {
            'CHAOST2': 'MRI T2-SPIR',
            'BTCV': 'CT',
        },
        'num_classes': 5,
        'class_names': ['Background', 'Liver', 'Right Kidney', 'Left Kidney', 'Spleen'],
        'target_spacing': list(target_spacing),
        'preprocessing': {
            'steps': [
                '1. N4 bias field correction (MRI) / CT window clipping [-125, 275] (CT)',
                '2. Reorient to RAS coordinate system',
                '3. Resample to target spacing (x,y: median, z: min for highest resolution)',
                '4. ROI cropping: body region + foreground with 20% margin',
                '5. Intensity normalization: 0.5%-99.5% percentile clipping + min-max',
            ],
            'ct_window': [-125, 275],
            'mri_n4_correction': True,
            'orientation': 'RAS',
            'foreground_margin_ratio': 0.2,
            'percentile_clip': [0.5, 99.5],
            'normalization': 'min-max to [0, 1]',
        },
        'spacing_statistics': {
            'chaost2_spacings': [list(s) for s in all_spacings.get('CHAOST2', [])],
            'btcv_spacings': [list(s) for s in all_spacings.get('BTCV', [])],
            'target_spacing': list(target_spacing),
            'spacing_strategy': 'x,y: median; z: minimum (highest resolution)',
        },
        'splits': all_splits,
        'test_cases': {
            'CHAOST2': CHAOST2_TEST_CASES,
            'BTCV': BTCV_TEST_CASES,
        },
        'case_metadata': case_metadata,
    }
    
    # Save metadata (convert numpy types to native Python types)
    metadata = convert_to_native_types(metadata)
    metadata_path = os.path.join(os.path.dirname(output_dir), 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("Preprocessing completed!")
    print(f"{'=' * 60}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Total cases: train={len(all_splits['train'])}, test={len(all_splits['test'])}")
    print(f"Target spacing (x, y, z): {target_spacing} mm")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Preprocess ABDOMINAL dataset')
    parser.add_argument('--raw_dir', type=str, default=None,
                        help='Path to raw data directory (default: ./raw)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to output directory (default: ./preprocessed)')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set default paths relative to script location
    if args.raw_dir is None:
        args.raw_dir = os.path.join(script_dir, 'raw')
    if args.output_dir is None:
        args.output_dir = os.path.join(script_dir, 'preprocessed')
    
    preprocess_abdominal_dataset(
        raw_base_dir=args.raw_dir,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
