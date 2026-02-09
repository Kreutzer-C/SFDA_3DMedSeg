"""
Convert preprocessed cases to 3D Slicer compatible format (NIfTI).

This is a generic script that reads dataset information from metadata.json
and can be used for any dataset following the same structure.

Usage:
    # Convert a single case
    python visualize_case.py --case CHAOST2_1
    
    # Convert multiple cases
    python visualize_case.py --case CHAOST2_1 BTCV_0001
    
    # Convert all cases from a domain
    python visualize_case.py --domain CHAOST2
    
    # Convert all test cases
    python visualize_case.py --split test
    
    # Convert all cases
    python visualize_case.py --all
    
    # Specify output directory
    python visualize_case.py --case CHAOST2_1 --output_dir ./visualization

Output:
    For each case, generates:
    - {case_id}_image.nii.gz: Image volume
    - {case_id}_label.nii.gz: Segmentation label
    
    These files can be directly opened in 3D Slicer for visualization.
"""

import os
import argparse
import json
import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required. Install with: pip install nibabel")


def load_metadata(metadata_path):
    """
    Load dataset metadata from JSON file.
    
    Args:
        metadata_path: path to metadata.json
    
    Returns:
        dict: metadata dictionary
    """
    if not os.path.exists(metadata_path):
        return None
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def load_preprocessed_case(npz_path):
    """
    Load a preprocessed case from .npz file.
    
    Args:
        npz_path: path to .npz file
    
    Returns:
        image: 3D numpy array
        label: 3D numpy array
        spacing: voxel spacing (z, y, x)
    """
    data = np.load(npz_path)
    image = data['image']
    label = data['label']
    spacing = data['spacing'] if 'spacing' in data else np.array([1.0, 1.0, 1.0])
    
    return image, label, spacing


def save_as_nifti(volume, spacing, output_path):
    """
    Save a 3D volume as NIfTI file.
    
    Args:
        volume: 3D numpy array (D, H, W)
        spacing: voxel spacing (z, y, x)
        output_path: output file path
    """
    # Transpose from (D, H, W) to (W, H, D) for NIfTI format
    volume_nifti = np.transpose(volume, (2, 1, 0))
    
    # Create affine matrix with spacing
    # spacing is (z, y, x), need to reverse for NIfTI (x, y, z)
    spacing_xyz = spacing[::-1]
    affine = np.diag([spacing_xyz[0], spacing_xyz[1], spacing_xyz[2], 1.0])
    
    # Create NIfTI image
    nii = nib.Nifti1Image(volume_nifti, affine)
    
    # Save
    nib.save(nii, output_path)


def convert_case(case_id, preprocessed_dir, output_dir, metadata=None):
    """
    Convert a single preprocessed case to NIfTI format.
    
    Args:
        case_id: case identifier (e.g., 'CHAOST2_1', 'BTCV_0001')
        preprocessed_dir: directory containing preprocessed .npz files
        output_dir: directory to save NIfTI files
        metadata: optional metadata dict for additional info
    
    Returns:
        bool: True if successful, False otherwise
    """
    npz_path = os.path.join(preprocessed_dir, f"{case_id}.npz")
    
    if not os.path.exists(npz_path):
        print(f"Warning: Case {case_id} not found at {npz_path}")
        return False
    
    # Load data
    image, label, spacing = load_preprocessed_case(npz_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save image
    image_path = os.path.join(output_dir, f"{case_id}_image.nii.gz")
    save_as_nifti(image.astype(np.float32), spacing, image_path)
    
    # Save label
    label_path = os.path.join(output_dir, f"{case_id}_label.nii.gz")
    save_as_nifti(label.astype(np.int16), spacing, label_path)
    
    # Get domain info from metadata
    domain_info = ""
    if metadata:
        for domain in metadata.get('domains', []):
            if case_id.startswith(domain):
                modality = metadata.get('modalities', {}).get(domain, '')
                if modality:
                    domain_info = f" ({modality})"
                break
    
    print(f"Converted {case_id}{domain_info}:")
    print(f"  Image: {image_path} (shape: {image.shape}, dtype: {image.dtype})")
    print(f"  Label: {label_path} (shape: {label.shape}, unique: {np.unique(label)})")
    print(f"  Spacing: {spacing} mm")
    
    return True


def get_all_cases(preprocessed_dir, domain=None, metadata=None):
    """
    Get list of all available cases.
    
    Args:
        preprocessed_dir: directory containing preprocessed .npz files
        domain: optional domain filter (e.g., 'CHAOST2', 'BTCV')
        metadata: optional metadata dict
    
    Returns:
        list of case IDs
    """
    cases = []
    
    if os.path.exists(preprocessed_dir):
        for f in os.listdir(preprocessed_dir):
            if f.endswith('.npz'):
                case_id = f.replace('.npz', '')
                if domain is None or case_id.startswith(domain):
                    cases.append(case_id)
    
    return sorted(cases)


def get_cases_by_split(metadata, split):
    """
    Get cases from a specific split (train/test).
    
    Args:
        metadata: metadata dict
        split: 'train' or 'test'
    
    Returns:
        list of case IDs
    """
    if metadata and 'splits' in metadata:
        return metadata['splits'].get(split, [])
    return []


def print_dataset_info(metadata, preprocessed_dir):
    """
    Print dataset information from metadata.
    
    Args:
        metadata: metadata dict
        preprocessed_dir: directory containing preprocessed files
    """
    if metadata:
        print(f"\nDataset: {metadata.get('name', 'Unknown')}")
        print(f"Description: {metadata.get('description', 'N/A')}")
        
        # Print domains
        domains = metadata.get('domains', [])
        modalities = metadata.get('modalities', {})
        if domains:
            print(f"\nDomains:")
            for domain in domains:
                modality = modalities.get(domain, '')
                print(f"  - {domain}" + (f" ({modality})" if modality else ""))
        
        # Print class names
        class_names = metadata.get('class_names', [])
        if class_names:
            print(f"\nLabel mapping:")
            for i, name in enumerate(class_names):
                print(f"  {i}: {name}")
        
        # Print splits
        splits = metadata.get('splits', {})
        if splits:
            print(f"\nSplits:")
            for split_name, cases in splits.items():
                print(f"  {split_name}: {len(cases)} cases")
    
    # Print available cases
    cases = get_all_cases(preprocessed_dir)
    if cases:
        print(f"\nAvailable preprocessed cases: {len(cases)}")
        
        # Group by domain
        if metadata:
            for domain in metadata.get('domains', []):
                domain_cases = [c for c in cases if c.startswith(domain)]
                if domain_cases:
                    print(f"\n  {domain}: {len(domain_cases)} cases")
                    for c in domain_cases[:5]:  # Show first 5
                        print(f"    - {c}")
                    if len(domain_cases) > 5:
                        print(f"    ... and {len(domain_cases) - 5} more")
        else:
            for c in cases[:10]:
                print(f"  - {c}")
            if len(cases) > 10:
                print(f"  ... and {len(cases) - 10} more")
    else:
        print("\nNo preprocessed cases found.")


def main():
    parser = argparse.ArgumentParser(
        description='Convert preprocessed cases to 3D Slicer compatible NIfTI format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List dataset info and available cases
    python visualize_case.py --info
    
    # Convert a single case
    python visualize_case.py --case CHAOST2_1
    
    # Convert multiple cases
    python visualize_case.py --case CHAOST2_1 BTCV_0001
    
    # Convert all cases from a specific domain
    python visualize_case.py --domain CHAOST2
    
    # Convert all test cases
    python visualize_case.py --split test
    
    # Convert all cases
    python visualize_case.py --all

In 3D Slicer:
    1. Open 3D Slicer
    2. File -> Add Data -> Choose the .nii.gz files
    3. For label visualization:
       - Go to "Segment Editor" module
       - Or use "Volume Rendering" for 3D view
        """
    )
    
    parser.add_argument('--case', type=str, nargs='+', default=None,
                        help='Case ID(s) to convert')
    parser.add_argument('--domain', type=str, default=None,
                        help='Convert all cases from a specific domain')
    parser.add_argument('--split', type=str, default=None,
                        choices=['train', 'test'],
                        help='Convert all cases from a specific split')
    parser.add_argument('--all', action='store_true',
                        help='Convert all available cases')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                        help='Path to preprocessed data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for NIfTI files')
    parser.add_argument('--info', action='store_true',
                        help='Show dataset info and available cases')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set default paths
    if args.preprocessed_dir is None:
        args.preprocessed_dir = os.path.join(script_dir, 'preprocessed')
    if args.output_dir is None:
        args.output_dir = os.path.join(script_dir, 'visualization')
    
    # Load metadata
    metadata_path = os.path.join(script_dir, 'metadata.json')
    metadata = load_metadata(metadata_path)
    
    # Info mode
    if args.info:
        print_dataset_info(metadata, args.preprocessed_dir)
        return
    
    # Check if preprocessed directory exists
    if not os.path.exists(args.preprocessed_dir):
        print(f"Error: Preprocessed directory not found: {args.preprocessed_dir}")
        print("Please run the preprocessor first.")
        return
    
    # Determine which cases to convert
    cases_to_convert = []
    
    if args.case:
        cases_to_convert = args.case
    elif args.domain:
        cases_to_convert = get_all_cases(args.preprocessed_dir, domain=args.domain, metadata=metadata)
    elif args.split:
        cases_to_convert = get_cases_by_split(metadata, args.split)
        # Filter to only existing cases
        existing_cases = set(get_all_cases(args.preprocessed_dir))
        cases_to_convert = [c for c in cases_to_convert if c in existing_cases]
    elif args.all:
        cases_to_convert = get_all_cases(args.preprocessed_dir, metadata=metadata)
    else:
        parser.print_help()
        print("\nError: Please specify --case, --domain, --split, --all, or --info")
        return
    
    if not cases_to_convert:
        print("No cases to convert.")
        return
    
    # Print header with dataset info
    if metadata:
        print(f"Dataset: {metadata.get('name', 'Unknown')}")
        class_names = metadata.get('class_names', [])
        if class_names:
            print(f"Labels: {', '.join([f'{i}={n}' for i, n in enumerate(class_names)])}")
    
    print(f"\nConverting {len(cases_to_convert)} case(s) to NIfTI format...")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    success_count = 0
    for case_id in cases_to_convert:
        if convert_case(case_id, args.preprocessed_dir, args.output_dir, metadata):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"Conversion complete: {success_count}/{len(cases_to_convert)} cases")
    print(f"\nTo visualize in 3D Slicer:")
    print(f"  1. Open 3D Slicer")
    print(f"  2. File -> Add Data")
    print(f"  3. Select the *_image.nii.gz and *_label.nii.gz files from:")
    print(f"     {args.output_dir}")


if __name__ == '__main__':
    main()
