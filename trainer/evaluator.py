"""
Evaluator for model inference and metric computation.
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm

from utils.metrics import compute_metrics, MetricTracker
from dataloader.utils import sliding_window_inference


class Evaluator:
    """
    Evaluator class for inference and metric computation.
    Handles sliding window inference for full volume prediction.
    """
    
    def __init__(
        self,
        model,
        device='cuda',
        num_classes=5,
        patch_size=(96, 96, 96),
        stride=(48, 48, 48),
        save_predictions=False,
        output_dir=None,
    ):
        """
        Args:
            model: trained segmentation model
            device: device to use
            num_classes: number of segmentation classes
            patch_size: patch size for sliding window inference
            stride: stride for sliding window inference
            save_predictions: whether to save predictions
            output_dir: directory to save predictions
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.stride = stride
        self.save_predictions = save_predictions
        self.output_dir = output_dir
        
        if save_predictions and output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Checkpoint loaded successfully")
    
    def evaluate(self, dataloader, class_names=None):
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: data loader for evaluation
            class_names: optional list of class names
        
        Returns:
            dict: evaluation results
        """
        self.model.eval()
        
        all_results = []
        metric_tracker = MetricTracker(['dice', 'assd', 'hd95'])
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                image = batch['image']
                label = batch['label']
                case_id = batch['case_id'][0] if isinstance(batch['case_id'], list) else batch['case_id']
                spacing = batch.get('spacing', np.array([1.0, 1.0, 1.0]))
                if isinstance(spacing, torch.Tensor):
                    spacing = spacing.numpy()
                if spacing.ndim > 1:
                    spacing = spacing[0]
                
                # Sliding window inference
                pred = sliding_window_inference(
                    image=image,
                    patch_size=self.patch_size,
                    stride=self.stride,
                    model=self.model,
                    device=self.device,
                    num_classes=self.num_classes,
                )
                
                # Convert label to numpy
                if isinstance(label, torch.Tensor):
                    label = label.squeeze().cpu().numpy()
                
                # Compute metrics
                case_metrics = compute_metrics(
                    pred=pred,
                    target=label,
                    num_classes=self.num_classes,
                    spacing=tuple(spacing),
                    include_background=False,
                )
                
                # Store results
                case_result = {
                    'case_id': case_id,
                    'metrics': case_metrics,
                }
                all_results.append(case_result)
                
                # Update tracker
                for c, metrics in case_metrics.items():
                    metric_tracker.update(metrics, class_idx=c)
                
                # Save prediction if requested
                if self.save_predictions and self.output_dir:
                    pred_path = os.path.join(self.output_dir, f'{case_id}_pred.npy')
                    np.save(pred_path, pred)
        
        # Compute summary statistics
        summary = self._compute_summary(all_results, class_names)
        
        return {
            'per_case_results': all_results,
            'summary': summary,
            'class_averages': metric_tracker.get_all_class_averages(),
        }
    
    def _compute_summary(self, results, class_names=None):
        """Compute summary statistics from results."""
        summary = {}
        
        # Collect metrics per class
        class_metrics = {}
        for result in results:
            for c, metrics in result['metrics'].items():
                if c not in class_metrics:
                    class_metrics[c] = {'dice': [], 'assd': [], 'hd95': []}
                for metric_name, value in metrics.items():
                    if not np.isinf(value):
                        class_metrics[c][metric_name].append(value)
        
        # Compute per-class statistics
        for c, metrics in class_metrics.items():
            class_name = class_names[c] if class_names and c < len(class_names) else f'Class_{c}'
            summary[class_name] = {}
            
            for metric_name, values in metrics.items():
                if values:
                    summary[class_name][f'{metric_name}_mean'] = np.mean(values)
                    summary[class_name][f'{metric_name}_std'] = np.std(values)
                else:
                    summary[class_name][f'{metric_name}_mean'] = 0.0
                    summary[class_name][f'{metric_name}_std'] = 0.0
        
        # Compute overall statistics
        all_dice = []
        all_assd = []
        all_hd95 = []
        
        for c, metrics in class_metrics.items():
            all_dice.extend(metrics['dice'])
            all_assd.extend([v for v in metrics['assd'] if not np.isinf(v)])
            all_hd95.extend([v for v in metrics['hd95'] if not np.isinf(v)])
        
        summary['Overall'] = {
            'dice_mean': np.mean(all_dice) if all_dice else 0.0,
            'dice_std': np.std(all_dice) if all_dice else 0.0,
            'assd_mean': np.mean(all_assd) if all_assd else 0.0,
            'assd_std': np.std(all_assd) if all_assd else 0.0,
            'hd95_mean': np.mean(all_hd95) if all_hd95 else 0.0,
            'hd95_std': np.std(all_hd95) if all_hd95 else 0.0,
        }
        
        return summary
    
    def save_results(self, results, output_path):
        """Save evaluation results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def print_summary(self, results):
        """Print evaluation summary."""
        summary = results['summary']
        
        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        
        for class_name, metrics in summary.items():
            print(f"\n{class_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        print("\n" + "=" * 60)


def evaluate_model(
    model,
    dataloader,
    checkpoint_path,
    num_classes=5,
    patch_size=(96, 96, 96),
    stride=(48, 48, 48),
    device='cuda',
    save_predictions=False,
    output_dir=None,
    class_names=None,
):
    """
    Convenience function to evaluate a model.
    
    Args:
        model: segmentation model
        dataloader: data loader for evaluation
        checkpoint_path: path to model checkpoint
        num_classes: number of segmentation classes
        patch_size: patch size for sliding window
        stride: stride for sliding window
        device: device to use
        save_predictions: whether to save predictions
        output_dir: directory to save results
        class_names: optional list of class names
    
    Returns:
        dict: evaluation results
    """
    evaluator = Evaluator(
        model=model,
        device=device,
        num_classes=num_classes,
        patch_size=patch_size,
        stride=stride,
        save_predictions=save_predictions,
        output_dir=output_dir,
    )
    
    evaluator.load_checkpoint(checkpoint_path)
    results = evaluator.evaluate(dataloader, class_names=class_names)
    evaluator.print_summary(results)
    
    if output_dir:
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        evaluator.save_results(results, results_path)
    
    return results
