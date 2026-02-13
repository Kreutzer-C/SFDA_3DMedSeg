"""
Source domain pretraining trainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base_trainer import BaseTrainer
from utils.metrics import compute_dice_per_class
from utils.losses import DiceLoss, CrossEntropyLoss, GeneralizedDiceLoss


class SourcePretrainTrainer(BaseTrainer):
    """
    Trainer for source domain pretraining.
    Standard supervised training on labeled source domain data.
    
    Loss: Dice + CrossEntropy (combined)
    """
    
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        test_loader=None,
        scheduler=None,
        device='cuda',
        config=None,
        num_classes=5,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            scheduler=scheduler,
            device=device,
            config=config,
        )
        self.num_classes = num_classes
        
        # Loss functions (defined here, not via config)
        # Use Generalized Dice Loss to handle class imbalance
        self.dice_loss = GeneralizedDiceLoss(normalization='softmax')
        self.ce_loss = CrossEntropyLoss()

        print(f"Length of train loader: {len(train_loader)} patches")
        print(f"Length of test loader: {len(test_loader)} cases")
    
    def compute_loss(self, outputs, labels):
        """
        Compute combined loss.
        Override this method in subclasses for custom loss functions.
        
        Args:
            outputs: model outputs (logits)
            labels: ground truth labels
        
        Returns:
            loss: combined loss value
        """
        dice = self.dice_loss(outputs, labels)
        ce = self.ce_loss(outputs, labels)
        return dice + ce
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            # Use return_logits=True to get raw logits for loss computation
            outputs = self.model(images, return_logits=True)
            
            # Handle model output (returns tuple: (probs, logits))
            if isinstance(outputs, tuple):
                logits = outputs[1]  # Use logits for loss computation
            else:
                logits = outputs
            
            # Compute loss
            loss = self.compute_loss(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                dice_scores = compute_dice_per_class(
                    pred.cpu().numpy(),
                    labels.cpu().numpy(),
                    num_classes=self.num_classes,
                    include_background=False
                )
                batch_dice = sum(dice_scores.values()) / len(dice_scores) if dice_scores else 0.0
            
            total_loss += loss.item()
            total_dice += batch_dice
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{batch_dice:.4f}'
            })
        
        return {
            'loss': total_loss / num_batches,
            'dice': total_dice / num_batches,
        }
    
    def sliding_window_inference(self, image, patch_size=64, overlap=0.5):
        """
        Perform sliding window inference on a full volume.
        
        Args:
            image: (1, C, D, H, W) input tensor
            patch_size: size of the patch (int or tuple)
            overlap: overlap ratio between patches
        
        Returns:
            logits: (1, num_classes, D, H, W) output logits
        """
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        
        _, C, D, H, W = image.shape
        stride = [int(p * (1 - overlap)) for p in patch_size]
        
        # Initialize output and count tensors
        output = torch.zeros(1, self.num_classes, D, H, W, device=self.device)
        count = torch.zeros(1, 1, D, H, W, device=self.device)
        
        # Generate patch positions
        d_positions = list(range(0, max(1, D - patch_size[0] + 1), stride[0]))
        h_positions = list(range(0, max(1, H - patch_size[1] + 1), stride[1]))
        w_positions = list(range(0, max(1, W - patch_size[2] + 1), stride[2]))
        
        # Ensure we cover the entire volume
        if d_positions[-1] + patch_size[0] < D:
            d_positions.append(D - patch_size[0])
        if h_positions[-1] + patch_size[1] < H:
            h_positions.append(H - patch_size[1])
        if w_positions[-1] + patch_size[2] < W:
            w_positions.append(W - patch_size[2])
        
        for d in d_positions:
            for h in h_positions:
                for w in w_positions:
                    # Extract patch
                    patch = image[:, :, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]]
                    
                    # Pad if necessary (for edge cases)
                    pad_d = patch_size[0] - patch.shape[2]
                    pad_h = patch_size[1] - patch.shape[3]
                    pad_w = patch_size[2] - patch.shape[4]
                    if pad_d > 0 or pad_h > 0 or pad_w > 0:
                        patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d))
                    
                    # Forward pass
                    patch_output = self.model(patch, return_logits=True)
                    if isinstance(patch_output, tuple):
                        patch_logits = patch_output[1]
                    else:
                        patch_logits = patch_output
                    
                    # Remove padding if applied
                    if pad_d > 0:
                        patch_logits = patch_logits[:, :, :-pad_d, :, :]
                    if pad_h > 0:
                        patch_logits = patch_logits[:, :, :, :-pad_h, :]
                    if pad_w > 0:
                        patch_logits = patch_logits[:, :, :, :, :-pad_w]
                    
                    # Accumulate
                    actual_d = min(patch_size[0], D - d)
                    actual_h = min(patch_size[1], H - h)
                    actual_w = min(patch_size[2], W - w)
                    
                    output[:, :, d:d+actual_d, h:h+actual_h, w:w+actual_w] += patch_logits[:, :, :actual_d, :actual_h, :actual_w]
                    count[:, :, d:d+actual_d, h:h+actual_h, w:w+actual_w] += 1
        
        # Average overlapping regions
        output = output / count.clamp(min=1)
        return output
    
    def test(self):
        """Test the model using sliding window inference."""
        self.model.eval()
        
        total_loss = 0.0
        class_dice_scores = {c: [] for c in range(1, self.num_classes)}
        num_batches = 0
        
        # Get patch size from config
        patch_size = self.config.get('patch_size', 64) if self.config else 64
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Use sliding window inference for full volumes
                logits = self.sliding_window_inference(images, patch_size=patch_size, overlap=0.5)
                
                # Compute loss
                loss = self.compute_loss(logits, labels)
                total_loss += loss.item()
                
                # Compute per-class Dice
                pred = logits.argmax(dim=1)
                dice_scores = compute_dice_per_class(
                    pred.cpu().numpy(),
                    labels.cpu().numpy(),
                    num_classes=self.num_classes,
                    include_background=False
                )
                
                for c, score in dice_scores.items():
                    class_dice_scores[c].append(score)
                
                num_batches += 1
        
        # Compute average metrics
        metrics = {
            'loss': total_loss / num_batches,
        }
        
        # Per-class Dice
        for c in range(1, self.num_classes):
            if class_dice_scores[c]:
                metrics[f'dice_class_{c}'] = sum(class_dice_scores[c]) / len(class_dice_scores[c])
        
        # Mean Dice (excluding background)
        dice_values = [metrics[f'dice_class_{c}'] for c in range(1, self.num_classes) if f'dice_class_{c}' in metrics]
        metrics['dice'] = sum(dice_values) / len(dice_values) if dice_values else 0.0
        
        return metrics


def create_source_trainer(
    model,
    train_loader,
    test_loader=None,
    config=None,
):
    """
    Factory function to create a source pretraining trainer.
    
    Args:
        model: segmentation model
        train_loader: training data loader
        test_loader: test data loader
        config: configuration dictionary
    
    Returns:
        SourcePretrainTrainer instance
    """
    config = config or {}
    
    # Setup optimizer
    lr = config.get('learning_rate', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup scheduler
    scheduler_config = config.get('scheduler', None)
    if scheduler_config:
        scheduler_name = scheduler_config.get('name', 'CosineAnnealingLR')
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get('num_epochs', 100),
                eta_min=config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        else:
            scheduler = None
    else:
        scheduler = None
    
    trainer = SourcePretrainTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        device=config.get('device', 'cuda'),
        config=config,
        num_classes=config.get('num_classes', 5),
    )
    
    return trainer
