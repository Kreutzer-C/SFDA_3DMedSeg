"""
Loss functions for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten(tensor):
    """Flattens a tensor: (N, C, D, H, W) -> (C, N * D * H * W)"""
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes Dice coefficient per channel.
    
    Args:
        input: NxCxSpatial input tensor (probabilities)
        target: NxCxSpatial target tensor
        epsilon: prevents division by zero
        weight: Cx1 tensor of weight per channel/class
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target).float()

    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    Reference: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, weight=None, normalization="softmax", smooth=1e-6):
        """
        Args:
            weight: class weights tensor
            normalization: 'sigmoid', 'softmax', or 'none'
            smooth: smoothing factor to prevent division by zero
        """
        super().__init__()
        self.register_buffer("weight", weight)
        self.smooth = smooth
        
        assert normalization in ["sigmoid", "softmax", "none"]
        if normalization == "sigmoid":
            self.normalization = nn.Sigmoid()
        elif normalization == "softmax":
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward(self, input, target):
        """
        Args:
            input: (N, C, D, H, W) logits
            target: (N, C, D, H, W) one-hot encoded or (N, D, H, W) class indices
        """
        # Convert target to one-hot if needed
        if input.dim() != target.dim():
            target = self._to_one_hot(target, input.size(1))
        
        input = self.normalization(input)
        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.smooth, weight=self.weight)
        return 1.0 - torch.mean(per_channel_dice)

    def _to_one_hot(self, target, num_classes):
        """Convert class indices to one-hot encoding."""
        N = target.size(0)
        spatial_dims = target.shape[1:]
        target_one_hot = torch.zeros(N, num_classes, *spatial_dims, device=target.device, dtype=torch.float32)
        target_one_hot.scatter_(1, target.unsqueeze(1).long(), 1)
        return target_one_hot


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss for handling class imbalance.
    Reference: https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, normalization="softmax", epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        
        assert normalization in ["sigmoid", "softmax", "none"]
        if normalization == "sigmoid":
            self.normalization = nn.Sigmoid()
        elif normalization == "softmax":
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward(self, input, target):
        if input.dim() != target.dim():
            target = self._to_one_hot(target, input.size(1))
            
        input = self.normalization(input)
        
        input = flatten(input)
        target = flatten(target).float()

        if input.size(0) == 1:
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1) * w_l
        denominator = ((input + target).sum(-1) * w_l).clamp(min=self.epsilon)

        return 1 - 2 * (intersect.sum() / denominator.sum())

    def _to_one_hot(self, target, num_classes):
        N = target.size(0)
        spatial_dims = target.shape[1:]
        target_one_hot = torch.zeros(N, num_classes, *spatial_dims, device=target.device, dtype=torch.float32)
        target_one_hot.scatter_(1, target.unsqueeze(1).long(), 1)
        return target_one_hot


class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss wrapper with optional class weights."""

    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, input, target):
        """
        Args:
            input: (N, C, D, H, W) logits
            target: (N, D, H, W) class indices
        """
        if target.dim() == input.dim():
            # Convert one-hot to class indices
            target = target.argmax(dim=1)
        return self.ce_loss(input, target.long())


class DiceCELoss(nn.Module):
    """
    Combined Dice and Cross Entropy Loss.
    """

    def __init__(self, dice_weight=1.0, ce_weight=1.0, class_weight=None, ignore_index=-100):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(weight=class_weight, normalization="softmax")
        self.ce_loss = CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index)

    def forward(self, input, target):
        dice = self.dice_loss(input, target)
        ce = self.ce_loss(input, target)
        return self.dice_weight * dice + self.ce_weight * ce


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if target.dim() == input.dim():
            target = target.argmax(dim=1)
            
        ce_loss = F.cross_entropy(input, target.long(), reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """Factory function to get loss function by name."""
    losses = {
        'DiceLoss': DiceLoss,
        'GeneralizedDiceLoss': GeneralizedDiceLoss,
        'CrossEntropyLoss': CrossEntropyLoss,
        'DiceCELoss': DiceCELoss,
        'FocalLoss': FocalLoss,
    }
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    return losses[loss_name](**kwargs)
