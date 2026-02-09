"""
Trainer module for SFDA 3D Medical Image Segmentation.
"""

from .base_trainer import BaseTrainer
from .source_trainer import SourcePretrainTrainer
from .target_trainer import OracleAdaptationTrainer
from .evaluator import Evaluator

__all__ = [
    'BaseTrainer',
    'SourcePretrainTrainer',
    'OracleAdaptationTrainer',
    'Evaluator'
]
