"""
Base trainer class for 3D medical image segmentation.
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime

import torch
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_logger(name, log_file=None, level=logging.INFO):
    """Create a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class BaseTrainer(ABC):
    """
    Base class for all trainers.
    Provides common functionality for training, testing, checkpointing, and logging.
    
    Note: No validation set is used. Testing is performed after each training epoch.
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
    ):
        """
        Args:
            model: segmentation model
            optimizer: optimizer
            train_loader: training data loader
            test_loader: test data loader (for evaluation after each epoch)
            scheduler: learning rate scheduler
            device: device to use ('cuda' or 'cpu')
            config: configuration dictionary
        """
        self.config = config or {}
        
        # Setup device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Model setup with DataParallel for multi-GPU
        self.model = model
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Setup wandb
        self._setup_wandb()
    
    def _setup_directories(self):
        """Setup result directories."""
        # Get paths from config
        results_dir = self.config.get('results_dir', './results')
        dataset_name = self.config.get('dataset_name', 'unknown')
        domain_name = self.config.get('domain_name', 'unknown')
        exp_name = self.config.get('exp_name', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Create directory structure: results/dataset/domain/exp_name/
        self.exp_dir = os.path.join(results_dir, dataset_name, domain_name, exp_name)
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def _setup_logging(self):
        """Setup logging."""
        log_file = os.path.join(self.log_dir, 'train.log')
        self.logger = get_logger('Trainer', log_file)
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        self.use_wandb = self.config.get('use_wandb', False) and WANDB_AVAILABLE
        
        if self.use_wandb:
            wandb.init(
                project=self.config.get('wandb_project', 'SFDA_3DMedSeg'),
                name=self.config.get('exp_name', None),
                config=self.config,
                dir=self.exp_dir,
            )
            wandb.watch(self.model)
    
    def train(self, num_epochs):
        """
        Main training loop.
        
        Args:
            num_epochs: number of epochs to train
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Results will be saved to: {self.exp_dir}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self.train_epoch()
            
            # Log training metrics
            self._log_metrics(train_metrics, prefix='train', epoch=epoch)
            
            # Test after each epoch
            if self.test_loader is not None:
                test_metrics = self.test()
                self._log_metrics(test_metrics, prefix='test', epoch=epoch)
                
                # Check if best model
                current_metric = test_metrics.get('dice', 0.0)
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    self.logger.info(f"New best model! Dice: {current_metric:.4f}")
            else:
                is_best = False
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(test_metrics.get('loss', train_metrics.get('loss', 0)))
                else:
                    self.scheduler.step()
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.use_wandb:
                wandb.log({'learning_rate': current_lr}, step=epoch)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Best test Dice: {self.best_metric:.4f}")
        
        # Close logging
        if self.use_wandb:
            wandb.finish()
    
    @abstractmethod
    def train_epoch(self):
        """
        Train for one epoch.
        Must be implemented by subclasses.
        
        Returns:
            dict: training metrics
        """
        pass
    
    @abstractmethod
    def test(self):
        """
        Test the model.
        Must be implemented by subclasses.
        
        Returns:
            dict: test metrics
        """
        pass
    
    def _log_metrics(self, metrics, prefix, epoch):
        """Log metrics to wandb and console."""
        # Console logging
        metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} [{prefix}] {metrics_str}")
        
        # Wandb
        if self.use_wandb:
            wandb_metrics = {f'{prefix}/{name}': value for name, value in metrics.items()}
            wandb.log(wandb_metrics, step=epoch)
    
    def save_checkpoint(self, is_best=False, filename=None):
        """Save model checkpoint."""
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch}.pth'
        
        # Get model state dict (handle DataParallel)
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save last checkpoint
        last_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
        
        # Save periodic checkpoint
        if self.current_epoch % self.config.get('save_every', 10) == 0:
            periodic_path = os.path.join(self.checkpoint_dir, filename)
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        self.logger.info(f"Resumed from epoch {self.current_epoch}, best metric: {self.best_metric:.4f}")
