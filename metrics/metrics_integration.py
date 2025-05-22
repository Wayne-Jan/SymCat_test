#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics Integration Module

This module provides utilities for integrating all metrics tracking tools
into the training pipeline in a seamless way.

Key features:
- Integration with train_eval.py and train_baseline.py
- Comprehensive metrics collection during training
- Automatic visualization generation
- Result storing alongside model checkpoints

Usage:
    from metrics.metrics_integration import MetricsManager
    
    # Initialize the manager
    metrics_manager = MetricsManager(
        model=model,
        attributes=attributes,
        label_to_idx_dict=label_to_idx_dict,
        idx_to_label_dict=idx_to_label_dict,
        hmt_info=hmt_info,
        save_dir="visualizations"
    )
    
    # During training loop (each epoch)
    metrics_manager.start_epoch(epoch)
    
    # Before forward pass
    metrics_manager.start_component("forward_pass")
    
    # ... Forward pass operations ...
    
    # After forward pass
    metrics_manager.end_component("forward_pass")
    
    # Before backward pass
    metrics_manager.start_component("backward_pass")
    
    # ... Backward pass operations ...
    
    # After backward pass
    metrics_manager.end_component("backward_pass")
    
    # Update with epoch results
    metrics_manager.update_epoch_results(logits_dict, labels_dict, batch_size, num_samples)
    
    # At the end of training
    metrics_manager.finalize()
    
    # Get all metrics data
    metrics_data = metrics_manager.get_metrics()
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from .class_performance import ClassPerformanceTracker
from .training_dynamics import GradientTracker, AttentionVisualizer
from .resource_usage import ResourceMonitor


class MetricsManager:
    """
    Integrates and manages all metrics tracking tools in the training pipeline.
    
    This class provides a unified interface to:
    1. Track class performance metrics
    2. Monitor training dynamics
    3. Track resource usage
    4. Generate visualizations
    5. Store metrics alongside model checkpoints
    """
    
    def __init__(self, model=None, attributes=None, label_to_idx_dict=None, 
                idx_to_label_dict=None, hmt_info=None, save_dir=None):
        """
        Initialize the metrics manager.
        
        Args:
            model: PyTorch model to track (optional)
            attributes: List of attributes to track (optional)
            label_to_idx_dict: Dictionary mapping labels to indices (optional)
            idx_to_label_dict: Dictionary mapping indices to labels (optional)
            hmt_info: Information about head/medium/tail class distribution (optional)
            save_dir: Directory to save visualizations (optional)
        """
        self.model = model
        self.attributes = attributes
        self.label_to_idx_dict = label_to_idx_dict
        self.idx_to_label_dict = idx_to_label_dict
        self.hmt_info = hmt_info
        self.save_dir = save_dir
        
        # Create save directories
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(os.path.join(save_dir, "class_metrics"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "training_dynamics"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "resource_usage"), exist_ok=True)
        
        # Initialize trackers
        self._init_trackers()
        
        # Current state
        self.current_epoch = None
        self.is_tracking = False
    
    def _init_trackers(self):
        """Initialize individual metric trackers."""
        # Class performance tracker
        if self.attributes and self.label_to_idx_dict and self.idx_to_label_dict:
            class_save_dir = os.path.join(self.save_dir, "class_metrics") if self.save_dir else None
            self.class_tracker = ClassPerformanceTracker(
                attributes=self.attributes,
                label_to_idx_dict=self.label_to_idx_dict,
                idx_to_label_dict=self.idx_to_label_dict,
                hmt_info=self.hmt_info,
                save_dir=class_save_dir
            )
        else:
            self.class_tracker = None
        
        # Training dynamics trackers
        if self.model:
            dynamics_save_dir = os.path.join(self.save_dir, "training_dynamics") if self.save_dir else None
            
            # Gradient tracker
            self.gradient_tracker = GradientTracker(
                model=self.model,
                save_dir=dynamics_save_dir,
                track_weights=True
            )
            
            # Attention visualizer
            self.attention_visualizer = AttentionVisualizer(
                model=self.model,
                save_dir=dynamics_save_dir
            )
        else:
            self.gradient_tracker = None
            self.attention_visualizer = None
        
        # Resource monitor
        resource_save_dir = os.path.join(self.save_dir, "resource_usage") if self.save_dir else None
        self.resource_monitor = ResourceMonitor(
            save_dir=resource_save_dir,
            track_gpu_memory=torch.cuda.is_available()
        )
    
    def start_tracking(self):
        """Start metrics tracking."""
        if self.is_tracking:
            return
        
        # Register hooks for training dynamics
        if self.gradient_tracker:
            self.gradient_tracker.register_hooks()
        
        if self.attention_visualizer:
            self.attention_visualizer.register_hooks()
        
        self.is_tracking = True
    
    def stop_tracking(self):
        """Stop metrics tracking."""
        if not self.is_tracking:
            return
        
        # Remove hooks
        if self.gradient_tracker:
            self.gradient_tracker.remove_hooks()
        
        if self.attention_visualizer:
            self.attention_visualizer.remove_hooks()
        
        self.is_tracking = False
    
    def start_epoch(self, epoch_num):
        """
        Start tracking a new epoch.
        
        Args:
            epoch_num: Current epoch number
        """
        self.current_epoch = epoch_num
        
        # Start tracking if not already tracking
        if not self.is_tracking:
            self.start_tracking()
        
        # Start resource monitoring
        self.resource_monitor.start_epoch(epoch_num)
        
        # Update trackers with epoch info
        if self.gradient_tracker:
            self.gradient_tracker.update(epoch=epoch_num)
        
        if self.attention_visualizer:
            self.attention_visualizer.update(epoch=epoch_num)
    
    def end_epoch(self, batch_size=None, num_samples=None):
        """
        End tracking for the current epoch.
        
        Args:
            batch_size: Batch size used in the epoch (optional)
            num_samples: Total number of samples processed (optional)
        """
        # End resource monitoring
        self.resource_monitor.end_epoch(batch_size=batch_size, num_samples=num_samples)
    
    def start_component(self, component_name):
        """
        Start timing a specific component of the training process.
        
        Args:
            component_name: Name of the component (e.g., "forward_pass", "backward_pass")
        """
        self.resource_monitor.start_component(component_name)
    
    def end_component(self, component_name=None):
        """
        End timing a specific component.
        
        Args:
            component_name: Name of the component (optional, uses current if None)
        """
        self.resource_monitor.end_component(component_name)
    
    def update_batch(self, batch_idx=None):
        """
        Update tracking with current batch information.
        
        Args:
            batch_idx: Current batch index (optional)
        """
        # Update trackers with batch info
        if self.gradient_tracker:
            self.gradient_tracker.update(batch=batch_idx)
        
        if self.attention_visualizer:
            self.attention_visualizer.update(batch=batch_idx)
    
    def update_epoch_results(self, logits_dict, labels_dict, batch_size=None, num_samples=None):
        """
        Update tracking with epoch results.
        
        Args:
            logits_dict: Dictionary of model logits for each attribute
            labels_dict: Dictionary of ground truth labels for each attribute
            batch_size: Batch size used (optional)
            num_samples: Total number of samples processed (optional)
        """
        # Update class performance tracking
        if self.class_tracker:
            self.class_tracker.update(logits_dict, labels_dict)
        
        # End the epoch
        self.end_epoch(batch_size=batch_size, num_samples=num_samples)
    
    def generate_visualizations(self):
        """Generate visualizations for all tracked metrics."""
        results = {}
        
        # Class performance visualizations
        if self.class_tracker:
            class_results = self.class_tracker.visualize()
            results["class_performance"] = class_results
        
        # Training dynamics visualizations
        if self.gradient_tracker:
            gradient_results = self.gradient_tracker.visualize()
            results["gradient_tracking"] = gradient_results
        
        if self.attention_visualizer:
            attention_results = self.attention_visualizer.visualize()
            results["attention_visualization"] = attention_results
        
        # Resource usage visualizations
        resource_results = self.resource_monitor.visualize()
        results["resource_usage"] = resource_results
        
        # Save consolidated results metadata
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_path = os.path.join(self.save_dir, f"metrics_visualizations_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def get_metrics(self):
        """Get all metrics data as a dictionary."""
        metrics_data = {}
        
        # Class performance metrics
        if self.class_tracker:
            metrics_data["class_performance"] = {
                "class_metrics": self.class_tracker.get_metrics(),
                "type_metrics": self.class_tracker.get_class_type_metrics()
            }
        
        # Resource usage metrics
        metrics_data["resource_usage"] = {
            "epoch_data": self.resource_monitor.epoch_data,
            "component_data": {k: v for k, v in self.resource_monitor.component_data.items()}
        }
        
        # Add gradient issues if available
        if self.gradient_tracker:
            gradient_issues = self.gradient_tracker.detect_gradient_issues()
            if gradient_issues:
                metrics_data["gradient_issues"] = gradient_issues
        
        return metrics_data
    
    def finalize(self):
        """Finalize tracking, generate visualizations, and save metrics."""
        # Stop tracking
        self.stop_tracking()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save metrics
        metrics_data = self.get_metrics()
        
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_path = os.path.join(self.save_dir, f"all_metrics_{timestamp}.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, cls=JSONEncoder, indent=2)
        
        return metrics_data


class JSONEncoder(json.JSONEncoder):
    """Extended JSON encoder that can handle numpy and torch types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return super().default(obj)