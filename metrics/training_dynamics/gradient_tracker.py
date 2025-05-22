#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradient Tracking Module

This module provides utilities for tracking and visualizing gradient flow
during neural network training.

Key features:
- Layer-wise gradient norm tracking
- Gradient vanishing/exploding detection
- Weight distribution visualization over time

Usage:
    from metrics.training_dynamics.gradient_tracker import GradientTracker
    
    # Initialize the tracker
    tracker = GradientTracker(
        model=model,
        save_dir="visualizations/training_dynamics"
    )
    
    # Register model hooks
    tracker.register_hooks()
    
    # During training (after loss.backward())
    tracker.update()
    
    # Generate visualizations
    tracker.visualize()
    
    # Remove hooks when done
    tracker.remove_hooks()
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# Set plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


class GradientTracker:
    """
    Tracks and analyzes gradient flow and weight distributions during training.
    
    This class provides utilities to:
    1. Track per-layer gradient norms
    2. Detect gradient vanishing/exploding
    3. Visualize weight distributions over time
    """
    
    def __init__(self, model, save_dir=None, track_weights=True):
        """
        Initialize the gradient tracker.
        
        Args:
            model: PyTorch model to track
            save_dir: Directory to save visualizations (optional)
            track_weights: Whether to track weight distributions (default: True)
        """
        self.model = model
        self.save_dir = save_dir
        self.track_weights = track_weights
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Initialize storage
        self.hooks = []
        self.gradients = defaultdict(list)
        self.weights = defaultdict(list) if track_weights else None
        self.epochs = []
        self.current_epoch = 0
        self.current_batch = 0
        self.gradient_norms = defaultdict(list)
        self.weight_norms = defaultdict(list) if track_weights else None
    
    def register_hooks(self):
        """Register hooks to capture gradients during backpropagation."""
        self.remove_hooks()  # Remove any existing hooks
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, name=name: self._gradient_hook(grad, name)
                )
                self.hooks.append(hook)
        
        return self
    
    def _gradient_hook(self, grad, name):
        """Hook function to store gradients."""
        if self.current_batch % 10 == 0:  # Only store every 10 batches to save memory
            self.gradients[name].append(grad.detach().cpu().clone())
        
        # Always compute and store the norm
        self.gradient_norms[name].append(grad.norm().item())
        
        return grad
    
    def update(self, epoch=None, batch=None):
        """
        Update tracking with current state.
        
        Args:
            epoch: Current epoch number (optional)
            batch: Current batch number (optional)
        """
        if epoch is not None:
            self.current_epoch = epoch
        if batch is not None:
            self.current_batch = batch
        
        # Track weight distributions if enabled
        if self.track_weights:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if self.current_batch % 10 == 0:  # Only store every 10 batches
                        self.weights[name].append(param.data.detach().cpu().clone())
                    
                    # Always compute and store the norm
                    self.weight_norms[name].append(param.data.norm().item())
        
        # Add epoch to tracking list if it's a new epoch
        if epoch is not None and (not self.epochs or self.epochs[-1] != epoch):
            self.epochs.append(epoch)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_gradient_flow(self):
        """Visualize the gradient flow through the network layers."""
        if not self.gradient_norms:
            return None
        
        # Get layer names and sort them by parameter count
        all_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.gradient_norms:
                all_layers.append((name, param.numel()))
        
        # Sort layers by parameter count
        all_layers.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 20 layers by parameter count
        layers = [name for name, _ in all_layers[:20]]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # For each layer, plot gradient norm over time
        for i, name in enumerate(layers):
            if name in self.gradient_norms and len(self.gradient_norms[name]) > 0:
                # Take median over batches for each epoch
                avg_grads = []
                pos = 0
                for epoch in self.epochs:
                    next_pos = pos + 10  # Assuming 10 batches per epoch for simplicity
                    if next_pos <= len(self.gradient_norms[name]):
                        epoch_grads = self.gradient_norms[name][pos:next_pos]
                        avg_grads.append(np.median(epoch_grads))
                    pos = next_pos
                
                if avg_grads:
                    plt.semilogy(self.epochs[:len(avg_grads)], avg_grads, label=name)
        
        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title('Gradient Flow')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"gradient_flow_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_weight_distribution(self, top_n=5):
        """
        Visualize weight distribution changes over time.
        
        Args:
            top_n: Number of top layers (by parameter count) to visualize
        """
        if not self.track_weights or not self.weights:
            return None
        
        # Get layer names and sort them by parameter count
        all_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.weights:
                all_layers.append((name, param.numel()))
        
        # Sort layers by parameter count
        all_layers.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N layers
        layers = [name for name, _ in all_layers[:top_n]]
        
        # Create figure with subplots
        fig, axes = plt.subplots(top_n, 1, figsize=(12, 4*top_n))
        
        # If only one layer, axes is not a list
        if top_n == 1:
            axes = [axes]
        
        # For each layer, plot weight distributions at start, middle, and end
        for i, name in enumerate(layers):
            if name in self.weights and len(self.weights[name]) > 2:
                # Get weights at start, middle, and end
                start_idx = 0
                mid_idx = len(self.weights[name]) // 2
                end_idx = len(self.weights[name]) - 1
                
                weights_start = self.weights[name][start_idx].flatten().numpy()
                weights_mid = self.weights[name][mid_idx].flatten().numpy()
                weights_end = self.weights[name][end_idx].flatten().numpy()
                
                # Plot histograms
                axes[i].hist(weights_start, bins=50, alpha=0.5, label='Start', density=True)
                axes[i].hist(weights_mid, bins=50, alpha=0.5, label='Middle', density=True)
                axes[i].hist(weights_end, bins=50, alpha=0.5, label='Current', density=True)
                
                # Add labels
                axes[i].set_title(f'Weight Distribution for {name}')
                axes[i].set_xlabel('Weight Value')
                axes[i].set_ylabel('Density')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"weight_distribution_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_weight_norm_changes(self):
        """Visualize how weight norms change over training."""
        if not self.track_weights or not self.weight_norms:
            return None
        
        # Get layer names and sort them by parameter count
        all_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.weight_norms:
                all_layers.append((name, param.numel()))
        
        # Sort layers by parameter count
        all_layers.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 20 layers by parameter count
        layers = [name for name, _ in all_layers[:20]]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # For each layer, plot weight norm over time
        for i, name in enumerate(layers):
            if name in self.weight_norms and len(self.weight_norms[name]) > 0:
                # Take median over batches for each epoch
                avg_norms = []
                pos = 0
                for epoch in self.epochs:
                    next_pos = pos + 10  # Assuming 10 batches per epoch for simplicity
                    if next_pos <= len(self.weight_norms[name]):
                        epoch_norms = self.weight_norms[name][pos:next_pos]
                        avg_norms.append(np.median(epoch_norms))
                    pos = next_pos
                
                if avg_norms:
                    plt.plot(self.epochs[:len(avg_norms)], avg_norms, label=name)
        
        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Weight Norm')
        plt.title('Weight Norm Changes During Training')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"weight_norm_changes_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_gradient_correlation(self):
        """Visualize correlation between gradients and layer depth."""
        if not self.gradient_norms:
            return None
        
        # Get layer depths (assuming ordered by depth based on name)
        layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.gradient_norms:
                # Extract depth information from name
                depth = name.count('.')  # Count dots as proxy for depth
                layers.append((name, depth, np.median(self.gradient_norms[name])))
        
        # Sort layers by depth
        layers.sort(key=lambda x: x[1])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create scatter plot
        names = [name for name, _, _ in layers]
        depths = [depth for _, depth, _ in layers]
        norms = [norm for _, _, norm in layers]
        
        scatter = plt.scatter(depths, norms, c=depths, cmap='viridis', 
                            alpha=0.7, s=100)
        
        # Add layer names as annotations
        for i, name in enumerate(names):
            plt.annotate(name, (depths[i], norms[i]), fontsize=8,
                       xytext=(5, 5), textcoords='offset points')
        
        # Add labels and title
        plt.xlabel('Layer Depth')
        plt.ylabel('Median Gradient Norm')
        plt.title('Gradient Magnitude vs. Layer Depth')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Layer Depth')
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"gradient_correlation_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def detect_gradient_issues(self):
        """Detect vanishing or exploding gradients."""
        if not self.gradient_norms:
            return None
        
        # Check for vanishing gradients (median norm < 1e-3)
        vanishing_layers = []
        for name, norms in self.gradient_norms.items():
            if np.median(norms) < 1e-3:
                vanishing_layers.append((name, np.median(norms)))
        
        # Check for exploding gradients (median norm > 1e3)
        exploding_layers = []
        for name, norms in self.gradient_norms.items():
            if np.median(norms) > 1e3:
                exploding_layers.append((name, np.median(norms)))
        
        # Create report
        result = {
            "vanishing_gradients": {
                "detected": bool(vanishing_layers),
                "layers": [{"name": name, "norm": float(norm)} for name, norm in vanishing_layers]
            },
            "exploding_gradients": {
                "detected": bool(exploding_layers),
                "layers": [{"name": name, "norm": float(norm)} for name, norm in exploding_layers]
            }
        }
        
        # Save the report
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"gradient_issues_{timestamp}.json")
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
        
        return result
    
    def visualize(self):
        """Generate all visualizations."""
        results = {}
        
        # Gradient flow
        gradient_flow_path = self.visualize_gradient_flow()
        if gradient_flow_path:
            results["gradient_flow"] = gradient_flow_path
        
        # Weight distribution
        if self.track_weights:
            weight_dist_path = self.visualize_weight_distribution()
            if weight_dist_path:
                results["weight_distribution"] = weight_dist_path
            
            weight_norm_path = self.visualize_weight_norm_changes()
            if weight_norm_path:
                results["weight_norm_changes"] = weight_norm_path
        
        # Gradient correlation
        gradient_corr_path = self.visualize_gradient_correlation()
        if gradient_corr_path:
            results["gradient_correlation"] = gradient_corr_path
        
        # Gradient issues
        gradient_issues = self.detect_gradient_issues()
        if gradient_issues:
            results["gradient_issues"] = gradient_issues
        
        # Save results metadata
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_path = os.path.join(self.save_dir, f"gradient_tracking_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                metadata = {k: v for k, v in results.items() if k != "gradient_issues"}
                metadata["gradient_issues_detected"] = bool(gradient_issues["vanishing_gradients"]["detected"] or 
                                                          gradient_issues["exploding_gradients"]["detected"])
                json.dump(metadata, f, indent=2)
        
        return results