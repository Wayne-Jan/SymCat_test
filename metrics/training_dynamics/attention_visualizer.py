#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VSI Attention Visualization Module

This module provides utilities for visualizing attention maps from
the Visual-Semantic Interaction (VSI) module during training.

Key features:
- Attention map extraction
- Visualization of attention patterns
- Comparison of attention focus across different inputs

Usage:
    from metrics.training_dynamics.attention_visualizer import AttentionVisualizer
    
    # Initialize the visualizer
    visualizer = AttentionVisualizer(
        model=model,
        save_dir="visualizations/training_dynamics"
    )
    
    # Register hooks to capture attention
    visualizer.register_hooks()
    
    # Forward pass in model (attention maps will be captured)
    outputs = model(inputs)
    
    # Visualize attention maps
    visualizer.visualize_attention()
    
    # Remove hooks when done
    visualizer.remove_hooks()
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Set plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


class AttentionVisualizer:
    """
    Visualizes attention maps from the VSI module during training.
    
    This class provides utilities to:
    1. Extract attention maps from forward passes
    2. Visualize attention patterns across different prompts
    3. Track changes in attention focus during training
    """
    
    def __init__(self, model, save_dir=None):
        """
        Initialize the attention visualizer.
        
        Args:
            model: PyTorch model with VSI module
            save_dir: Directory to save visualizations (optional)
        """
        self.model = model
        self.save_dir = save_dir
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        self.hooks = []
        self.attention_maps = []
        self.epoch = 0
        self.batch = 0
    
    def register_hooks(self):
        """Register hooks to capture attention maps during forward pass."""
        self.remove_hooks()  # Remove any existing hooks
        
        # Find VSI module in the model
        vsi_module = self._find_vsi_module(self.model)
        
        if vsi_module is None:
            print("Warning: VSI module not found in the model")
            return self
        
        # Register hook for attention
        hook = vsi_module.register_forward_hook(
            lambda module, input, output: self._attention_hook(module, input, output)
        )
        self.hooks.append(hook)
        
        return self
    
    def _find_vsi_module(self, model):
        """Find the VSI module in the model."""
        # Try to find a module named VSI or VisualSemanticInteraction
        for name, module in model.named_modules():
            if "vsi" in name.lower() or "visualsemanticinteraction" in name.lower():
                return module
            
            # Check if module has an attribute 'attention' which might be a multi-head attention
            if hasattr(module, 'attention') and hasattr(module.attention, 'softmax'):
                return module
        
        return None
    
    def _attention_hook(self, module, input, output):
        """Hook function to store attention maps."""
        # Try to extract attention weights
        # This depends on the specific implementation of the VSI module
        attention = None
        
        # Check if module has attention weights as attribute
        if hasattr(module, 'attention_weights'):
            attention = module.attention_weights.detach().cpu()
        
        # Check if output is a tuple and contains attention weights
        elif isinstance(output, tuple) and len(output) > 1:
            attention = output[1].detach().cpu()
        
        if attention is not None:
            self.attention_maps.append({
                'epoch': self.epoch,
                'batch': self.batch,
                'attention': attention
            })
    
    def update(self, epoch=None, batch=None):
        """
        Update tracking with current state.
        
        Args:
            epoch: Current epoch number (optional)
            batch: Current batch number (optional)
        """
        if epoch is not None:
            self.epoch = epoch
        if batch is not None:
            self.batch = batch
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_attention(self, sample_idx=0, head_idx=0):
        """
        Visualize attention maps.
        
        Args:
            sample_idx: Index of the sample in the batch to visualize
            head_idx: Index of the attention head to visualize
        """
        if not self.attention_maps:
            return None
        
        # Get the latest attention map
        latest = self.attention_maps[-1]
        attention = latest['attention']
        
        # Check dimensions and extract the sample and head
        if len(attention.shape) == 4:  # [batch_size, num_heads, q_len, k_len]
            attention_map = attention[sample_idx, head_idx]
        elif len(attention.shape) == 3:  # [batch_size, q_len, k_len]
            attention_map = attention[sample_idx]
        else:
            print(f"Warning: Unexpected attention shape: {attention.shape}")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot attention as a heatmap
        sns.heatmap(attention_map.numpy(), cmap='viridis', cbar=True)
        
        # Add labels and title
        plt.xlabel('Key Position (Text)')
        plt.ylabel('Query Position (Image)')
        plt.title(f'VSI Attention Map (Epoch {self.epoch}, Batch {self.batch})')
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"attention_map_e{self.epoch}_b{self.batch}_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_attention_evolution(self, sample_idx=0, head_idx=0, num_maps=5):
        """
        Visualize how attention maps evolve during training.
        
        Args:
            sample_idx: Index of the sample in the batch to visualize
            head_idx: Index of the attention head to visualize
            num_maps: Number of attention maps to show
        """
        if len(self.attention_maps) < 2:
            return None
        
        # Select a subset of attention maps at regular intervals
        indices = np.linspace(0, len(self.attention_maps)-1, num_maps, dtype=int)
        selected_maps = [self.attention_maps[i] for i in indices]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, num_maps, figsize=(16, 4))
        
        for i, map_data in enumerate(selected_maps):
            attention = map_data['attention']
            
            # Check dimensions and extract the sample and head
            if len(attention.shape) == 4:  # [batch_size, num_heads, q_len, k_len]
                attention_map = attention[sample_idx, head_idx]
            elif len(attention.shape) == 3:  # [batch_size, q_len, k_len]
                attention_map = attention[sample_idx]
            else:
                continue
            
            # Plot attention as a heatmap
            im = sns.heatmap(attention_map.numpy(), cmap='viridis', cbar=(i == num_maps-1),
                           ax=axes[i])
            
            # Add title
            epoch = map_data['epoch']
            axes[i].set_title(f'Epoch {epoch}')
            
            # Remove labels except for first plot
            if i > 0:
                axes[i].set_ylabel('')
            else:
                axes[i].set_ylabel('Query Position (Image)')
            
            # Add x-label only to the bottom row
            if i == num_maps // 2:
                axes[i].set_xlabel('Key Position (Text)')
        
        plt.suptitle('Evolution of VSI Attention Maps During Training')
        plt.tight_layout()
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"attention_evolution_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_attention_across_heads(self, sample_idx=0, max_heads=8):
        """
        Visualize attention maps across different heads.
        
        Args:
            sample_idx: Index of the sample in the batch to visualize
            max_heads: Maximum number of heads to display
        """
        if not self.attention_maps:
            return None
        
        # Get the latest attention map
        latest = self.attention_maps[-1]
        attention = latest['attention']
        
        # Check if we have multi-head attention
        if len(attention.shape) != 4:
            print("Warning: No multi-head attention detected")
            return None
        
        # Determine number of heads to display
        num_heads = min(attention.shape[1], max_heads)
        
        # Create figure with subplots
        nrows = (num_heads + 3) // 4  # Ceiling division
        ncols = min(4, num_heads)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3*nrows))
        
        # Handle case where there's only one row
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each head's attention
        for i in range(num_heads):
            row, col = divmod(i, ncols)
            attention_map = attention[sample_idx, i]
            
            # Plot attention as a heatmap
            sns.heatmap(attention_map.numpy(), cmap='viridis', cbar=(i == num_heads-1),
                       ax=axes[row, col])
            
            # Add title
            axes[row, col].set_title(f'Head {i+1}')
            
            # Remove labels except for leftmost plots
            if col > 0:
                axes[row, col].set_ylabel('')
            else:
                axes[row, col].set_ylabel('Query Position (Image)')
            
            # Add x-label only to the bottom row
            if row == nrows - 1:
                axes[row, col].set_xlabel('Key Position (Text)')
            
        # Hide any unused subplots
        for i in range(num_heads, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row, col].axis('off')
        
        plt.suptitle(f'VSI Attention Across Different Heads (Epoch {self.epoch})')
        plt.tight_layout()
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"attention_heads_e{self.epoch}_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_attention_statistics(self):
        """Visualize statistics about attention patterns over time."""
        if len(self.attention_maps) < 2:
            return None
        
        # Extract data
        epochs = [data['epoch'] for data in self.attention_maps]
        
        # Compute attention entropy (measure of focus)
        entropies = []
        for data in self.attention_maps:
            attention = data['attention']
            if len(attention.shape) == 4:  # [batch_size, num_heads, q_len, k_len]
                # Average over batch and heads
                attention = attention.mean(dim=[0, 1])
            elif len(attention.shape) == 3:  # [batch_size, q_len, k_len]
                # Average over batch
                attention = attention.mean(dim=0)
            
            # Compute entropy
            attention = attention / attention.sum()
            entropy = -torch.sum(attention * torch.log(attention + 1e-10))
            entropies.append(entropy.item())
        
        # Compute attention sparsity (measure of concentration)
        sparsities = []
        for data in self.attention_maps:
            attention = data['attention']
            if len(attention.shape) == 4:
                attention = attention.mean(dim=[0, 1])
            elif len(attention.shape) == 3:
                attention = attention.mean(dim=0)
            
            # Compute sparsity (1 - L1/L0)
            l1_norm = torch.sum(torch.abs(attention))
            l0_norm = torch.sum(attention != 0).float()
            if l0_norm > 0:
                sparsity = 1 - l1_norm / l0_norm
            else:
                sparsity = torch.tensor(1.0)
            sparsities.append(sparsity.item())
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot entropy over time
        ax1.plot(epochs, entropies, 'o-', label='Entropy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Attention Entropy')
        ax1.set_title('Attention Focus Over Time (lower entropy = more focus)')
        ax1.grid(True, alpha=0.3)
        
        # Plot sparsity over time
        ax2.plot(epochs, sparsities, 'o-', label='Sparsity')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Attention Sparsity')
        ax2.set_title('Attention Concentration Over Time (higher sparsity = more selective)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"attention_statistics_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize(self):
        """Generate all visualizations."""
        results = {}
        
        # Current attention map
        attention_map_path = self.visualize_attention()
        if attention_map_path:
            results["attention_map"] = attention_map_path
        
        # Attention evolution
        evolution_path = self.visualize_attention_evolution()
        if evolution_path:
            results["attention_evolution"] = evolution_path
        
        # Attention across heads
        heads_path = self.visualize_attention_across_heads()
        if heads_path:
            results["attention_heads"] = heads_path
        
        # Attention statistics
        stats_path = self.visualize_attention_statistics()
        if stats_path:
            results["attention_statistics"] = stats_path
        
        # Save results metadata
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_path = os.path.join(self.save_dir, f"attention_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results