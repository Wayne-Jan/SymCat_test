#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resource Usage Monitoring Module

This module provides utilities for tracking and visualizing computational
resource usage during model training, including GPU memory and computation time.

Key features:
- GPU memory usage tracking
- Component-wise timing analysis
- Throughput monitoring

Usage:
    from metrics.resource_usage.resource_monitor import ResourceMonitor
    
    # Initialize the monitor
    monitor = ResourceMonitor(save_dir="visualizations/resource_usage")
    
    # Start timing a training epoch
    monitor.start_epoch(epoch_num=1)
    
    # Start timing a specific component
    monitor.start_component("forward_pass")
    
    # ... Forward pass operations ...
    
    # End component timing
    monitor.end_component("forward_pass")
    
    # ... Repeat for other components ...
    
    # End epoch timing and record batch size
    monitor.end_epoch(batch_size=32, num_samples=1024)
    
    # Generate visualizations
    monitor.visualize()
"""

import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Set plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


class ResourceMonitor:
    """
    Monitors computational resources during model training.
    
    This class provides utilities to:
    1. Track GPU memory usage
    2. Measure timing of different training components
    3. Monitor throughput (samples/second)
    """
    
    def __init__(self, save_dir=None, track_gpu_memory=True):
        """
        Initialize the resource monitor.
        
        Args:
            save_dir: Directory to save visualizations (optional)
            track_gpu_memory: Whether to track GPU memory usage (default: True)
        """
        self.save_dir = save_dir
        self.track_gpu_memory = track_gpu_memory
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Initialize storage
        self.epoch_data = []
        self.component_data = defaultdict(list)
        
        # Current tracking state
        self.current_epoch = None
        self.current_component = None
        self.epoch_start_time = None
        self.component_start_times = {}
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available() and track_gpu_memory
    
    def _get_gpu_memory_usage(self):
        """Get current GPU memory usage in MB."""
        if not self.cuda_available:
            return 0
        
        # Get current device
        device = torch.cuda.current_device()
        
        # Get memory usage in bytes
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        
        # Convert to MB
        memory_allocated_mb = memory_allocated / (1024 * 1024)
        memory_reserved_mb = memory_reserved / (1024 * 1024)
        
        return {
            "allocated": memory_allocated_mb,
            "reserved": memory_reserved_mb
        }
    
    def start_epoch(self, epoch_num):
        """
        Start timing a training epoch.
        
        Args:
            epoch_num: Current epoch number
        """
        if self.current_epoch is not None:
            # End previous epoch timing if not already ended
            self.end_epoch()
        
        self.current_epoch = epoch_num
        self.epoch_start_time = time.time()
        
        # Record initial GPU memory
        if self.cuda_available:
            self.epoch_start_memory = self._get_gpu_memory_usage()
    
    def end_epoch(self, batch_size=None, num_samples=None):
        """
        End timing a training epoch and record metrics.
        
        Args:
            batch_size: Batch size used in the epoch (optional)
            num_samples: Total number of samples processed (optional)
        """
        if self.current_epoch is None or self.epoch_start_time is None:
            return
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.epoch_start_time
        
        # Record final GPU memory
        if self.cuda_available:
            end_memory = self._get_gpu_memory_usage()
            memory_change = {
                "allocated": end_memory["allocated"] - self.epoch_start_memory["allocated"],
                "reserved": end_memory["reserved"] - self.epoch_start_memory["reserved"]
            }
        else:
            end_memory = {"allocated": 0, "reserved": 0}
            memory_change = {"allocated": 0, "reserved": 0}
        
        # Calculate throughput
        throughput = None
        if num_samples is not None and elapsed_time > 0:
            throughput = num_samples / elapsed_time
        
        # Record epoch data
        self.epoch_data.append({
            "epoch": self.current_epoch,
            "time": elapsed_time,
            "memory": end_memory,
            "memory_change": memory_change,
            "batch_size": batch_size,
            "num_samples": num_samples,
            "throughput": throughput
        })
        
        # Reset tracking state
        self.current_epoch = None
        self.epoch_start_time = None
    
    def start_component(self, component_name):
        """
        Start timing a specific component of the training process.
        
        Args:
            component_name: Name of the component (e.g., "forward_pass", "backward_pass")
        """
        self.component_start_times[component_name] = time.time()
        self.current_component = component_name
    
    def end_component(self, component_name=None):
        """
        End timing a specific component and record its duration.
        
        Args:
            component_name: Name of the component (optional, uses current if None)
        """
        if component_name is None:
            component_name = self.current_component
        
        if component_name not in self.component_start_times:
            return
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.component_start_times[component_name]
        
        # Record component data
        self.component_data[component_name].append({
            "epoch": self.current_epoch,
            "time": elapsed_time
        })
        
        # Remove from tracking
        del self.component_start_times[component_name]
        if self.current_component == component_name:
            self.current_component = None
    
    def visualize_memory_usage(self):
        """Visualize GPU memory usage across epochs."""
        if not self.epoch_data or not self.cuda_available:
            return None
        
        # Extract data
        epochs = [data["epoch"] for data in self.epoch_data]
        allocated = [data["memory"]["allocated"] for data in self.epoch_data]
        reserved = [data["memory"]["reserved"] for data in self.epoch_data]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot memory usage
        plt.plot(epochs, allocated, 'o-', label='Allocated Memory')
        plt.plot(epochs, reserved, 's-', label='Reserved Memory')
        
        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Memory Usage (MB)')
        plt.title('GPU Memory Usage During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"memory_usage_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_component_times(self):
        """Visualize time spent in different components across epochs."""
        if not self.component_data:
            return None
        
        # Extract data
        components = list(self.component_data.keys())
        
        # Group data by epoch
        epoch_component_times = defaultdict(dict)
        
        for component, times in self.component_data.items():
            for entry in times:
                epoch = entry["epoch"]
                time_val = entry["time"]
                
                if epoch in epoch_component_times:
                    if component in epoch_component_times[epoch]:
                        epoch_component_times[epoch][component].append(time_val)
                    else:
                        epoch_component_times[epoch][component] = [time_val]
                else:
                    epoch_component_times[epoch] = {component: [time_val]}
        
        # Calculate average time per component per epoch
        epochs = sorted(epoch_component_times.keys())
        component_avg_times = {component: [] for component in components}
        
        for epoch in epochs:
            for component in components:
                if component in epoch_component_times[epoch]:
                    avg_time = np.mean(epoch_component_times[epoch][component])
                    component_avg_times[component].append(avg_time)
                else:
                    component_avg_times[component].append(0)
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 6))
        
        bottom = np.zeros(len(epochs))
        for i, component in enumerate(components):
            values = component_avg_times[component]
            plt.bar(epochs, values, bottom=bottom, label=component)
            bottom += np.array(values)
        
        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Component-wise Training Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"component_times_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_throughput(self):
        """Visualize training throughput (samples/second) across epochs."""
        if not self.epoch_data:
            return None
        
        # Extract data
        epochs = [data["epoch"] for data in self.epoch_data]
        throughputs = [data["throughput"] for data in self.epoch_data if data["throughput"] is not None]
        
        if not throughputs:
            return None
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot throughput
        plt.plot(epochs[:len(throughputs)], throughputs, 'o-')
        
        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Throughput (samples/second)')
        plt.title('Training Throughput')
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"throughput_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_memory_vs_batch_size(self):
        """Visualize the relationship between memory usage and batch size."""
        if not self.epoch_data or not self.cuda_available:
            return None
        
        # Extract data
        batch_sizes = [data["batch_size"] for data in self.epoch_data if data["batch_size"] is not None]
        memory_allocated = [data["memory"]["allocated"] for data in self.epoch_data if data["batch_size"] is not None]
        
        if not batch_sizes:
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(batch_sizes, memory_allocated, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(batch_sizes, memory_allocated, 1)
        p = np.poly1d(z)
        plt.plot(sorted(batch_sizes), p(sorted(batch_sizes)), "r--", alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Batch Size')
        plt.ylabel('GPU Memory Allocated (MB)')
        plt.title('Memory Usage vs. Batch Size')
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"memory_vs_batch_size_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_component_pie_chart(self):
        """Visualize the relative time spent in each component as a pie chart."""
        if not self.component_data:
            return None
        
        # Calculate average time per component across all epochs
        component_total_times = {}
        for component, times in self.component_data.items():
            time_values = [entry["time"] for entry in times]
            component_total_times[component] = np.sum(time_values)
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        
        labels = list(component_total_times.keys())
        sizes = list(component_total_times.values())
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
              shadow=False)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.title('Relative Time Spent in Training Components')
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"component_pie_chart_{timestamp}.png")
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
        
        # Memory usage
        if self.cuda_available:
            memory_path = self.visualize_memory_usage()
            if memory_path:
                results["memory_usage"] = memory_path
            
            memory_batch_path = self.visualize_memory_vs_batch_size()
            if memory_batch_path:
                results["memory_vs_batch_size"] = memory_batch_path
        
        # Component times
        component_times_path = self.visualize_component_times()
        if component_times_path:
            results["component_times"] = component_times_path
        
        # Component pie chart
        pie_chart_path = self.visualize_component_pie_chart()
        if pie_chart_path:
            results["component_pie_chart"] = pie_chart_path
        
        # Throughput
        throughput_path = self.visualize_throughput()
        if throughput_path:
            results["throughput"] = throughput_path
        
        # Save results metadata
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_path = os.path.join(self.save_dir, f"resource_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def save_metrics(self, output_path=None):
        """
        Save resource usage metrics to file.
        
        Args:
            output_path: Path to save the metrics (optional)
        """
        metrics = {
            "epoch_data": self.epoch_data,
            "component_data": {k: v for k, v in self.component_data.items()}
        }
        
        if output_path is None and self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.save_dir, f"resource_metrics_{timestamp}.json")
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(metrics, f, cls=JSONEncoder, indent=2)
                
        return metrics


class JSONEncoder(json.JSONEncoder):
    """Extended JSON encoder that can handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)