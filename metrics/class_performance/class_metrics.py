#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Class-level Performance Metrics

This module provides utilities for tracking and analyzing per-class performance metrics,
with special focus on head, medium, and tail class distributions.

Key features:
- Per-class precision, recall, and F1 tracking
- Head/Medium/Tail class performance analysis
- Class distribution visualization

Usage:
    from metrics.class_performance.class_metrics import ClassPerformanceTracker
    
    # Initialize the tracker
    tracker = ClassPerformanceTracker(
        attributes=["crop", "part", "symptomCategories", "symptomTags"],
        label_to_idx_dict=label_to_idx_dict,
        idx_to_label_dict=idx_to_label_dict,
        hmt_info=hmt_info,
        save_dir="visualizations/class_metrics"
    )
    
    # Track metrics during evaluation
    tracker.update(logits_dict, labels_dict)
    
    # Generate visualizations
    tracker.visualize()
    
    # Get metrics as dictionary
    metrics_dict = tracker.get_metrics()
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from datetime import datetime
from pathlib import Path

# Set plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


class ClassPerformanceTracker:
    """
    Tracks and analyzes per-class performance metrics throughout training.
    
    This class provides utilities to:
    1. Track per-class performance metrics
    2. Compare head, medium, and tail class performance
    3. Generate visualizations of class distributions and performance gaps
    """
    
    def __init__(self, attributes, label_to_idx_dict, idx_to_label_dict, hmt_info=None, save_dir=None):
        """
        Initialize the class performance tracker.
        
        Args:
            attributes: List of attributes to track (e.g., ["crop", "part", "symptomCategories", "symptomTags"])
            label_to_idx_dict: Dictionary mapping labels to indices for each attribute
            idx_to_label_dict: Dictionary mapping indices to labels for each attribute
            hmt_info: Information about head/medium/tail class distribution (optional)
            save_dir: Directory to save visualizations (optional)
        """
        self.attributes = attributes
        self.label_to_idx_dict = label_to_idx_dict
        self.idx_to_label_dict = idx_to_label_dict
        self.hmt_info = hmt_info
        self.save_dir = save_dir
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Initialize storage for class metrics
        self.initialize_metrics()
    
    def initialize_metrics(self):
        """Initialize the metrics storage."""
        self.class_metrics = {attr: {} for attr in self.attributes}
        
        # For each attribute, initialize metrics for each class
        for attr in self.attributes:
            if attr not in self.label_to_idx_dict:
                continue
                
            for label_id, idx in self.label_to_idx_dict[attr].items():
                self.class_metrics[attr][label_id] = {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "predictions": [],
                    "ground_truths": [],
                    "type": self._get_class_type(attr, label_id)
                }
    
    def _get_class_type(self, attr, label_id):
        """Determine if a class is head, medium, or tail."""
        if not self.hmt_info or attr not in self.hmt_info:
            return "unknown"
            
        for class_type in ["head", "medium", "tail"]:
            if class_type in self.hmt_info[attr] and label_id in self.hmt_info[attr][class_type]:
                return class_type
                
        return "unknown"
    
    def update(self, logits_dict, labels_dict):
        """
        Update metrics with a new batch of predictions.
        
        Args:
            logits_dict: Dictionary of model logits for each attribute
            labels_dict: Dictionary of ground truth labels for each attribute
        """
        for attr in self.attributes:
            if attr not in logits_dict or attr not in labels_dict:
                continue
                
            # Convert logits to probabilities
            probs = torch.sigmoid(logits_dict[attr]).cpu().numpy()
            labels = labels_dict[attr].cpu().numpy()
            
            # Convert to binary predictions using 0.5 threshold
            preds = (probs > 0.5).astype(np.int32)
            
            # Update metrics for each class
            for label_id, idx in self.label_to_idx_dict[attr].items():
                if idx >= labels.shape[1]:
                    continue
                    
                # Store raw predictions and labels for later PR curve analysis
                self.class_metrics[attr][label_id]["predictions"].extend(probs[:, idx].tolist())
                self.class_metrics[attr][label_id]["ground_truths"].extend(labels[:, idx].tolist())
                
                # Update confusion matrix components
                self.class_metrics[attr][label_id]["true_positives"] += np.sum(
                    (preds[:, idx] == 1) & (labels[:, idx] == 1)
                )
                self.class_metrics[attr][label_id]["false_positives"] += np.sum(
                    (preds[:, idx] == 1) & (labels[:, idx] == 0)
                )
                self.class_metrics[attr][label_id]["false_negatives"] += np.sum(
                    (preds[:, idx] == 0) & (labels[:, idx] == 1)
                )
    
    def compute_metrics(self):
        """Compute precision, recall, and F1 for all classes."""
        for attr in self.attributes:
            for label_id, metrics in self.class_metrics[attr].items():
                tp = metrics["true_positives"]
                fp = metrics["false_positives"]
                fn = metrics["false_negatives"]
                
                # Compute precision, recall, F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Store computed metrics
                self.class_metrics[attr][label_id]["precision"] = precision
                self.class_metrics[attr][label_id]["recall"] = recall
                self.class_metrics[attr][label_id]["f1"] = f1
                
                # Compute AP score if we have enough data
                preds = np.array(metrics["predictions"])
                gts = np.array(metrics["ground_truths"])
                
                if np.sum(gts) > 0:
                    ap = average_precision_score(gts, preds)
                    self.class_metrics[attr][label_id]["ap"] = ap
                else:
                    self.class_metrics[attr][label_id]["ap"] = 0
    
    def get_metrics(self):
        """Get the computed metrics as a dictionary."""
        self.compute_metrics()
        return self.class_metrics
    
    def get_class_type_metrics(self):
        """Aggregate metrics by class type (head, medium, tail)."""
        self.compute_metrics()
        
        type_metrics = {attr: {"head": {}, "medium": {}, "tail": {}, "unknown": {}} 
                       for attr in self.attributes}
        
        for attr in self.attributes:
            # Initialize counters for each type
            for class_type in ["head", "medium", "tail", "unknown"]:
                type_metrics[attr][class_type] = {
                    "f1_sum": 0.0,
                    "precision_sum": 0.0,
                    "recall_sum": 0.0,
                    "ap_sum": 0.0,
                    "count": 0
                }
            
            # Aggregate metrics by class type
            for label_id, metrics in self.class_metrics[attr].items():
                class_type = metrics["type"]
                
                if "f1" in metrics:
                    type_metrics[attr][class_type]["f1_sum"] += metrics["f1"]
                    type_metrics[attr][class_type]["precision_sum"] += metrics["precision"]
                    type_metrics[attr][class_type]["recall_sum"] += metrics["recall"]
                    type_metrics[attr][class_type]["ap_sum"] += metrics["ap"]
                    type_metrics[attr][class_type]["count"] += 1
            
            # Compute averages
            for class_type in ["head", "medium", "tail", "unknown"]:
                count = type_metrics[attr][class_type]["count"]
                if count > 0:
                    type_metrics[attr][class_type]["f1"] = type_metrics[attr][class_type]["f1_sum"] / count
                    type_metrics[attr][class_type]["precision"] = type_metrics[attr][class_type]["precision_sum"] / count
                    type_metrics[attr][class_type]["recall"] = type_metrics[attr][class_type]["recall_sum"] / count
                    type_metrics[attr][class_type]["ap"] = type_metrics[attr][class_type]["ap_sum"] / count
                else:
                    type_metrics[attr][class_type]["f1"] = 0
                    type_metrics[attr][class_type]["precision"] = 0
                    type_metrics[attr][class_type]["recall"] = 0
                    type_metrics[attr][class_type]["ap"] = 0
                
                # Remove the sums
                del type_metrics[attr][class_type]["f1_sum"]
                del type_metrics[attr][class_type]["precision_sum"]
                del type_metrics[attr][class_type]["recall_sum"]
                del type_metrics[attr][class_type]["ap_sum"]
        
        return type_metrics
    
    def visualize_class_distribution(self, attr):
        """
        Visualize the class distribution for an attribute.
        
        Args:
            attr: The attribute to visualize
        """
        if attr not in self.class_metrics:
            return
        
        # Count examples per class
        class_counts = {}
        for label_id, metrics in self.class_metrics[attr].items():
            gt_count = np.sum(metrics["ground_truths"])
            class_counts[label_id] = gt_count
        
        # Sort by count
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        x = range(len(sorted_counts))
        heights = [count for _, count in sorted_counts]
        bars = plt.bar(x, heights)
        
        # Color bars by class type
        for i, (label_id, _) in enumerate(sorted_counts):
            class_type = self.class_metrics[attr][label_id]["type"]
            if class_type == "head":
                bars[i].set_color('red')
            elif class_type == "medium":
                bars[i].set_color('blue')
            elif class_type == "tail":
                bars[i].set_color('green')
            else:
                bars[i].set_color('gray')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Head'),
            Patch(facecolor='blue', label='Medium'),
            Patch(facecolor='green', label='Tail'),
            Patch(facecolor='gray', label='Unknown')
        ]
        plt.legend(handles=legend_elements)
        
        # Add labels and title
        plt.xlabel('Class Index (sorted by frequency)')
        plt.ylabel('Number of Examples')
        plt.title(f'Class Distribution for {attr}')
        plt.yscale('log')  # Log scale to better see the long tail
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"{attr}_class_distribution_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_type_performance(self, attr, metric="f1"):
        """
        Visualize performance by class type for an attribute.
        
        Args:
            attr: The attribute to visualize
            metric: The metric to visualize (f1, precision, recall, ap)
        """
        if attr not in self.class_metrics:
            return
        
        type_metrics = self.get_class_type_metrics()
        if attr not in type_metrics:
            return
        
        # Get values for each class type
        head_val = type_metrics[attr]["head"].get(metric, 0) * 100  # Convert to percentage
        medium_val = type_metrics[attr]["medium"].get(metric, 0) * 100
        tail_val = type_metrics[attr]["tail"].get(metric, 0) * 100
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        x = ['Head', 'Medium', 'Tail']
        heights = [head_val, medium_val, tail_val]
        colors = ['red', 'blue', 'green']
        
        bars = plt.bar(x, heights, color=colors)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Add labels and title
        metric_name = metric.upper() if metric == "ap" else metric.capitalize()
        plt.xlabel('Class Type')
        plt.ylabel(f'{metric_name} Score (%)')
        plt.title(f'{metric_name} by Class Type for {attr}')
        plt.ylim(0, 105)  # Set y-axis limit to 0-105%
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"{attr}_{metric}_by_class_type_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize_precision_recall_curves(self, attr):
        """
        Visualize precision-recall curves by class type.
        
        Args:
            attr: The attribute to visualize
        """
        if attr not in self.class_metrics:
            return
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Track classes by type for the legend
        head_plotted = False
        medium_plotted = False
        tail_plotted = False
        unknown_plotted = False
        
        # Iterate through classes
        for label_id, metrics in self.class_metrics[attr].items():
            # Skip classes with no positive examples
            if np.sum(metrics["ground_truths"]) == 0:
                continue
            
            # Get precision-recall curve
            precision, recall, _ = precision_recall_curve(
                np.array(metrics["ground_truths"]),
                np.array(metrics["predictions"])
            )
            
            # Determine line style and color based on class type
            class_type = metrics["type"]
            label = None
            
            if class_type == "head" and not head_plotted:
                color = 'red'
                label = 'Head Classes'
                head_plotted = True
            elif class_type == "medium" and not medium_plotted:
                color = 'blue'
                label = 'Medium Classes'
                medium_plotted = True
            elif class_type == "tail" and not tail_plotted:
                color = 'green'
                label = 'Tail Classes'
                tail_plotted = True
            elif class_type == "unknown" and not unknown_plotted:
                color = 'gray'
                label = 'Unknown Classes'
                unknown_plotted = True
            else:
                # Classes of the same type share color but don't add to legend
                if class_type == "head":
                    color = 'red'
                elif class_type == "medium":
                    color = 'blue'
                elif class_type == "tail":
                    color = 'green'
                else:
                    color = 'gray'
            
            # Plot the curve with reduced alpha for better visibility
            plt.plot(recall, precision, color=color, alpha=0.3, label=label)
        
        # Average precision-recall curves by class type
        type_curves = {"head": [], "medium": [], "tail": [], "unknown": []}
        
        for label_id, metrics in self.class_metrics[attr].items():
            class_type = metrics["type"]
            
            # Skip classes with no positive examples
            if np.sum(metrics["ground_truths"]) == 0:
                continue
                
            precision, recall, _ = precision_recall_curve(
                np.array(metrics["ground_truths"]),
                np.array(metrics["predictions"])
            )
            
            # Interpolate to a standard recall grid
            if len(recall) > 1:
                std_recall = np.linspace(0, 1, 100)
                # Precision is decreasing with recall, so we need to flip
                interp_precision = np.interp(std_recall, recall[::-1], precision[::-1])
                type_curves[class_type].append((std_recall, interp_precision))
        
        # Plot average curves
        for class_type, curves in type_curves.items():
            if not curves:
                continue
                
            # Average the precision values
            avg_precision = np.zeros(100)
            for _, precision in curves:
                avg_precision += precision
            avg_precision /= len(curves)
            
            # Plot average curve with thicker line
            if class_type == "head":
                color = 'red'
                label = 'Head Average'
            elif class_type == "medium":
                color = 'blue'
                label = 'Medium Average'
            elif class_type == "tail":
                color = 'green'
                label = 'Tail Average'
            else:
                color = 'gray'
                label = 'Unknown Average'
                
            plt.plot(std_recall, avg_precision, color=color, linewidth=2, 
                    label=label)
        
        # Add labels and title
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves by Class Type for {attr}')
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"{attr}_pr_curves_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None
    
    def visualize(self):
        """Generate all visualizations for all attributes."""
        results = {}
        
        for attr in self.attributes:
            attr_results = {}
            
            # Class distribution
            dist_path = self.visualize_class_distribution(attr)
            if dist_path:
                attr_results["class_distribution"] = dist_path
            
            # Performance by class type
            for metric in ["f1", "precision", "recall", "ap"]:
                metric_path = self.visualize_type_performance(attr, metric)
                if metric_path:
                    attr_results[f"{metric}_by_type"] = metric_path
            
            # Precision-recall curves
            pr_path = self.visualize_precision_recall_curves(attr)
            if pr_path:
                attr_results["pr_curves"] = pr_path
                
            results[attr] = attr_results
        
        # Save results metadata
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_path = os.path.join(self.save_dir, f"visualization_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def save_metrics(self, output_path=None):
        """
        Save class performance metrics to file.
        
        Args:
            output_path: Path to save the metrics (optional)
        """
        metrics = {
            "class_metrics": self.get_metrics(),
            "type_metrics": self.get_class_type_metrics()
        }
        
        if output_path is None and self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.save_dir, f"class_performance_metrics_{timestamp}.json")
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        return metrics