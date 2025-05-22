# Enhanced Metrics Tracking System

This document provides an overview of the enhanced metrics tracking system for the SymCat project. The system offers comprehensive insights into model performance, training dynamics, and resource utilization.

## Overview

The metrics system consists of three main components:

1. **Class Performance Analysis**
   - Per-class precision, recall, and F1 tracking
   - Head/Medium/Tail class performance comparison
   - Class distribution visualization
   - Precision-recall curves by class type

2. **Training Dynamics**
   - Gradient flow visualization
   - Weight distribution tracking
   - Vanishing/exploding gradient detection
   - VSI attention visualization

3. **Resource Usage**
   - GPU memory tracking
   - Component-wise timing analysis
   - Training throughput monitoring
   - Memory vs. batch size analysis

## Directory Structure

```
metrics/
├── class_performance/    # Per-class metrics tracking
├── training_dynamics/    # Gradient and attention tracking
├── resource_usage/       # Memory and timing monitoring
├── metrics_integration.py # Unified integration module
```

Visualizations are saved to:

```
visualizations/
├── class_metrics/        # Class performance visualizations
├── training_dynamics/    # Gradient and attention visualizations
├── resource_usage/       # Resource usage visualizations
```

## Integration Guide

### Quick Start

The easiest way to use the metrics system is through the `MetricsManager` class, which provides a unified interface to all metrics tools:

```python
from metrics import MetricsManager

# Initialize the manager
metrics_manager = MetricsManager(
    model=model,
    attributes=attributes,
    label_to_idx_dict=label_to_idx_dict,
    idx_to_label_dict=idx_to_label_dict,
    hmt_info=hmt_info,
    save_dir="visualizations"
)

# Start tracking (typically in train_multi_cprfl or train_baseline_classifier)
for epoch in range(1, epochs + 1):
    # Start tracking epoch
    metrics_manager.start_epoch(epoch)
    
    # Training loop
    for batch_idx, batch in enumerate(train_dataloader):
        # Update batch info
        metrics_manager.update_batch(batch_idx)
        
        # Component timing (optional)
        metrics_manager.start_component("forward_pass")
        # Forward pass
        metrics_manager.end_component("forward_pass")
        
        metrics_manager.start_component("backward_pass")
        # Backward pass
        metrics_manager.end_component("backward_pass")
    
    # Update with validation results
    metrics_manager.update_epoch_results(
        logits_dict, 
        labels_dict, 
        batch_size=batch_size,
        num_samples=len(train_dataset)
    )

# At the end of training
metrics_manager.finalize()

# Get all metrics data
metrics_data = metrics_manager.get_metrics()
```

### Individual Components

You can also use the individual components directly:

#### Class Performance Tracking

```python
from metrics.class_performance import ClassPerformanceTracker

tracker = ClassPerformanceTracker(
    attributes=attributes,
    label_to_idx_dict=label_to_idx_dict,
    idx_to_label_dict=idx_to_label_dict,
    hmt_info=hmt_info,
    save_dir="visualizations/class_metrics"
)

# Update with new predictions
tracker.update(logits_dict, labels_dict)

# Generate visualizations
tracker.visualize()

# Get metrics
metrics = tracker.get_metrics()
```

#### Gradient Tracking

```python
from metrics.training_dynamics import GradientTracker

tracker = GradientTracker(
    model=model,
    save_dir="visualizations/training_dynamics"
)

# Register hooks
tracker.register_hooks()

# Forward and backward pass happens here
# ...

# Update with current epoch
tracker.update(epoch=epoch_num)

# Visualize
tracker.visualize()

# Remove hooks when done
tracker.remove_hooks()
```

#### Attention Visualization

```python
from metrics.training_dynamics import AttentionVisualizer

visualizer = AttentionVisualizer(
    model=model,
    save_dir="visualizations/training_dynamics"
)

# Register hooks
visualizer.register_hooks()

# Forward pass happens here
# ...

# Update with current epoch
visualizer.update(epoch=epoch_num)

# Visualize
visualizer.visualize_attention()

# Remove hooks when done
visualizer.remove_hooks()
```

#### Resource Monitoring

```python
from metrics.resource_usage import ResourceMonitor

monitor = ResourceMonitor(
    save_dir="visualizations/resource_usage"
)

# Start timing an epoch
monitor.start_epoch(epoch_num)

# Start timing a component
monitor.start_component("forward_pass")
# ... component operations ...
monitor.end_component("forward_pass")

# End epoch and record metrics
monitor.end_epoch(batch_size=batch_size, num_samples=num_samples)

# Visualize
monitor.visualize()
```

## Integration with main.py

To integrate with `main.py`, simply add the `MetricsManager` at the beginning of the training function and pass it to the training functions:

```python
# In train_multi_cprfl or train_baseline_classifier
def train_with_metrics(model, metrics_manager=None, ...):
    # Check if metrics tracking is enabled
    if metrics_manager is not None:
        metrics_manager.start_tracking()
        
    # Regular training code with metrics tracking
    for epoch in range(1, epochs + 1):
        if metrics_manager is not None:
            metrics_manager.start_epoch(epoch)
            
        # ... existing code ...
        
        if metrics_manager is not None:
            metrics_manager.update_epoch_results(logits_dict, labels_dict)
    
    # End tracking
    if metrics_manager is not None:
        metrics_manager.finalize()
        
    return model
```

## Visualization Examples

### Class Performance

- **Class Distribution**
  - Shows the distribution of classes by frequency
  - Colored by head (red), medium (blue), and tail (green) classes
  
- **Performance by Class Type**
  - Bar charts comparing F1, precision, recall and AP for head/medium/tail classes
  - Clearly shows the performance gap between common and rare classes
  
- **Precision-Recall Curves**
  - PR curves grouped by class type
  - Average curves for each class type

### Training Dynamics

- **Gradient Flow**
  - Line chart showing gradient norms across epochs
  - Helps identify vanishing or exploding gradients
  
- **Weight Distribution**
  - Histograms showing weight distribution changes during training
  - Shown for start, middle, and end of training
  
- **Attention Maps**
  - Heatmaps of VSI attention patterns
  - Multiple attention heads visualization
  - Attention evolution over time

### Resource Usage

- **Memory Usage**
  - Line chart showing GPU memory usage over time
  - Memory vs. batch size correlation
  
- **Component Times**
  - Stacked bar chart showing time spent in different components
  - Pie chart of relative time distribution
  
- **Throughput**
  - Line chart showing samples/second over epochs

## Data Export

All metrics data can be exported to JSON format using the `metrics_manager.get_metrics()` method. The resulting JSON file can be loaded into other tools for further analysis.

## Requirements

The metrics system requires the following Python packages:

- numpy
- matplotlib
- seaborn
- torch
- scikit-learn

## Performance Considerations

The metrics system is designed to have minimal impact on training performance:

- Class performance tracking has minimal overhead, primarily during validation
- Gradient and attention tracking use PyTorch hooks which add some overhead
- Resource monitoring uses lightweight timing mechanisms

For production training, you may want to disable some of the more expensive metrics (e.g., gradient tracking) to maximize performance.

## Metrics Analysis

For detailed analysis of the metrics, you can use the following tools:

1. **Visual Inspection**: View the generated PNG files in the visualizations directory
2. **JSON Analysis**: Load and analyze the JSON output files
3. **Custom Reports**: Use the metrics data to generate custom reports

## Troubleshooting

- **Memory Issues**: If you encounter memory issues, try disabling the weight distribution tracking by setting `track_weights=False` in the GradientTracker
- **Missing Visualizations**: Ensure that the save directory exists and is writable
- **Incorrect Metrics**: Verify that the model outputs and ground truth labels are correctly formatted