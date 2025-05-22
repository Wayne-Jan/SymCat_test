# Performance Metrics Tracking

This directory contains modules for tracking and analyzing additional performance metrics during model training and evaluation.

## Directory Structure

- **class_performance/**: Metrics related to per-class performance and distribution analysis
- **training_dynamics/**: Tracking of gradient flow, weight evolution, and attention mechanisms
- **resource_usage/**: Memory usage and computation time measurements

## Usage

Metrics collected here are integrated with the main training pipeline and results are saved in the corresponding directories under `visualizations/`.

## Integration

These metrics are collected during model training in `train_eval.py` and `train_baseline.py` and can be visualized using the associated utilities in each module.