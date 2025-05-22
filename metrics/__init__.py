"""
Enhanced metrics tracking for SymCat project.

This package provides comprehensive metrics tracking and visualization tools:
- Class-level performance analysis with head/medium/tail class insights
- Training dynamics tracking (gradients, weights, attention patterns)
- Resource usage monitoring (memory, computation time, throughput)
- Easy integration with the main training pipeline
"""

from .metrics_integration import MetricsManager

__all__ = ['MetricsManager']