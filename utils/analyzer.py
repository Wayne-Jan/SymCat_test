#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Result Analysis and Visualization Script for SymCat Project

This script provides functionality for:
1. Loss tracking during training
2. Standardized result schema
3. Automated report generation with visualizations
4. Hyperparameter tracking and correlation analysis

Usage:
- To track losses during training, use the LossTracker class
- To analyze and visualize results, use the ResultAnalyzer class
- To generate reports, use the ReportGenerator class
"""

import os
import json
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict
import argparse

# Set plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
COLORS = sns.color_palette("muted", 10)

class LossTracker:
    """
    Tracks and saves loss values during model training.

    This class can be integrated into the training loops to:
    - Track loss per epoch (total loss and component losses)
    - Track validation metrics per epoch
    - Save these metrics to a file for later analysis
    """
    
    def __init__(self, save_dir: str = './training_logs', 
                 run_name: Optional[str] = None,
                 log_interval: int = 1):
        """
        Initialize the loss tracker.
        
        Args:
            save_dir: Directory to save loss logs
            run_name: Name of the current training run (if None, timestamp will be used)
            log_interval: How often to log the loss (in epochs)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate a run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"run_{timestamp}"
        else:
            self.run_name = run_name
        
        self.log_interval = log_interval
        self.log_file = self.save_dir / f"{self.run_name}_losses.json"
        
        # Initialize tracking containers
        self.epoch_logs = []
        self.hyperparameters = {}
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "run_name": self.run_name
        }
        
        # Save initial metadata
        self._save_logs()
    
    def add_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Add hyperparameters to track with this run.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.hyperparameters.update(hyperparams)
        self._save_logs()
    
    def log_epoch(self, epoch: int, 
                 losses: Dict[str, float], 
                 val_metrics: Optional[Dict[str, float]] = None,
                 learning_rate: Optional[float] = None) -> None:
        """
        Log metrics for a training epoch.
        
        Args:
            epoch: Current epoch number
            losses: Dictionary of loss values (e.g., {'total': 1.2, 'class': 0.8, 'contrastive': 0.4})
            val_metrics: Optional validation metrics dictionary
            learning_rate: Current learning rate
        """
        if epoch % self.log_interval != 0:
            return
        
        # Create epoch log
        epoch_data = {
            "epoch": epoch,
            "losses": losses,
            "timestamp": datetime.now().isoformat()
        }
        
        if learning_rate is not None:
            epoch_data["learning_rate"] = learning_rate
            
        if val_metrics is not None:
            epoch_data["validation"] = val_metrics
        
        # Add to logs and save
        self.epoch_logs.append(epoch_data)
        self._save_logs()
    
    def finish(self, final_results: Optional[Dict[str, Any]] = None) -> None:
        """
        Finalize the training logs.
        
        Args:
            final_results: Optional dictionary with final evaluation metrics
        """
        self.metadata["end_time"] = datetime.now().isoformat()
        
        if final_results:
            self.metadata["final_results"] = final_results
            
        self._save_logs()
        
    def _save_logs(self) -> None:
        """Save the current logs to a JSON file."""
        data = {
            "metadata": self.metadata,
            "hyperparameters": self.hyperparameters,
            "epoch_logs": self.epoch_logs
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
            

class ResultAnalyzer:
    """
    Analyzes training results, compares different runs, and creates visualizations.
    """
    
    def __init__(self, results_dir: str = './training_logs'):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Directory containing training logs
        """
        self.results_dir = Path(results_dir)
        self.log_files = list(self.results_dir.glob("*_losses.json"))
        self.run_data = {}
        self.load_all_runs()
        
    def load_all_runs(self) -> None:
        """Load all available run data."""
        self.run_data = {}
        
        for log_file in self.log_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                
                run_name = data["metadata"]["run_name"]
                self.run_data[run_name] = data
            except Exception as e:
                print(f"Error loading {log_file}: {str(e)}")
    
    def get_run_names(self) -> List[str]:
        """Get names of all available runs."""
        return list(self.run_data.keys())
    
    def get_run_hyperparams(self, run_name: str) -> Dict[str, Any]:
        """
        Get hyperparameters for a specific run.
        
        Args:
            run_name: Name of the run
            
        Returns:
            Dictionary of hyperparameters
        """
        if run_name not in self.run_data:
            raise ValueError(f"Run '{run_name}' not found")
            
        return self.run_data[run_name]["hyperparameters"]
    
    def get_loss_curves(self, run_names: Union[str, List[str]], 
                      loss_types: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Get loss curves for specified runs.
        
        Args:
            run_names: Name or list of names of runs to analyze
            loss_types: Optional specific loss type(s) to extract (if None, all are returned)
            
        Returns:
            DataFrame with loss data for plotting
        """
        if isinstance(run_names, str):
            run_names = [run_names]
            
        if isinstance(loss_types, str):
            loss_types = [loss_types]
        
        all_data = []
        
        for run_name in run_names:
            if run_name not in self.run_data:
                print(f"Warning: Run '{run_name}' not found, skipping")
                continue
                
            run_data = self.run_data[run_name]
            
            for epoch_log in run_data["epoch_logs"]:
                epoch = epoch_log["epoch"]
                
                # Process specified loss types or all available
                for loss_name, loss_value in epoch_log["losses"].items():
                    if loss_types is None or loss_name in loss_types:
                        all_data.append({
                            "run": run_name,
                            "epoch": epoch,
                            "loss_type": loss_name,
                            "value": loss_value
                        })
        
        return pd.DataFrame(all_data)
    
    def get_metric_curves(self, run_names: Union[str, List[str]], 
                        metrics: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Get validation metric curves for specified runs.
        
        Args:
            run_names: Name or list of names of runs to analyze
            metrics: Optional specific metric(s) to extract (if None, all are returned)
            
        Returns:
            DataFrame with metric data for plotting
        """
        if isinstance(run_names, str):
            run_names = [run_names]
            
        if isinstance(metrics, str):
            metrics = [metrics]
        
        all_data = []
        
        for run_name in run_names:
            if run_name not in self.run_data:
                print(f"Warning: Run '{run_name}' not found, skipping")
                continue
                
            run_data = self.run_data[run_name]
            
            for epoch_log in run_data["epoch_logs"]:
                epoch = epoch_log["epoch"]
                
                # Skip if no validation data
                if "validation" not in epoch_log:
                    continue
                
                # Process specified metrics or all available
                for metric_name, metric_value in epoch_log["validation"].items():
                    if metrics is None or metric_name in metrics:
                        all_data.append({
                            "run": run_name,
                            "epoch": epoch,
                            "metric": metric_name,
                            "value": metric_value
                        })
        
        return pd.DataFrame(all_data)
    
    def plot_loss_curves(self, run_names: Union[str, List[str]], 
                       loss_types: Optional[Union[str, List[str]]] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot loss curves for specified runs.
        
        Args:
            run_names: Name or list of names of runs to analyze
            loss_types: Optional specific loss type(s) to plot
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        df = self.get_loss_curves(run_names, loss_types)
        
        if df.empty:
            print("No loss data found for the specified runs and loss types")
            return plt.figure()
        
        # Determine layout based on number of loss types
        loss_types_in_data = df["loss_type"].unique()
        num_plots = len(loss_types_in_data)
        
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            ncols = min(2, num_plots)
            nrows = (num_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                                     sharex=True, squeeze=False)
            axes = axes.flatten()
        
        # Plot each loss type
        for i, loss_type in enumerate(loss_types_in_data):
            if i < len(axes):
                ax = axes[i]
                subset = df[df["loss_type"] == loss_type]
                
                for j, run in enumerate(subset["run"].unique()):
                    run_data = subset[subset["run"] == run]
                    ax.plot(run_data["epoch"], run_data["value"], 
                           marker='o', markersize=4, label=run, color=COLORS[j % len(COLORS)])
                
                ax.set_title(f"{loss_type} Loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss Value")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_metric_curves(self, run_names: Union[str, List[str]], 
                         metrics: Optional[Union[str, List[str]]] = None,
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot validation metric curves for specified runs.
        
        Args:
            run_names: Name or list of names of runs to analyze
            metrics: Optional specific metric(s) to plot
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        df = self.get_metric_curves(run_names, metrics)
        
        if df.empty:
            print("No metric data found for the specified runs and metrics")
            return plt.figure()
        
        # Determine layout based on number of metrics
        metrics_in_data = df["metric"].unique()
        num_plots = len(metrics_in_data)
        
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            ncols = min(2, num_plots)
            nrows = (num_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                                     sharex=True, squeeze=False)
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics_in_data):
            if i < len(axes):
                ax = axes[i]
                subset = df[df["metric"] == metric]
                
                for j, run in enumerate(subset["run"].unique()):
                    run_data = subset[subset["run"] == run]
                    ax.plot(run_data["epoch"], run_data["value"], 
                           marker='o', markersize=4, label=run, color=COLORS[j % len(COLORS)])
                
                ax.set_title(f"{metric} Metric")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def compare_hyperparameters(self, run_names: Union[str, List[str]], 
                             target_metric: Optional[str] = None) -> pd.DataFrame:
        """
        Compare hyperparameters across runs with optional correlation to a target metric.
        
        Args:
            run_names: Name or list of names of runs to analyze
            target_metric: Optional metric to correlate hyperparameters with
            
        Returns:
            DataFrame with hyperparameter comparison
        """
        if isinstance(run_names, str):
            run_names = [run_names]
            
        all_params = {}
        result_values = {}
        
        for run_name in run_names:
            if run_name not in self.run_data:
                print(f"Warning: Run '{run_name}' not found, skipping")
                continue
                
            # Get hyperparameters
            run_params = self.get_run_hyperparams(run_name)
            all_params[run_name] = run_params
            
            # Get target metric if specified
            if target_metric and "metadata" in self.run_data[run_name]:
                metadata = self.run_data[run_name]["metadata"]
                if "final_results" in metadata and target_metric in metadata["final_results"]:
                    result_values[run_name] = metadata["final_results"][target_metric]
        
        # Create DataFrame
        param_df = pd.DataFrame(all_params).T
        
        if target_metric and result_values:
            param_df[target_metric] = pd.Series(result_values)
            
        return param_df
    
    def plot_hyperparam_correlation(self, param_name: str, 
                                  metric_name: str,
                                  run_names: Optional[List[str]] = None,
                                  figsize: Tuple[int, int] = (10, 6),
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation between a hyperparameter and a metric.
        
        Args:
            param_name: Name of the hyperparameter
            metric_name: Name of the metric
            run_names: Optional list of run names to include (default: all runs)
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if run_names is None:
            run_names = self.get_run_names()
            
        data = []
        
        for run_name in run_names:
            if run_name not in self.run_data:
                continue
                
            run_data = self.run_data[run_name]
            
            # Get hyperparameter value
            if param_name not in run_data["hyperparameters"]:
                continue
            param_value = run_data["hyperparameters"][param_name]
            
            # Get metric value
            metric_value = None
            if "metadata" in run_data and "final_results" in run_data["metadata"]:
                if metric_name in run_data["metadata"]["final_results"]:
                    metric_value = run_data["metadata"]["final_results"][metric_name]
            
            if metric_value is not None:
                data.append({
                    "run": run_name,
                    "param_value": param_value,
                    "metric_value": metric_value
                })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            print(f"No data found for parameter '{param_name}' and metric '{metric_name}'")
            return plt.figure()
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Check parameter type and create appropriate plot
        if isinstance(df["param_value"].iloc[0], (int, float, np.number)):
            # Numeric parameter - scatter plot with trend line
            sns.regplot(x="param_value", y="metric_value", data=df, ax=ax, 
                       scatter_kws={"s": 80, "alpha": 0.7}, line_kws={"color": "red"})
            
            # Add run names as annotations
            for i, row in df.iterrows():
                ax.annotate(row["run"], (row["param_value"], row["metric_value"]), 
                           fontsize=8, alpha=0.7, ha='center', va='bottom')
        else:
            # Categorical parameter - bar plot
            sns.barplot(x="param_value", y="metric_value", data=df, ax=ax)
            
            # Add run names as annotations
            for i, bar in enumerate(ax.patches):
                if i < len(df):
                    ax.annotate(df.iloc[i]["run"], 
                               (bar.get_x() + bar.get_width()/2, bar.get_height()), 
                               fontsize=8, ha='center', va='bottom')
        
        ax.set_title(f"Correlation: {param_name} vs {metric_name}")
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class ReportGenerator:
    """
    Generates comprehensive reports from training results.
    """
    
    def __init__(self, analyzer: ResultAnalyzer, 
                output_dir: str = './reports'):
        """
        Initialize the report generator.
        
        Args:
            analyzer: ResultAnalyzer instance to use for data
            output_dir: Directory to save reports
        """
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_single_run_report(self, run_name: str, 
                                 output_format: str = 'html') -> str:
        """
        Generate a report for a single run.
        
        Args:
            run_name: Name of the run to report on
            output_format: Output format ('html', 'pdf', or 'md')
            
        Returns:
            Path to the generated report
        """
        if run_name not in self.analyzer.run_data:
            raise ValueError(f"Run '{run_name}' not found")
            
        run_data = self.analyzer.run_data[run_name]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{run_name}_report_{timestamp}"
        
        # Create output path based on format
        if output_format == 'html':
            output_path = self.output_dir / f"{report_name}.html"
            self._generate_html_report(run_name, output_path)
        elif output_format == 'pdf':
            output_path = self.output_dir / f"{report_name}.pdf"
            self._generate_pdf_report(run_name, output_path)
        elif output_format == 'md':
            output_path = self.output_dir / f"{report_name}.md"
            self._generate_md_report(run_name, output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        return str(output_path)
    
    def generate_comparison_report(self, run_names: List[str], 
                                 metrics: List[str] = None,
                                 output_format: str = 'html') -> str:
        """
        Generate a comparison report for multiple runs.
        
        Args:
            run_names: List of run names to compare
            metrics: List of metrics to include in comparison
            output_format: Output format ('html', 'pdf', or 'md')
            
        Returns:
            Path to the generated report
        """
        # Validate run names
        valid_runs = []
        for run in run_names:
            if run in self.analyzer.run_data:
                valid_runs.append(run)
            else:
                print(f"Warning: Run '{run}' not found, skipping")
        
        if not valid_runs:
            raise ValueError("No valid runs found for comparison")
            
        # Generate comparison name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_name = f"comparison_{'_vs_'.join(valid_runs[:3])}"
        if len(valid_runs) > 3:
            comparison_name += f"_and_{len(valid_runs)-3}_more"
        comparison_name += f"_{timestamp}"
        
        # Create output path based on format
        if output_format == 'html':
            output_path = self.output_dir / f"{comparison_name}.html"
            self._generate_html_comparison(valid_runs, metrics, output_path)
        elif output_format == 'pdf':
            output_path = self.output_dir / f"{comparison_name}.pdf"
            self._generate_pdf_comparison(valid_runs, metrics, output_path)
        elif output_format == 'md':
            output_path = self.output_dir / f"{comparison_name}.md"
            self._generate_md_comparison(valid_runs, metrics, output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        return str(output_path)
    
    def _generate_html_report(self, run_name: str, output_path: Path) -> None:
        """Generate HTML report for a single run."""
        run_data = self.analyzer.run_data[run_name]
        
        # Create images directory
        img_dir = output_path.parent / f"{output_path.stem}_images"
        img_dir.mkdir(exist_ok=True)
        
        # Generate plots
        loss_plot_path = img_dir / "loss_curves.png"
        metric_plot_path = img_dir / "metric_curves.png"
        
        self.analyzer.plot_loss_curves(run_name, save_path=str(loss_plot_path))
        self.analyzer.plot_metric_curves(run_name, save_path=str(metric_plot_path))
        
        # Build HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Report: {run_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .plot {{ margin-top: 20px; text-align: center; }}
                .plot img {{ max-width: 100%; }}
                .metadata {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Training Report: {run_name}</h1>
                
                <div class="metadata">
                    <h2>Metadata</h2>
                    <table>
                        <tr><th>Field</th><th>Value</th></tr>
        """
        
        # Add metadata
        for key, value in run_data["metadata"].items():
            if key != "final_results":
                html_content += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html_content += """
                    </table>
                </div>
                
                <div class="hyperparameters">
                    <h2>Hyperparameters</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
        """
        
        # Add hyperparameters
        for key, value in run_data["hyperparameters"].items():
            html_content += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
            
        html_content += """
                    </table>
                </div>
                
                <div class="training-results">
                    <h2>Training Results</h2>
                </div>
        """
        
        # Add plots
        if loss_plot_path.exists():
            html_content += f"""
                <div class="plot">
                    <h3>Loss Curves</h3>
                    <img src="{loss_plot_path.name}" alt="Loss Curves">
                </div>
            """
            
        if metric_plot_path.exists():
            html_content += f"""
                <div class="plot">
                    <h3>Validation Metrics</h3>
                    <img src="{metric_plot_path.name}" alt="Validation Metrics">
                </div>
            """
            
        # Add final results if available
        if "final_results" in run_data["metadata"]:
            html_content += """
                <div class="final-results">
                    <h2>Final Evaluation Results</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
            """
            
            for key, value in run_data["metadata"]["final_results"].items():
                html_content += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
                
            html_content += """
                    </table>
                </div>
            """
            
        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_html_comparison(self, run_names: List[str], 
                                metrics: List[str], 
                                output_path: Path) -> None:
        """Generate HTML comparison report for multiple runs."""
        # Create images directory
        img_dir = output_path.parent / f"{output_path.stem}_images"
        img_dir.mkdir(exist_ok=True)
        
        # Generate plots
        loss_plot_path = img_dir / "loss_comparison.png"
        metric_plot_path = img_dir / "metric_comparison.png"
        
        self.analyzer.plot_loss_curves(run_names, save_path=str(loss_plot_path))
        self.analyzer.plot_metric_curves(run_names, metrics=metrics, 
                                        save_path=str(metric_plot_path))
        
        # Get hyperparameter comparison
        if metrics:
            primary_metric = metrics[0]
            hyperparam_df = self.analyzer.compare_hyperparameters(run_names, primary_metric)
        else:
            hyperparam_df = self.analyzer.compare_hyperparameters(run_names)
        
        # Build HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .plot {{ margin-top: 20px; text-align: center; }}
                .plot img {{ max-width: 100%; }}
                .comparison {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Training Comparison Report</h1>
                <p>Comparing runs: {', '.join(run_names)}</p>
                
                <div class="comparison">
                    <h2>Hyperparameter Comparison</h2>
        """
        
        # Add hyperparameter table
        html_content += hyperparam_df.to_html()
        
        # Add plots
        if loss_plot_path.exists():
            html_content += f"""
                <div class="plot">
                    <h2>Loss Comparison</h2>
                    <img src="{loss_plot_path.name}" alt="Loss Comparison">
                </div>
            """
            
        if metric_plot_path.exists():
            html_content += f"""
                <div class="plot">
                    <h2>Validation Metrics Comparison</h2>
                    <img src="{metric_plot_path.name}" alt="Validation Metrics Comparison">
                </div>
            """
            
        # Add final results comparison
        html_content += """
            <div class="final-results">
                <h2>Final Results Comparison</h2>
                <table>
                    <tr><th>Run</th>
        """
        
        # Get all possible metrics
        all_metrics = set()
        for run_name in run_names:
            if run_name in self.analyzer.run_data:
                run_data = self.analyzer.run_data[run_name]
                if "metadata" in run_data and "final_results" in run_data["metadata"]:
                    all_metrics.update(run_data["metadata"]["final_results"].keys())
        
        # Filter metrics if specified
        if metrics:
            display_metrics = [m for m in metrics if m in all_metrics]
        else:
            display_metrics = sorted(all_metrics)
        
        # Add metric headers
        for metric in display_metrics:
            html_content += f"<th>{metric}</th>"
        
        html_content += "</tr>\n"
        
        # Add data rows
        for run_name in run_names:
            html_content += f"<tr><td>{run_name}</td>"
            
            if run_name in self.analyzer.run_data:
                run_data = self.analyzer.run_data[run_name]
                if "metadata" in run_data and "final_results" in run_data["metadata"]:
                    final_results = run_data["metadata"]["final_results"]
                    
                    for metric in display_metrics:
                        if metric in final_results:
                            html_content += f"<td>{final_results[metric]}</td>"
                        else:
                            html_content += "<td>-</td>"
                else:
                    for _ in display_metrics:
                        html_content += "<td>-</td>"
            else:
                for _ in display_metrics:
                    html_content += "<td>-</td>"
                    
            html_content += "</tr>\n"
        
        html_content += """
                </table>
            </div>
        """
            
        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_md_report(self, run_name: str, output_path: Path) -> None:
        """Generate Markdown report for a single run."""
        run_data = self.analyzer.run_data[run_name]
        
        # Create images directory
        img_dir = output_path.parent / f"{output_path.stem}_images"
        img_dir.mkdir(exist_ok=True)
        
        # Generate plots
        loss_plot_path = img_dir / "loss_curves.png"
        metric_plot_path = img_dir / "metric_curves.png"
        
        self.analyzer.plot_loss_curves(run_name, save_path=str(loss_plot_path))
        self.analyzer.plot_metric_curves(run_name, save_path=str(metric_plot_path))
        
        # Build markdown content
        md_content = f"""# Training Report: {run_name}

## Metadata

"""
        
        # Add metadata
        for key, value in run_data["metadata"].items():
            if key != "final_results":
                md_content += f"- **{key}**: {value}\n"
        
        md_content += "\n## Hyperparameters\n\n"
        
        # Add hyperparameters
        for key, value in run_data["hyperparameters"].items():
            md_content += f"- **{key}**: {value}\n"
            
        md_content += "\n## Training Results\n\n"
        
        # Add plots
        if loss_plot_path.exists():
            md_content += f"### Loss Curves\n\n![Loss Curves]({loss_plot_path.name})\n\n"
            
        if metric_plot_path.exists():
            md_content += f"### Validation Metrics\n\n![Validation Metrics]({metric_plot_path.name})\n\n"
            
        # Add final results if available
        if "final_results" in run_data["metadata"]:
            md_content += "## Final Evaluation Results\n\n"
            
            for key, value in run_data["metadata"]["final_results"].items():
                md_content += f"- **{key}**: {value}\n"
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(md_content)
    
    def _generate_md_comparison(self, run_names: List[str], 
                              metrics: List[str], 
                              output_path: Path) -> None:
        """Generate Markdown comparison report for multiple runs."""
        # Create images directory
        img_dir = output_path.parent / f"{output_path.stem}_images"
        img_dir.mkdir(exist_ok=True)
        
        # Generate plots
        loss_plot_path = img_dir / "loss_comparison.png"
        metric_plot_path = img_dir / "metric_comparison.png"
        
        self.analyzer.plot_loss_curves(run_names, save_path=str(loss_plot_path))
        self.analyzer.plot_metric_curves(run_names, metrics=metrics, 
                                        save_path=str(metric_plot_path))
        
        # Get hyperparameter comparison
        if metrics:
            primary_metric = metrics[0]
            hyperparam_df = self.analyzer.compare_hyperparameters(run_names, primary_metric)
        else:
            hyperparam_df = self.analyzer.compare_hyperparameters(run_names)
        
        # Build markdown content
        md_content = f"""# Training Comparison Report

Comparing runs: {', '.join(run_names)}

## Hyperparameter Comparison

{hyperparam_df.to_markdown()}

"""
        
        # Add plots
        if loss_plot_path.exists():
            md_content += f"## Loss Comparison\n\n![Loss Comparison]({loss_plot_path.name})\n\n"
            
        if metric_plot_path.exists():
            md_content += f"## Validation Metrics Comparison\n\n![Validation Metrics Comparison]({metric_plot_path.name})\n\n"
            
        # Add final results comparison
        md_content += "## Final Results Comparison\n\n"
        
        # Get all possible metrics
        all_metrics = set()
        for run_name in run_names:
            if run_name in self.analyzer.run_data:
                run_data = self.analyzer.run_data[run_name]
                if "metadata" in run_data and "final_results" in run_data["metadata"]:
                    all_metrics.update(run_data["metadata"]["final_results"].keys())
        
        # Filter metrics if specified
        if metrics:
            display_metrics = [m for m in metrics if m in all_metrics]
        else:
            display_metrics = sorted(all_metrics)
        
        # Create table header
        md_content += "| Run |"
        for metric in display_metrics:
            md_content += f" {metric} |"
        md_content += "\n|-----|"
        for _ in display_metrics:
            md_content += "------|"
        md_content += "\n"
        
        # Add data rows
        for run_name in run_names:
            md_content += f"| {run_name} |"
            
            if run_name in self.analyzer.run_data:
                run_data = self.analyzer.run_data[run_name]
                if "metadata" in run_data and "final_results" in run_data["metadata"]:
                    final_results = run_data["metadata"]["final_results"]
                    
                    for metric in display_metrics:
                        if metric in final_results:
                            md_content += f" {final_results[metric]} |"
                        else:
                            md_content += " - |"
                else:
                    for _ in display_metrics:
                        md_content += " - |"
            else:
                for _ in display_metrics:
                    md_content += " - |"
                    
            md_content += "\n"
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(md_content)
    
    # PDF report generation methods are placeholders and would require additional libraries
    def _generate_pdf_report(self, run_name: str, output_path: Path) -> None:
        """Generate PDF report for a single run (placeholder)."""
        # Generate HTML first, then convert to PDF
        # This would require a library like pdfkit or reportlab
        temp_html = self.output_dir / f"{output_path.stem}_temp.html"
        self._generate_html_report(run_name, temp_html)
        
        # TODO: Implement PDF conversion
        print(f"PDF generation not implemented. HTML report saved at {temp_html}")
    
    def _generate_pdf_comparison(self, run_names: List[str], 
                               metrics: List[str], 
                               output_path: Path) -> None:
        """Generate PDF comparison report (placeholder)."""
        # Generate HTML first, then convert to PDF
        temp_html = self.output_dir / f"{output_path.stem}_temp.html"
        self._generate_html_comparison(run_names, metrics, temp_html)
        
        # TODO: Implement PDF conversion
        print(f"PDF generation not implemented. HTML report saved at {temp_html}")


def main():
    """Command-line interface for the result analyzer."""
    parser = argparse.ArgumentParser(description="Analyze and visualize training results")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze training results")
    analyze_parser.add_argument("--logs-dir", default="./training_logs", 
                              help="Directory containing training logs")
    analyze_parser.add_argument("--runs", nargs="+", help="Run names to analyze")
    analyze_parser.add_argument("--loss-types", nargs="+", help="Loss types to analyze")
    analyze_parser.add_argument("--metrics", nargs="+", help="Metrics to analyze")
    analyze_parser.add_argument("--output-dir", default="./reports", 
                              help="Directory to save reports")
    analyze_parser.add_argument("--format", choices=["html", "pdf", "md"], default="html",
                              help="Report output format")
    analyze_parser.add_argument("--compare", action="store_true", 
                              help="Generate comparison report instead of individual reports")
    
    # Generate report command
    report_parser = subparsers.add_parser("report", help="Generate report from existing logs")
    report_parser.add_argument("--logs-dir", default="./training_logs", 
                             help="Directory containing training logs")
    report_parser.add_argument("--runs", nargs="+", required=True, help="Run names to include in report")
    report_parser.add_argument("--metrics", nargs="+", help="Metrics to include in report")
    report_parser.add_argument("--output-dir", default="./reports", 
                             help="Directory to save reports")
    report_parser.add_argument("--format", choices=["html", "pdf", "md"], default="html",
                             help="Report output format")
    
    # Track command
    track_parser = subparsers.add_parser("track", 
                                       help="Create a new LossTracker for a training run")
    track_parser.add_argument("--run-name", required=True, help="Name for the training run")
    track_parser.add_argument("--save-dir", default="./training_logs", 
                            help="Directory to save training logs")
    track_parser.add_argument("--log-interval", type=int, default=1, 
                            help="Interval (in epochs) at which to log losses")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyzer = ResultAnalyzer(args.logs_dir)
        
        if not args.runs:
            args.runs = analyzer.get_run_names()
            
        if args.compare:
            # Generate comparison report
            report_generator = ReportGenerator(analyzer, args.output_dir)
            report_path = report_generator.generate_comparison_report(
                args.runs, args.metrics, args.format)
            print(f"Comparison report generated: {report_path}")
        else:
            # Generate individual reports
            report_generator = ReportGenerator(analyzer, args.output_dir)
            for run in args.runs:
                try:
                    report_path = report_generator.generate_single_run_report(
                        run, args.format)
                    print(f"Report generated for {run}: {report_path}")
                except ValueError as e:
                    print(f"Error generating report for {run}: {str(e)}")
    
    elif args.command == "report":
        analyzer = ResultAnalyzer(args.logs_dir)
        report_generator = ReportGenerator(analyzer, args.output_dir)
        
        if len(args.runs) == 1:
            # Single report
            try:
                report_path = report_generator.generate_single_run_report(
                    args.runs[0], args.format)
                print(f"Report generated: {report_path}")
            except ValueError as e:
                print(f"Error generating report: {str(e)}")
        else:
            # Comparison report
            try:
                report_path = report_generator.generate_comparison_report(
                    args.runs, args.metrics, args.format)
                print(f"Comparison report generated: {report_path}")
            except ValueError as e:
                print(f"Error generating comparison report: {str(e)}")
    
    elif args.command == "track":
        tracker = LossTracker(args.save_dir, args.run_name, args.log_interval)
        print(f"Loss tracker created for run '{args.run_name}'")
        print(f"Log file: {tracker.log_file}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()