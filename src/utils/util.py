"""
Utility functions for CPRFL
"""
import time
import sys
import os
import torch
import json
import numpy as np
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live

# Global console instance
console = Console()

def create_progress_bar():
    """Create a progress bar for training loops"""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

class TrainingProgressBar:
    """
    Premium progress bar for training loops with rich metrics display
    """
    def __init__(self, total_epochs, total_batches, description="Training", metrics=None):
        """
        Initialize training progress bar
        
        Args:
            total_epochs: Total number of epochs for training
            total_batches: Total number of batches per epoch
            description: Task description
            metrics: Dictionary of initial metrics to display (will be updated during training)
        """
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.description = description
        self.metrics = metrics or {}
        self.current_epoch = 0
        self.epoch_start_time = None
        self.training_start_time = time.time()
        
        # Create progress instances
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, style="cyan", complete_style="bright_cyan"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True
        )
        
        self.batch_progress = Progress(
            TextColumn("[bold yellow]{task.description}"),
            BarColumn(bar_width=30, style="yellow", complete_style="bright_yellow"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True
        )
        
        # Tasks
        self.epoch_task = None
        self.batch_task = None
        self.live = None
        
    def start(self):
        """Start the progress display"""
        self.epoch_task = self.progress.add_task(
            f"[cyan]{self.description}", 
            total=self.total_epochs
        )
        
        self.batch_task = self.batch_progress.add_task(
            f"[yellow]Epoch 1/{self.total_epochs}", 
            total=self.total_batches
        )
        
        # Create combined display with progress bars and metrics
        self.live = Live(self._generate_layout(), refresh_per_second=4, console=console)
        self.live.start()
        self.epoch_start_time = time.time()
        
    def update_batch(self, batch_idx, metrics=None):
        """Update batch progress and optionally update metrics"""
        if metrics:
            self.metrics.update(metrics)
        
        self.batch_progress.update(
            self.batch_task, 
            completed=batch_idx + 1,
            description=f"[yellow]Epoch {self.current_epoch + 1}/{self.total_epochs}"
        )
        
        if self.live:
            self.live.update(self._generate_layout())
            
    def update_epoch(self, epoch, metrics=None):
        """Update epoch progress and reset batch progress for next epoch"""
        self.current_epoch = epoch
        
        # Calculate epoch time and ETA
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_start_time = time.time()
        
        # Update metrics with epoch-level information
        if metrics:
            self.metrics.update(metrics)
        self.metrics["epoch_time"] = f"{epoch_time:.2f}s"
        
        # Update progress displays
        self.progress.update(
            self.epoch_task, 
            completed=epoch,
            description=f"[cyan]{self.description}"
        )
        
        # Reset batch progress for next epoch
        self.batch_progress.reset(
            self.batch_task,
            total=self.total_batches,
            description=f"[yellow]Epoch {epoch+1}/{self.total_epochs}"
        )
        
        if self.live:
            self.live.update(self._generate_layout())
            
    def _format_metrics(self):
        """Format metrics for display"""
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        
        # Common metrics to show at the top
        priority_metrics = ["val_map", "val_f1", "test_map", "test_f1", 
                          "loss", "lr", "lambda", "epoch_time"]
        
        # Add priority metrics first (if they exist)
        for key in priority_metrics:
            for metric_key in list(self.metrics.keys()):
                if key.lower() in metric_key.lower():
                    value = self.metrics[metric_key]
                    # Format percentages and small numbers
                    if isinstance(value, float):
                        if abs(value) < 0.01:
                            formatted_value = f"{value:.6f}"
                        elif "map" in metric_key.lower() or "f1" in metric_key.lower():
                            formatted_value = f"{value:.2f}%"
                        else:
                            formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    
                    # Clean up metric names
                    display_name = metric_key.replace("_", " ").title()
                    table.add_row(display_name, formatted_value)
        
        # Add remaining metrics
        for key, value in self.metrics.items():
            if not any(p.lower() in key.lower() for p in priority_metrics):
                # Format values
                if isinstance(value, float):
                    if abs(value) < 0.01:
                        formatted_value = f"{value:.6f}"
                    elif "map" in key.lower() or "f1" in key.lower():
                        formatted_value = f"{value:.2f}%"
                    else:
                        formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                
                # Clean up metric names
                display_name = key.replace("_", " ").title()
                table.add_row(display_name, formatted_value)
                
        return table
            
    def _generate_layout(self):
        """Generate combined layout with progress bars and metrics"""
        grid = Table.grid(expand=True)
        grid.add_column()
        
        # Calculate total elapsed time
        elapsed_time = time.time() - self.training_start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Add header with title and elapsed time
        header = Table.grid(expand=True)
        header.add_column(ratio=5)
        header.add_column(ratio=1, justify="right")
        header.add_row(
            Text("Training Progress", style="bold blue"), 
            Text(f"Total Time: {time_str}", style="bright_black")
        )
        grid.add_row(header)
        
        # Add progress bars
        grid.add_row(self.progress)
        grid.add_row(self.batch_progress)
        
        # Add metrics table if we have metrics
        if self.metrics:
            grid.add_row(
                Panel(
                    self._format_metrics(),
                    title="[bold green]Metrics[/bold green]",
                    border_style="green",
                    padding=(1, 1),
                    expand=True
                )
            )
        
        return grid
    
    def stop(self):
        """Stop the progress display"""
        if self.live:
            self.live.stop()
            
        # Print summary
        total_time = time.time() - self.training_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        summary = Panel(
            f"[bold green]Training Complete![/bold green]\n\n"
            f"[yellow]Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}[/yellow]\n"
            f"[cyan]Epochs: {self.current_epoch}/{self.total_epochs}[/cyan]",
            title="Training Summary",
            border_style="bright_blue",
            padding=(1, 2)
        )
        console.print(summary)


class EvaluationProgressBar:
    """Premium progress bar for evaluation"""
    def __init__(self, total_batches, description="Evaluating"):
        self.total_batches = total_batches
        self.description = description

        self.progress = Progress(
            SpinnerColumn("dots", style="bright_green"),
            TextColumn("[bold green]{task.description}"),
            BarColumn(bar_width=40, style="green", complete_style="bright_green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True
        )

        self.task = None
        self.is_running = False

    def start(self, use_console_print=True):
        """Start the progress display"""
        self.task = self.progress.add_task(f"[green]{self.description}", total=self.total_batches)

        # Instead of starting a live display, print a static progress bar
        if use_console_print:
            console.print(f"\n[bold green]{self.description}...[/bold green]")
            self.is_running = True
        else:
            try:
                self.progress.start()
                self.is_running = True
            except Exception as e:
                console.print(f"[yellow]Warning: Could not start evaluation progress bar: {e}[/yellow]")
                console.print(f"\n[bold green]{self.description}...[/bold green]")
                self.is_running = True

    def update(self, batch_idx):
        """Update progress"""
        if self.is_running:
            self.progress.update(self.task, completed=batch_idx + 1)

    def stop(self):
        """Stop the progress display"""
        if self.is_running:
            try:
                if hasattr(self.progress, '_live') and self.progress._live:
                    self.progress.stop()
            except Exception as e:
                console.print(f"[yellow]Warning: Could not properly stop evaluation progress bar: {e}[/yellow]")
            self.is_running = False


def display_results_table(results, title="Evaluation Results"):
    """Display evaluation results in a premium formatted table"""
    console.print()
    result_table = Table(
        title=title,
        title_style="bold magenta",
        caption="Values in percentages (%)",
        caption_style="dim",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_blue",
        box=None,
        expand=True
    )
    
    # Add columns
    result_table.add_column("Attribute", style="bright_white")
    result_table.add_column("mAP", justify="right", style="cyan")
    result_table.add_column("F1", justify="right", style="green")
    result_table.add_column("Head mAP", justify="right", style="yellow")
    result_table.add_column("Medium mAP", justify="right", style="yellow")
    result_table.add_column("Tail mAP", justify="right", style="yellow")
    result_table.add_column("Accuracy", justify="right", style="magenta")
    
    # Add rows for each attribute
    for attr, metrics in results.items():
        # Extract values, handle None values
        overall_map = f"{metrics.get('overall_mAP', 0):.2f}" if metrics.get('overall_mAP') is not None else "N/A"
        overall_f1 = f"{metrics.get('overall_f1', 0):.2f}" if metrics.get('overall_f1') is not None else "N/A"
        overall_acc = f"{metrics.get('overall_acc', 0):.2f}" if metrics.get('overall_acc') is not None else "N/A"
        
        # Get HMT metrics if available
        type_metrics = metrics.get('type_metrics', {})
        head_map = f"{type_metrics.get('head', {}).get('mAP', 0):.2f}" if type_metrics.get('head', {}).get('mAP') is not None else "N/A"
        medium_map = f"{type_metrics.get('medium', {}).get('mAP', 0):.2f}" if type_metrics.get('medium', {}).get('mAP') is not None else "N/A"
        tail_map = f"{type_metrics.get('tail', {}).get('mAP', 0):.2f}" if type_metrics.get('tail', {}).get('mAP') is not None else "N/A"
        
        result_table.add_row(
            attr.capitalize(),
            overall_map,
            overall_f1,
            head_map,
            medium_map,
            tail_map,
            overall_acc
        )
    
    console.print(result_table)
    console.print()