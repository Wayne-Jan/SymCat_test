#!/usr/bin/env python3
import json
import sys
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# Files to analyze
files = [
    "results/baseline_bce_e150_r3_results.json",
    "results/baseline_asl_e150_r3_results.json",
    "results/zero_shot_clip_results.json",
    # Assuming this is a multi-modal model result file with a specific fusion mode
    # If you want to compare different multi-modal configurations, add more files here
    "results/ablation_VSI_e150_r3_results.json"
]

# Categories to report on
categories = ['crop', 'part', 'symptomCategories', 'symptomTags']
# Metrics to report
metrics = ['overall_mAP', 'overall_f1']

model_summaries = {}

# Process each result file
for file_path in files:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if 'baseline_bce' in file_path:
            model_name = "Option 1: Baseline (BCE)"
        elif 'baseline_asl' in file_path:
            model_name = "Option 1: Baseline (ASL)"
        elif 'zero_shot' in file_path:
            model_name = "Option 3: Zero-Shot CLIP"
        elif 'ablation_VSI' in file_path:
            model_name = "Option 2: Multi-modal models"
            # This file contains multiple models, mark that we need special handling
            model_summaries[model_name] = {"special": "vsi_ablation"}
            continue
        else:
            model_name = file_path.split('/')[-1].split('_results')[0]
        
        model_summaries[model_name] = {}
        
        # Get the first setting in the file
        setting_name = list(data.keys())[0]
        setting = data[setting_name]
        
        # Extract average metrics for each category
        for category in categories:
            if category in setting['avg_metrics']:
                metrics_data = {}
                for metric in metrics:
                    if metric in setting['avg_metrics'][category]:
                        if 'mean' in setting['avg_metrics'][category][metric]:
                            value = setting['avg_metrics'][category][metric]['mean']
                            metrics_data[metric] = value
                        else:
                            metrics_data[metric] = None
                model_summaries[model_name][category] = metrics_data
            
    except Exception as e:
        console.print(f"[red]Error processing {file_path}: {e}[/red]")

# Handle VSI ablation separately
vsi_file = "results/ablation_VSI_e150_r3_results.json"
try:
    with open(vsi_file, 'r') as f:
        vsi_data = json.load(f)
    
    # Get VSI=True and VSI=False models
    vsi_true_model = None
    vsi_false_model = None
    for setting_name, setting in vsi_data.items():
        if "VSI=True" in setting_name:
            vsi_true_model = setting_name
        elif "VSI=False" in setting_name:
            vsi_false_model = setting_name
    
    if vsi_true_model:
        model_name = "Option 2: Multi-modal (VSI=True)"
        model_summaries[model_name] = {}
        for category in categories:
            if category in vsi_data[vsi_true_model]['avg_metrics']:
                metrics_data = {}
                for metric in metrics:
                    if metric in vsi_data[vsi_true_model]['avg_metrics'][category]:
                        if 'mean' in vsi_data[vsi_true_model]['avg_metrics'][category][metric]:
                            value = vsi_data[vsi_true_model]['avg_metrics'][category][metric]['mean']
                            metrics_data[metric] = value
                        else:
                            metrics_data[metric] = None
                model_summaries[model_name][category] = metrics_data
    
    if vsi_false_model:
        model_name = "Option 2: Multi-modal (VSI=False)"
        model_summaries[model_name] = {}
        for category in categories:
            if category in vsi_data[vsi_false_model]['avg_metrics']:
                metrics_data = {}
                for metric in metrics:
                    if metric in vsi_data[vsi_false_model]['avg_metrics'][category]:
                        if 'mean' in vsi_data[vsi_false_model]['avg_metrics'][category][metric]:
                            value = vsi_data[vsi_false_model]['avg_metrics'][category][metric]['mean']
                            metrics_data[metric] = value
                        else:
                            metrics_data[metric] = None
                model_summaries[model_name][category] = metrics_data
        
except Exception as e:
    console.print(f"[red]Error processing VSI ablation file: {e}[/red]")

# Remove the generic multi-modal entry if we processed specific ones
if "Option 2: Multi-modal models" in model_summaries and "special" in model_summaries["Option 2: Multi-modal models"]:
    del model_summaries["Option 2: Multi-modal models"]

# Create a summary table for each category
for category in categories:
    table = Table(title=f"{category.capitalize()} Classification Results Comparison", box=box.ROUNDED)
    
    # Add model column and metric columns
    table.add_column("Model", style="cyan")
    for metric in metrics:
        table.add_column(f"{metric}", justify="center")
    
    # Add rows for each model
    for model_name, model_data in sorted(model_summaries.items()):
        if category in model_data:
            row = [model_name]
            for metric in metrics:
                if metric in model_data[category] and model_data[category][metric] is not None:
                    value = model_data[category][metric]
                    row.append(f"{value:.2f}%")
                else:
                    row.append("N/A")
            table.add_row(*row)
    
    console.print(table)
    console.print()