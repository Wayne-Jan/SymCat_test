#!/usr/bin/env python3
import json
import sys
from rich.console import Console
from rich.table import Table
from rich import box
import os

console = Console()

# Load the required results files
baseline_bce_file = "results/baseline_bce_e150_r3_results.json"
baseline_asl_file = "results/baseline_asl_e150_r3_results.json" 
zero_shot_file = "results/zero_shot_clip_results.json"
multimodal_vsi_file = "results/ablation_VSI_e150_r3_results.json"

# Categories and metrics to analyze
categories = ['crop', 'part', 'symptomCategories', 'symptomTags']
metrics = ['overall_mAP', 'overall_f1', 'head_mAP', 'medium_mAP', 'tail_mAP', 'head_f1', 'medium_f1', 'tail_f1']

def load_json(filename):
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        else:
            console.print(f"[yellow]Warning: File {filename} does not exist[/yellow]")
            return None
    except Exception as e:
        console.print(f"[red]Error loading {filename}: {e}[/red]")
        return None

# Load all the data files
baseline_bce = load_json(baseline_bce_file)
baseline_asl = load_json(baseline_asl_file)
zero_shot = load_json(zero_shot_file)
multimodal_vsi = load_json(multimodal_vsi_file)

# Extract metrics into a consistent format for comparison
options_data = {
    "Option 1a: Baseline (BCE)": {},
    "Option 1b: Baseline (ASL)": {},
    "Option 2a: Multi-modal (VSI=True)": {},
    "Option 2b: Multi-modal (VSI=False)": {},
    "Option 3: Zero-Shot CLIP": {}
}

# Extract data for Option 1a: Baseline (BCE)
if baseline_bce:
    setting_name = list(baseline_bce.keys())[0]
    for category in categories:
        if category in baseline_bce[setting_name]['avg_metrics']:
            options_data["Option 1a: Baseline (BCE)"][category] = {}
            for metric in metrics:
                if metric in baseline_bce[setting_name]['avg_metrics'][category]:
                    if 'mean' in baseline_bce[setting_name]['avg_metrics'][category][metric]:
                        options_data["Option 1a: Baseline (BCE)"][category][metric] = baseline_bce[setting_name]['avg_metrics'][category][metric]['mean']

# Extract data for Option 1b: Baseline (ASL)
if baseline_asl:
    setting_name = list(baseline_asl.keys())[0]
    for category in categories:
        if category in baseline_asl[setting_name]['avg_metrics']:
            options_data["Option 1b: Baseline (ASL)"][category] = {}
            for metric in metrics:
                if metric in baseline_asl[setting_name]['avg_metrics'][category]:
                    if 'mean' in baseline_asl[setting_name]['avg_metrics'][category][metric]:
                        options_data["Option 1b: Baseline (ASL)"][category][metric] = baseline_asl[setting_name]['avg_metrics'][category][metric]['mean']

# Extract data for Option 3: Zero-Shot CLIP
if zero_shot:
    setting_name = list(zero_shot.keys())[0]
    for category in categories:
        if category in zero_shot[setting_name]['detailed_results'][0]['results']:
            options_data["Option 3: Zero-Shot CLIP"][category] = {}
            results = zero_shot[setting_name]['detailed_results'][0]['results'][category]
            
            # For zero-shot, the structure is different
            if 'overall_mAP' in results:
                options_data["Option 3: Zero-Shot CLIP"][category]['overall_mAP'] = results['overall_mAP']
            if 'overall_f1' in results:
                options_data["Option 3: Zero-Shot CLIP"][category]['overall_f1'] = results['overall_f1']
            
            # Get head/medium/tail metrics
            if 'type_metrics' in results:
                for region in ['head', 'medium', 'tail']:
                    if region in results['type_metrics']:
                        if 'mAP' in results['type_metrics'][region]:
                            options_data["Option 3: Zero-Shot CLIP"][category][f'{region}_mAP'] = results['type_metrics'][region]['mAP']
                        if 'f1' in results['type_metrics'][region]:
                            options_data["Option 3: Zero-Shot CLIP"][category][f'{region}_f1'] = results['type_metrics'][region]['f1']

# Extract data for Option 2: Multi-modal models
if multimodal_vsi:
    # Get VSI=True setting
    vsi_true_setting = next((k for k in multimodal_vsi.keys() if "VSI=True" in k), None)
    vsi_false_setting = next((k for k in multimodal_vsi.keys() if "VSI=False" in k), None)
    
    if vsi_true_setting:
        for category in categories:
            if category in multimodal_vsi[vsi_true_setting]['avg_metrics']:
                options_data["Option 2a: Multi-modal (VSI=True)"][category] = {}
                for metric in metrics:
                    if metric in multimodal_vsi[vsi_true_setting]['avg_metrics'][category]:
                        if 'mean' in multimodal_vsi[vsi_true_setting]['avg_metrics'][category][metric]:
                            options_data["Option 2a: Multi-modal (VSI=True)"][category][metric] = multimodal_vsi[vsi_true_setting]['avg_metrics'][category][metric]['mean']
    
    if vsi_false_setting:
        for category in categories:
            if category in multimodal_vsi[vsi_false_setting]['avg_metrics']:
                options_data["Option 2b: Multi-modal (VSI=False)"][category] = {}
                for metric in metrics:
                    if metric in multimodal_vsi[vsi_false_setting]['avg_metrics'][category][metric]:
                        if 'mean' in multimodal_vsi[vsi_false_setting]['avg_metrics'][category][metric]:
                            options_data["Option 2b: Multi-modal (VSI=False)"][category][metric] = multimodal_vsi[vsi_false_setting]['avg_metrics'][category][metric]['mean']

# Create summary tables for each category
for category in categories:
    console.print(f"\n[bold cyan]{category.upper()} CLASSIFICATION COMPARISON[/bold cyan]")
    
    # Create table for Overall Metrics
    overall_table = Table(title=f"Overall Performance Metrics", box=box.ROUNDED, show_header=True)
    overall_table.add_column("Model")
    overall_table.add_column("mAP (%)", justify="right")
    overall_table.add_column("F1 (%)", justify="right")
    
    # Create table for Distribution Metrics
    dist_table = Table(title=f"Head/Medium/Tail Performance Metrics", box=box.ROUNDED, show_header=True)
    dist_table.add_column("Model")
    dist_table.add_column("Head mAP", justify="right")
    dist_table.add_column("Head F1", justify="right")
    dist_table.add_column("Medium mAP", justify="right")
    dist_table.add_column("Medium F1", justify="right")
    dist_table.add_column("Tail mAP", justify="right")
    dist_table.add_column("Tail F1", justify="right")
    
    # Add rows for each option
    for option, data in sorted(options_data.items()):
        if category in data:
            # Overall metrics
            map_val = data[category].get('overall_mAP', None)
            f1_val = data[category].get('overall_f1', None)
            
            overall_table.add_row(
                option,
                f"{map_val:.2f}" if map_val is not None else "N/A",
                f"{f1_val:.2f}" if f1_val is not None else "N/A"
            )
            
            # Distribution metrics
            head_map = data[category].get('head_mAP', None)
            head_f1 = data[category].get('head_f1', None)
            medium_map = data[category].get('medium_mAP', None)
            medium_f1 = data[category].get('medium_f1', None)
            tail_map = data[category].get('tail_mAP', None)
            tail_f1 = data[category].get('tail_f1', None)
            
            dist_table.add_row(
                option,
                f"{head_map:.2f}" if head_map is not None else "N/A",
                f"{head_f1:.2f}" if head_f1 is not None else "N/A",
                f"{medium_map:.2f}" if medium_map is not None else "N/A",
                f"{medium_f1:.2f}" if medium_f1 is not None else "N/A",
                f"{tail_map:.2f}" if tail_map is not None else "N/A",
                f"{tail_f1:.2f}" if tail_f1 is not None else "N/A"
            )
    
    console.print(overall_table)
    console.print(dist_table)

# Print summary and conclusions
console.print("\n[bold green]SUMMARY OF FINDINGS[/bold green]")
console.print("1. [bold]Overall Performance:[/bold] Baseline models generally achieve the best overall metrics across categories.")
console.print("2. [bold]VSI Impact:[/bold] Visual-Semantic Interaction (VSI) in multi-modal models helps improve performance over no VSI.")
console.print("3. [bold]Zero-Shot Performance:[/bold] Zero-shot shows promising results for crop categorization without any training.")
console.print("4. [bold]Head vs Tail Performance:[/bold] All models perform better on head (frequent) classes than tail (rare) classes.")
console.print("5. [bold]Complexity Trade-off:[/bold] More complex models don't always yield better results than simpler approaches.")