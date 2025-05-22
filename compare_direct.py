#!/usr/bin/env python3
import json
import os
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

console = Console()

# Files to analyze
files = {
    "Option 1 (BCE)": "results/baseline_bce_e150_r3_results.json",
    "Option 1 (ASL)": "results/baseline_asl_e150_r3_results.json",
    "Option 2 VSI=True": "results/ablation_VSI_e150_r3_results.json",
    "Option 3": "results/zero_shot_clip_results.json"
}

# Categories and metrics
categories = ['crop', 'part', 'symptomCategories', 'symptomTags']
main_metrics = ['overall_mAP', 'overall_f1']
distribution_metrics = {
    'head': ['head_mAP', 'head_f1'],
    'medium': ['medium_mAP', 'medium_f1'],
    'tail': ['tail_mAP', 'tail_f1']
}

# Load results
results = {}
for model, filepath in files.items():
    if not os.path.exists(filepath):
        console.print(f"[yellow]Warning: File {filepath} not found[/yellow]")
        continue
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Special handling for VSI ablation
        if model == "Option 2 VSI=True":
            vsi_true_key = next((k for k in data.keys() if "VSI=True" in k), None)
            vsi_false_key = next((k for k in data.keys() if "VSI=False" in k), None)
            
            if vsi_true_key:
                results["Option 2 (VSI=True)"] = data[vsi_true_key]
            if vsi_false_key:
                results["Option 2 (VSI=False)"] = data[vsi_false_key]
        else:
            # For other files, just take the first entry
            key = next(iter(data))
            results[model] = data[key]
    except Exception as e:
        console.print(f"[red]Error loading {filepath}: {e}[/red]")

# Function to extract metric values with proper handling
def get_metric_value(result_data, category, metric):
    if 'avg_metrics' in result_data and category in result_data['avg_metrics']:
        if metric in result_data['avg_metrics'][category]:
            metric_data = result_data['avg_metrics'][category][metric]
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                return metric_data['mean']
            return metric_data
    
    # Check if it's a zero-shot result
    if 'detailed_results' in result_data and len(result_data['detailed_results']) > 0:
        if 'results' in result_data['detailed_results'][0]:
            zero_shot_results = result_data['detailed_results'][0]['results']
            if category in zero_shot_results:
                if metric.startswith('head_') or metric.startswith('medium_') or metric.startswith('tail_'):
                    # Extract from type_metrics
                    distribution = metric.split('_')[0]  # head, medium, tail
                    metric_type = metric.split('_')[1]   # mAP, f1
                    
                    if 'type_metrics' in zero_shot_results[category] and distribution in zero_shot_results[category]['type_metrics']:
                        return zero_shot_results[category]['type_metrics'][distribution].get(metric_type)
                else:
                    # Direct metrics in zero-shot
                    return zero_shot_results[category].get(metric)
    
    return None

# Process results for each category
for category in categories:
    console.print(f"\n[bold cyan]===== {category.upper()} CLASSIFICATION COMPARISON =====[/bold cyan]")
    
    # Create table for overall metrics
    overall_table = Table(box=box.ROUNDED)
    overall_table.add_column("Model")
    for metric in main_metrics:
        overall_table.add_column(metric.replace('overall_', '').upper())
    
    # Add rows for each model
    all_models = sorted(results.keys())
    for model in all_models:
        row = [model]
        for metric in main_metrics:
            value = get_metric_value(results[model], category, metric)
            if value is not None:
                row.append(f"{value:.2f}%")
            else:
                row.append("N/A")
        overall_table.add_row(*row)
    
    # Create table for distribution metrics
    dist_table = Table(box=box.ROUNDED)
    dist_table.add_column("Model")
    for dist in ['head', 'medium', 'tail']:
        dist_table.add_column(f"{dist.capitalize()} mAP")
        dist_table.add_column(f"{dist.capitalize()} F1")
    
    # Add rows for distribution metrics
    for model in all_models:
        row = [model]
        for dist in ['head', 'medium', 'tail']:
            for metric_suffix in ['mAP', 'f1']:
                metric = f"{dist}_{metric_suffix}"
                value = get_metric_value(results[model], category, metric)
                if value is not None:
                    row.append(f"{value:.2f}%")
                else:
                    row.append("N/A")
        dist_table.add_row(*row)
    
    # Print tables
    console.print(Panel(overall_table, title="Overall Metrics"))
    console.print(Panel(dist_table, title="Head/Medium/Tail Distribution Metrics"))
    
# Analyze and provide key findings
console.print("\n[bold green]===== KEY FINDINGS =====[/bold green]")

console.print(Panel("[bold]CROP CLASSIFICATION:[/bold]\n"
                   "- Option 1 (Baseline) and Option 2 (VSI=True) achieve near-perfect results\n"
                   "- Option 3 (Zero-Shot) shows good mAP but lower F1\n"
                   "- All approaches perform well on crop identification", 
                   title="Crop Classification"))

console.print(Panel("[bold]PART CLASSIFICATION:[/bold]\n"
                   "- Option 1 (Baseline) performs best with both BCE and ASL\n"
                   "- Option 2 (VSI=True) shows strong but slightly lower performance\n"
                   "- Option 3 (Zero-Shot) struggles with part classification\n"
                   "- Tail classes remain challenging for all methods", 
                   title="Part Classification"))

console.print(Panel("[bold]SYMPTOM CATEGORIES CLASSIFICATION:[/bold]\n"
                   "- Option 1 (Baseline with ASL) outperforms other approaches\n"
                   "- Option 2 (Multi-modal) shows declining advantage\n"
                   "- Option 3 (Zero-Shot) performs poorly\n"
                   "- Gap between head and tail classes increases", 
                   title="Symptom Categories"))

console.print(Panel("[bold]SYMPTOM TAGS CLASSIFICATION:[/bold]\n"
                   "- Option 1 (Baseline with ASL) performs best\n"
                   "- Option 2 (Multi-modal) struggles with complex symptom classification\n"
                   "- Option 3 (Zero-Shot) performs poorly\n"
                   "- Tail class performance drops significantly for all methods", 
                   title="Symptom Tags"))

console.print(Panel("[bold]OVERALL COMPARISON:[/bold]\n"
                   "1. Baseline classifiers generally outperform more complex models\n"
                   "2. ASL loss provides advantage for imbalanced datasets (symptom tags)\n"
                   "3. Visual-Semantic Interaction helps in multi-modal approaches but not enough to surpass baseline\n"
                   "4. Zero-shot approach works best on general categories (crops) but struggles with specifics\n"
                   "5. Rare class (tail) performance remains the biggest challenge for all methods", 
                   title="Overall Comparison"))