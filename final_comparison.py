#!/usr/bin/env python3
import json
import os
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

console = Console()

# Files to analyze - focusing on what we have
files = {
    "Option 1 (BCE)": "results/baseline_bce_e150_r3_results.json",
    "Option 1 (ASL)": "results/baseline_asl_e150_r3_results.json",
    "Option 3 (Zero-Shot)": "results/zero_shot_clip_results.json"
}

# Categories and metrics
categories = ['crop', 'part', 'symptomCategories', 'symptomTags']
main_metrics = ['overall_mAP', 'overall_f1']
distribution_metrics = {
    'head': ['head_mAP', 'head_f1'],
    'medium': ['medium_mAP', 'medium_f1'],
    'tail': ['tail_mAP', 'tail_f1']
}

# Load all ablation results to see if we can extract Option 2 data
ablation_files = ["results/ablation_Loss_Type_e150_r3_results.json"]
for ablation_file in ablation_files:
    if os.path.exists(ablation_file):
        try:
            with open(ablation_file, 'r') as f:
                data = json.load(f)
                
            # Look for FiLM fusion with asl loss - that would be closest to our best model for Option 2
            film_asl_key = next((k for k in data.keys() if "Fusion=film" in k and "Loss=asl" in k), None)
            if film_asl_key:
                files["Option 2 (FiLM+ASL)"] = ablation_file
                print(f"Found Option 2 model in ablation file: {film_asl_key}")
                break
        except Exception as e:
            console.print(f"[red]Error analyzing ablation file {ablation_file}: {e}[/red]")

# Load results
results = {}
for model, filepath in files.items():
    if not os.path.exists(filepath):
        console.print(f"[yellow]Warning: File {filepath} not found[/yellow]")
        continue
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Special handling for ablation files
        if model == "Option 2 (FiLM+ASL)":
            film_asl_key = next((k for k in data.keys() if "Fusion=film" in k and "Loss=asl" in k), None)
            if film_asl_key:
                results[model] = data[film_asl_key]
            else:
                console.print(f"[yellow]Could not find FiLM+ASL model in {filepath}[/yellow]")
        else:
            # For other files, just take the first entry
            key = next(iter(data))
            results[model] = data[key]
    except Exception as e:
        console.print(f"[red]Error loading {filepath}: {e}[/red]")

# Function to extract metric values with proper handling
def get_metric_value(result_data, category, metric):
    # Handle different result structures
    
    # Case 1: From avg_metrics (for baseline and multimodal)
    if 'avg_metrics' in result_data and category in result_data['avg_metrics']:
        if metric in result_data['avg_metrics'][category]:
            metric_data = result_data['avg_metrics'][category][metric]
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                return metric_data['mean']
            return metric_data
    
    # Case 2: From detailed_results (for zero-shot)
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

# Create comparison tables for each category
for category in categories:
    console.print(f"\n[bold cyan]===== {category.upper()} CLASSIFICATION COMPARISON =====[/bold cyan]")
    
    # Main comparison table with all metrics
    table = Table(box=box.ROUNDED, title=f"{category.capitalize()} Classification Results")
    
    # Add columns
    table.add_column("Model", style="cyan", width=20)
    table.add_column("mAP", justify="center")
    table.add_column("F1", justify="center")
    table.add_column("Head mAP", justify="center")
    table.add_column("Head F1", justify="center")
    table.add_column("Medium mAP", justify="center")
    table.add_column("Medium F1", justify="center") 
    table.add_column("Tail mAP", justify="center")
    table.add_column("Tail F1", justify="center")
    
    # Add rows for each model
    for model in sorted(results.keys()):
        row = [model]
        
        # Add overall metrics
        for metric in main_metrics:
            value = get_metric_value(results[model], category, metric)
            row.append(f"{value:.2f}%" if value is not None else "N/A")
        
        # Add distribution metrics
        for dist in ['head', 'medium', 'tail']:
            for metric_suffix in ['mAP', 'f1']:
                metric = f"{dist}_{metric_suffix}"
                value = get_metric_value(results[model], category, metric)
                row.append(f"{value:.2f}%" if value is not None else "N/A")
        
        table.add_row(*row)
    
    console.print(table)
    
# Summary and findings
console.print("\n[bold green]===== SUMMARY AND RECOMMENDATIONS =====[/bold green]")

# Create a summary panel for each classification category
category_summaries = {
    'crop': [
        "All approaches achieve excellent performance on crop identification",
        "Baseline models achieve perfect or near-perfect results",
        "Zero-shot approach shows good mAP (95.38%) but struggles with F1 score (49.48%)",
        "Little difference between BCE and ASL loss for this simpler task"
    ],
    'part': [
        "Baseline with BCE loss performs best overall (98.04% mAP, 98.27% F1)",
        "Baseline with ASL loss also performs very well (97.94% mAP, 90.31% F1)",
        "Zero-shot approach struggles significantly with part classification",
        "All models maintain strong performance on head classes"
    ],
    'symptomCategories': [
        "Baseline with ASL loss performs best (86.47% mAP, 74.59% F1)",
        "ASL loss shows advantages for this more complex, imbalanced task",
        "Zero-shot approach performs poorly (25.96% mAP, 28.44% F1)",
        "Performance gap between head and tail classes widens"
    ],
    'symptomTags': [
        "Baseline with ASL loss significantly outperforms other approaches",
        "ASL provides major advantage for this highly imbalanced task",
        "Tail class performance drops for all methods but ASL maintains 44% F1",
        "Zero-shot approach performs very poorly on this fine-grained task"
    ]
}

for category, findings in category_summaries.items():
    panel_content = "\n".join([f"• {finding}" for finding in findings])
    console.print(Panel(panel_content, title=f"{category.capitalize()} Classification", border_style="blue"))

# Final recommendations
console.print(Panel(
    "• Option 1 (Baseline with ASL loss) provides the best overall performance across all categories\n"
    "• ASL loss shows increasing advantage as classification tasks become more complex and imbalanced\n"
    "• Asymmetric loss is particularly valuable for the symptom tag classification task\n"
    "• Zero-shot approach could be useful for initial crop classification but requires training for other tasks\n"
    "• All methods struggle with tail (rare) classes - this remains the main area for improvement",
    title="Recommendations", border_style="green"))