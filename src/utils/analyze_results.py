#!/usr/bin/env python3
"""
analyze_results.py - Tool for analyzing experimental results

This script allows you to:
1. Load and analyze existing result files
2. Generate visualizations
3. Create publication-ready tables and figures
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from utils.result_util import (
        process_and_export_results,
        export_to_latex,
        export_to_csv,
        export_to_markdown,
        enhance_results
    )
except ImportError:
    print("Error: Could not import result_util module from utils. Make sure it exists.")
    sys.exit(1)


def load_results(filepath):
    """Load results from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results file: {e}")
        sys.exit(1)


def check_if_enhanced(results):
    """Check if results are already enhanced"""
    # Enhanced results have a metadata section and statistics
    if "metadata" in results and "results" in results:
        for setting in results["results"].values():
            if "statistics" in setting:
                return True
    return False


def create_bar_chart(results, output_path, attributes=None, metrics=None):
    """Create bar chart comparing metrics across settings"""
    if not check_if_enhanced(results):
        print("Results are not enhanced. Cannot create visualization.")
        return
    
    if attributes is None:
        attributes = results.get("metadata", {}).get("attributes", [])[:1]  # Default to first attribute
    
    if metrics is None:
        metrics = ["overall_mAP", "overall_f1"]
    
    # For each attribute and metric, create a bar chart
    for attr in attributes:
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            # Extract data
            settings = []
            values = []
            errors = []
            
            for setting_name, setting_data in results["results"].items():
                if attr in setting_data["statistics"]:
                    if metric in setting_data["statistics"][attr]:
                        stats = setting_data["statistics"][attr][metric]
                        mean = stats.get("mean")
                        std = stats.get("std")
                        
                        if mean is not None:
                            # For readability, shorten setting name
                            short_name = setting_name
                            if len(short_name) > 30:
                                # Try to extract the most relevant part (parameter value)
                                params = [p for p in short_name.split('_') if '=' in p]
                                if params:
                                    short_name = '_'.join(params)
                            
                            settings.append(short_name)
                            values.append(mean)
                            errors.append(std if std is not None else 0)
            
            # Create bar chart
            x = np.arange(len(settings))
            plt.bar(x, values, yerr=errors, align='center', alpha=0.7, 
                   color=sns.color_palette("husl", len(settings)))
            plt.xticks(x, settings, rotation=45, ha='right')
            plt.ylabel(f"{metric} (%)")
            plt.title(f"{attr} - {metric}")
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_path, f"barchart_{attr}_{metric}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Bar chart saved to {output_file}")
            plt.close()


def create_heatmap(results, output_path, attributes=None, metrics=None):
    """Create heatmap of effect sizes"""
    if not check_if_enhanced(results) or "ablation_analysis" not in results:
        print("Results are not enhanced or don't contain ablation analysis. Cannot create heatmap.")
        return
    
    if attributes is None:
        attributes = results.get("metadata", {}).get("attributes", [])
    
    if metrics is None:
        metrics = ["overall_mAP"]
    
    # Extract effect sizes
    reference = results["ablation_analysis"]["reference_setting"]
    effect_sizes = results["ablation_analysis"]["effect_sizes"]
    
    # Group by varied parameter
    param_groups = {}
    for setting, data in effect_sizes.items():
        param = data.get('varied_parameter', 'unknown')
        if param not in param_groups:
            param_groups[param] = {}
        
        # Extract parameter value
        if f"{param}=" in setting:
            param_val = setting.split(f"{param}=")[1].split("_")[0]
        else:
            param_val = "unknown"
        
        param_groups[param][param_val] = data
    
    # For each parameter group, create a heatmap
    for param, settings in param_groups.items():
        if len(settings) <= 1:
            continue
        
        for attr in attributes:
            for metric in metrics:
                # Prepare data for heatmap
                param_vals = []
                effect_vals = []
                p_vals = []
                
                for param_val, data in settings.items():
                    if attr in data["effects"]:
                        if metric in data["effects"][attr]:
                            effect = data["effects"][attr][metric].get("effect_size")
                            p_val = data["effects"][attr][metric].get("p_value")
                            
                            if effect is not None:
                                param_vals.append(param_val)
                                effect_vals.append(effect)
                                p_vals.append(p_val)
                
                if not param_vals:
                    continue
                
                # Create heatmap (single row)
                plt.figure(figsize=(12, 2))
                
                # Convert to numpy array
                effect_array = np.array(effect_vals).reshape(1, -1)
                
                # Create heatmap
                ax = sns.heatmap(effect_array, annot=True, fmt=".2f", cmap="RdBu_r", 
                                center=0, vmin=-1.5, vmax=1.5, cbar_kws={"label": "Effect Size"})
                
                # Add significance markers
                for i, p_val in enumerate(p_vals):
                    if p_val is not None and p_val < 0.05:
                        ax.text(i + 0.5, 0.5, '*', ha='center', va='center', color='black', fontsize=15)
                
                # Set labels
                plt.yticks([0.5], [f"{attr}_{metric}"], rotation=0)
                plt.xticks(np.arange(len(param_vals)) + 0.5, param_vals, rotation=45, ha='right')
                plt.title(f"Effect Sizes for {param}")
                plt.tight_layout()
                
                # Save figure
                output_file = os.path.join(output_path, f"heatmap_{param}_{attr}_{metric}.png")
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Heatmap saved to {output_file}")
                plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--output", "-o", default="results", help="Output directory")
    parser.add_argument("--attributes", "-a", nargs="+", help="Attributes to analyze")
    parser.add_argument("--metrics", "-m", nargs="+", help="Metrics to analyze")
    parser.add_argument("--enhance", action="store_true", help="Enhance results and export in multiple formats")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load results
    results = load_results(args.results_file)
    
    # Determine if we need to enhance the results first
    enhanced_results = results
    if not check_if_enhanced(results) or args.enhance:
        print("Enhancing results...")
        # Estimate settings from filename
        filename = os.path.basename(args.results_file)
        parts = filename.split("_")
        
        epochs = 0
        repetitions = 0
        for part in parts:
            if part.startswith("e") and part[1:].isdigit():
                epochs = int(part[1:])
            elif part.startswith("r") and part[1:].isdigit():
                repetitions = int(part[1:])
        
        # Get attributes
        attributes = args.attributes
        if attributes is None:
            # Try to extract from results
            attributes = []
            for setting in results.values():
                if "detailed_results" in setting:
                    for rep in setting["detailed_results"]:
                        if "results" in rep:
                            attributes.extend(list(rep["results"].keys()))
            attributes = list(set(attributes))
        
        settings = {
            "epochs": epochs,
            "repetitions": repetitions,
        }
        
        # Process and export in multiple formats
        base_filename = os.path.join(args.output, Path(args.results_file).stem)
        enhanced_results = enhance_results(results, settings, attributes)
        
        if args.enhance:
            # Export to various formats
            export_to_latex(enhanced_results, base_filename + ".tex", attributes, args.metrics)
            export_to_csv(enhanced_results, base_filename + ".csv", attributes, args.metrics)
            export_to_markdown(enhanced_results, base_filename + ".md", attributes, args.metrics)
            
            # Save enhanced JSON
            with open(base_filename + "_enhanced.json", "w", encoding="utf-8") as f:
                json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
            
            print(f"Enhanced results saved to {base_filename}_enhanced.json")
    
    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        create_bar_chart(enhanced_results, args.output, args.attributes, args.metrics)
        create_heatmap(enhanced_results, args.output, args.attributes, args.metrics)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()