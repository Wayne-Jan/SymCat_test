"""
result_utils.py - Enhanced results recording and statistical analysis

This module provides utilities for:
1. Storing experimental results in enhanced formats
2. Computing statistical measures for robust analysis
3. Exporting results in various formats (LaTeX, CSV, etc.)
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime
from collections import defaultdict
import hashlib


class NumpyEncoder(json.JSONEncoder):
    """JSON Encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def compute_statistics(values, alpha=0.05):
    """
    Compute statistical measures for a list of values
    
    Args:
        values: List of numeric values
        alpha: Significance level for confidence interval
    
    Returns:
        Dict with statistical measures
    """
    if not values or len(values) == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "ci_lower": None,
            "ci_upper": None,
            "values": []
        }
        
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Use sample standard deviation
    
    # Calculate confidence interval
    if n > 1:
        # Use t-distribution for small samples
        t_critical = stats.t.ppf(1 - alpha/2, n-1)
        margin_error = t_critical * (std / np.sqrt(n))
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
    else:
        ci_lower = None
        ci_upper = None
    
    return {
        "mean": mean,
        "std": std,
        "min": np.min(values),
        "max": np.max(values),
        "median": np.median(values),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "values": values.tolist()
    }


def calculate_effect_size(values_a, values_b):
    """
    Calculate Cohen's d effect size between two groups
    
    Args:
        values_a: First group values
        values_b: Second group values
        
    Returns:
        Cohen's d effect size
    """
    if len(values_a) < 2 or len(values_b) < 2:
        return None
    
    # Convert to numpy arrays
    a = np.array(values_a)
    b = np.array(values_b)
    
    # Calculate means
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    
    # Calculate pooled standard deviation
    n_a = len(a)
    n_b = len(b)
    s_a = np.std(a, ddof=1)  # Sample standard deviation
    s_b = np.std(b, ddof=1)
    
    # Pooled standard deviation
    s_pooled = np.sqrt(((n_a - 1) * s_a**2 + (n_b - 1) * s_b**2) / (n_a + n_b - 2))
    
    # Cohen's d
    if s_pooled == 0:
        return float('inf') if mean_a != mean_b else 0.0
    
    d = (mean_b - mean_a) / s_pooled
    return d


def calculate_p_value(values_a, values_b):
    """
    Calculate p-value for the difference between two groups
    using independent t-test
    
    Args:
        values_a: First group values
        values_b: Second group values
        
    Returns:
        p-value and test statistic
    """
    if len(values_a) < 2 or len(values_b) < 2:
        return None, None
    
    # Independent samples t-test
    t_stat, p_val = stats.ttest_ind(values_a, values_b, equal_var=False)
    
    return p_val, t_stat


def compute_attribute_statistics(rep_results, attribute, metrics=None):
    """
    Compute statistics for a specific attribute across repetitions
    
    Args:
        rep_results: List of results from different repetitions
        attribute: The attribute to analyze
        metrics: List of metrics to analyze (default: all available)
        
    Returns:
        Dict with metrics and their statistics
    """
    if metrics is None:
        # Default metrics
        metrics = ['overall_mAP', 'overall_f1', 'overall_acc', 
                  'head_mAP', 'medium_mAP', 'tail_mAP',
                  'head_f1', 'medium_f1', 'tail_f1']
    
    # Extract metric values across repetitions
    metric_values = defaultdict(list)
    
    for rep in rep_results:
        if 'results' not in rep or attribute not in rep['results']:
            continue
        
        attr_results = rep['results'][attribute]
        
        # Collect overall metrics
        for metric in ['overall_mAP', 'overall_f1', 'overall_acc']:
            if metric in attr_results:
                metric_values[metric].append(attr_results[metric])
        
        # Collect H/M/T metrics
        if 'type_metrics' in attr_results:
            tm = attr_results['type_metrics']
            for typ in ['head', 'medium', 'tail']:
                if typ in tm:
                    for m in ['mAP', 'f1']:
                        key = f"{typ}_{m}"
                        if m in tm[typ] and tm[typ][m] is not None:
                            metric_values[key].append(tm[typ][m])
    
    # Compute statistics for each metric
    stats_dict = {}
    for metric, values in metric_values.items():
        stats_dict[metric] = compute_statistics(values)
    
    return stats_dict


def compute_ablation_effect_sizes(all_settings, reference_setting, attributes, metrics=None):
    """
    Compute effect sizes for ablation study comparisons
    
    Args:
        all_settings: Dict of all settings results
        reference_setting: Key of the reference setting to compare against
        attributes: List of attributes to analyze
        metrics: List of metrics to consider
        
    Returns:
        Dict with effect sizes for each setting compared to reference
    """
    if reference_setting not in all_settings:
        return {}
    
    if metrics is None:
        metrics = ['overall_mAP', 'overall_f1']
    
    ref_data = all_settings[reference_setting]
    effects = {}
    
    for setting_name, setting_data in all_settings.items():
        if setting_name == reference_setting:
            continue
        
        # Extract parameter that's being varied
        varied_param = identify_varied_parameter(setting_name, reference_setting)
        if not varied_param:
            varied_param = "unknown"
        
        setting_effects = {}
        for attr in attributes:
            attr_effects = {}
            
            for metric in metrics:
                # Get reference values
                if (attr in ref_data.get('detailed_results', [{}])[0].get('results', {}) and
                    metric in ref_data.get('detailed_results', [{}])[0].get('results', {}).get(attr, {})):
                    
                    ref_values = []
                    for rep in ref_data.get('detailed_results', []):
                        if (attr in rep.get('results', {}) and 
                            metric in rep.get('results', {}).get(attr, {})):
                            ref_values.append(rep['results'][attr][metric])
                    
                    # Get comparison values
                    comp_values = []
                    for rep in setting_data.get('detailed_results', []):
                        if (attr in rep.get('results', {}) and 
                            metric in rep.get('results', {}).get(attr, {})):
                            comp_values.append(rep['results'][attr][metric])
                    
                    # Calculate effect size
                    effect = calculate_effect_size(ref_values, comp_values)
                    p_value, _ = calculate_p_value(ref_values, comp_values)
                    
                    attr_effects[metric] = {
                        'effect_size': effect,
                        'p_value': p_value,
                        'reference_mean': np.mean(ref_values) if ref_values else None,
                        'comparison_mean': np.mean(comp_values) if comp_values else None,
                        'difference': (np.mean(comp_values) - np.mean(ref_values)) if ref_values and comp_values else None,
                        'significant': p_value < 0.05 if p_value is not None else None
                    }
            
            setting_effects[attr] = attr_effects
        
        effects[setting_name] = {
            'varied_parameter': varied_param,
            'effects': setting_effects
        }
    
    return effects


def identify_varied_parameter(setting_name, reference_name):
    """
    Try to identify which parameter varies between settings
    based on naming conventions
    """
    if '_' not in setting_name or '_' not in reference_name:
        return None
    
    # Split into parts
    setting_parts = setting_name.split('_')
    ref_parts = reference_name.split('_')
    
    # Compare parts
    for s, r in zip(setting_parts, ref_parts):
        if s != r:
            # Try to extract parameter name
            if '=' in s and '=' in r:
                param = s.split('=')[0]
                return param
    
    return None


def enhance_results(original_results, settings, attributes):
    """
    Enhance raw results with additional statistical analysis
    
    Args:
        original_results: Original results dictionary
        settings: Settings used for the experiments
        attributes: List of attributes analyzed
        
    Returns:
        Enhanced results dictionary
    """
    enhanced = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes,
            "settings_hash": hashlib.md5(str(settings).encode()).hexdigest()[:8]
        },
        "settings": settings,
        "results": {}
    }
    
    # Process each experimental setting
    for setting_name, setting_data in original_results.items():
        # Copy original detailed results
        detailed_results = setting_data.get('detailed_results', [])
        
        # Enhanced statistics
        attribute_stats = {}
        for attr in attributes:
            attribute_stats[attr] = compute_attribute_statistics(detailed_results, attr)
        
        enhanced["results"][setting_name] = {
            "setting_params": setting_data.get('setting_params', {}),
            "statistics": attribute_stats,
            "detailed_results": detailed_results
        }
    
    # For ablation studies, add effect sizes
    if len(original_results) > 1:
        # Find baseline/reference setting
        # Heuristic: First setting is typically reference
        reference_setting = list(original_results.keys())[0]
        
        # Compute effect sizes compared to reference
        effects = compute_ablation_effect_sizes(
            original_results, 
            reference_setting,
            attributes
        )
        
        enhanced["ablation_analysis"] = {
            "reference_setting": reference_setting,
            "effect_sizes": effects
        }
    
    return enhanced


def save_enhanced_results(results, filename):
    """
    Save enhanced results to JSON
    
    Args:
        results: Enhanced results dictionary
        filename: Output filename
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    
    print(f"Enhanced results saved to {filename}")


def export_to_latex(results, filename, attributes=None, metrics=None):
    """
    Export results to LaTeX table format
    
    Args:
        results: Enhanced results dictionary
        filename: Output filename
        attributes: List of attributes to include (default: all)
        metrics: List of metrics to include (default: mAP and F1)
    """
    if attributes is None:
        attributes = results.get("metadata", {}).get("attributes", [])
    
    if metrics is None:
        metrics = ['overall_mAP', 'overall_f1']
    
    # For ablation studies
    if "ablation_analysis" in results:
        reference = results["ablation_analysis"]["reference_setting"]
        effect_sizes = results["ablation_analysis"]["effect_sizes"]
        
        # Group by varied parameter
        param_groups = defaultdict(list)
        for setting, data in effect_sizes.items():
            param = data.get('varied_parameter', 'unknown')
            param_groups[param].append((setting, data))
        
        for param, settings in param_groups.items():
            # Skip if only one setting with this parameter
            if len(settings) <= 1:
                continue
                
            # Create LaTeX table for this parameter
            latex_content = []
            latex_content.append("\\begin{table}[htbp]")
            latex_content.append("\\centering")
            latex_content.append(f"\\caption{{Ablation study results for parameter: {param}}}")
            
            # Table header
            header = ["Setting"]
            for metric in metrics:
                header.append(f"{metric} (\\%)")
                header.append("Effect Size")
                header.append("p-value")
            
            latex_content.append("\\begin{tabular}{" + "l" + "ccc" * len(metrics) + "}")
            latex_content.append("\\toprule")
            latex_content.append(" & ".join(header) + " \\\\")
            latex_content.append("\\midrule")
            
            # Reference setting row
            ref_row = [f"{param}=Reference"]
            ref_results = results["results"][reference]
            
            for metric in metrics:
                for attr in attributes:
                    if attr in ref_results["statistics"]:
                        stats = ref_results["statistics"][attr].get(metric, {})
                        mean = stats.get("mean")
                        ci_lower = stats.get("ci_lower")
                        ci_upper = stats.get("ci_upper")
                        
                        if mean is not None:
                            if ci_lower is not None and ci_upper is not None:
                                ref_row.append(f"{mean:.2f} ({ci_lower:.2f}--{ci_upper:.2f})")
                            else:
                                ref_row.append(f"{mean:.2f}")
                        else:
                            ref_row.append("N/A")
                        
                        # Empty cells for effect size and p-value (reference)
                        ref_row.append("--")
                        ref_row.append("--")
            
            latex_content.append(" & ".join(ref_row) + " \\\\")
            
            # Other settings
            for setting_name, data in settings:
                setting_val = setting_name.split(f"{param}=")[1].split("_")[0] if f"{param}=" in setting_name else "Unknown"
                row = [f"{param}={setting_val}"]
                
                for metric in metrics:
                    for attr in attributes:
                        if attr in data["effects"]:
                            effect_data = data["effects"][attr].get(metric, {})
                            comp_mean = effect_data.get("comparison_mean")
                            effect = effect_data.get("effect_size")
                            p_val = effect_data.get("p_value")
                            
                            # Append mean
                            if comp_mean is not None:
                                row.append(f"{comp_mean:.2f}")
                            else:
                                row.append("N/A")
                            
                            # Append effect size
                            if effect is not None:
                                # Format effect size with sign and significance marker
                                effect_str = f"{effect:.2f}"
                                # Add significance marker
                                if p_val is not None and p_val < 0.05:
                                    effect_str += "*"
                                if p_val is not None and p_val < 0.01:
                                    effect_str += "*"
                                if p_val is not None and p_val < 0.001:
                                    effect_str += "*"
                                row.append(effect_str)
                            else:
                                row.append("--")
                            
                            # Append p-value
                            if p_val is not None:
                                if p_val < 0.001:
                                    row.append("$<$0.001")
                                else:
                                    row.append(f"{p_val:.3f}")
                            else:
                                row.append("--")
                
                latex_content.append(" & ".join(row) + " \\\\")
            
            latex_content.append("\\bottomrule")
            latex_content.append("\\end{tabular}")
            latex_content.append("\\end{table}")
            
            # Write to file
            param_filename = f"{filename.replace('.tex', '')}_{param}.tex"
            with open(param_filename, "w") as f:
                f.write("\n".join(latex_content))
            
            print(f"LaTeX table for parameter {param} saved to {param_filename}")
    
    # Generate overall results table
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Overall Results}")
    
    # Table header
    header = ["Setting"]
    for attr in attributes:
        for metric in metrics:
            header.append(f"{attr}_{metric} (\\%)")
    
    latex_content.append("\\begin{tabular}{" + "l" + "c" * (len(attributes) * len(metrics)) + "}")
    latex_content.append("\\toprule")
    latex_content.append(" & ".join(header) + " \\\\")
    latex_content.append("\\midrule")
    
    # Results rows
    for setting_name, setting_data in results["results"].items():
        setting_row = [setting_name]
        
        for attr in attributes:
            if attr in setting_data["statistics"]:
                for metric in metrics:
                    stats = setting_data["statistics"][attr].get(metric, {})
                    mean = stats.get("mean")
                    std = stats.get("std")
                    
                    if mean is not None:
                        if std is not None:
                            setting_row.append(f"{mean:.2f} $\\pm$ {std:.2f}")
                        else:
                            setting_row.append(f"{mean:.2f}")
                    else:
                        setting_row.append("N/A")
        
        latex_content.append(" & ".join(setting_row) + " \\\\")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Write to file
    overall_filename = f"{filename.replace('.tex', '')}_overall.tex"
    with open(overall_filename, "w") as f:
        f.write("\n".join(latex_content))
    
    print(f"Overall LaTeX table saved to {overall_filename}")


def export_to_csv(results, filename, attributes=None, metrics=None):
    """
    Export results to CSV format for spreadsheet analysis
    
    Args:
        results: Enhanced results dictionary
        filename: Output filename
        attributes: List of attributes to include (default: all)
        metrics: List of metrics to include (default: all)
    """
    if attributes is None:
        attributes = results.get("metadata", {}).get("attributes", [])
    
    if metrics is None:
        metrics = ['overall_mAP', 'overall_f1', 'overall_acc', 
                  'head_mAP', 'medium_mAP', 'tail_mAP',
                  'head_f1', 'medium_f1', 'tail_f1']
    
    # Prepare data for DataFrame
    data = []
    
    for setting_name, setting_data in results["results"].items():
        setting_params = setting_data.get("setting_params", {})
        
        # Extract params for columns
        row = {
            "setting_name": setting_name
        }
        
        # Add parameters
        for param, value in setting_params.items():
            row[f"param_{param}"] = value
        
        # Add statistics
        for attr in attributes:
            if attr in setting_data["statistics"]:
                for metric in metrics:
                    if metric in setting_data["statistics"][attr]:
                        stats = setting_data["statistics"][attr][metric]
                        
                        # Add main statistics
                        for stat in ["mean", "std", "ci_lower", "ci_upper", "min", "max", "median"]:
                            if stat in stats:
                                row[f"{attr}_{metric}_{stat}"] = stats[stat]
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results exported to CSV: {filename}")


def export_to_markdown(results, filename, attributes=None, metrics=None):
    """
    Export results to Markdown format
    
    Args:
        results: Enhanced results dictionary
        filename: Output filename
        attributes: List of attributes to include (default: all)
        metrics: List of metrics to include (default: mAP and F1)
    """
    if attributes is None:
        attributes = results.get("metadata", {}).get("attributes", [])
    
    if metrics is None:
        metrics = ['overall_mAP', 'overall_f1']
    
    md_content = []
    md_content.append("# Experiment Results")
    md_content.append("")
    
    # Add metadata
    md_content.append("## Metadata")
    md_content.append("")
    md_content.append(f"- Timestamp: {results.get('metadata', {}).get('timestamp', 'Unknown')}")
    md_content.append(f"- Attributes: {', '.join(attributes)}")
    md_content.append("")
    
    # Overall results table
    md_content.append("## Overall Results")
    md_content.append("")
    
    # Table header
    header = ["Setting"]
    for attr in attributes:
        for metric in metrics:
            header.append(f"{attr}_{metric}")
    
    md_content.append("| " + " | ".join(header) + " |")
    md_content.append("| " + " | ".join(["---" for _ in header]) + " |")
    
    # Results rows
    for setting_name, setting_data in results["results"].items():
        setting_row = [setting_name]
        
        for attr in attributes:
            if attr in setting_data["statistics"]:
                for metric in metrics:
                    stats = setting_data["statistics"][attr].get(metric, {})
                    mean = stats.get("mean")
                    ci_lower = stats.get("ci_lower")
                    ci_upper = stats.get("ci_upper")
                    
                    if mean is not None:
                        if ci_lower is not None and ci_upper is not None:
                            setting_row.append(f"{mean:.2f} ({ci_lower:.2f}–{ci_upper:.2f})")
                        else:
                            setting_row.append(f"{mean:.2f}")
                    else:
                        setting_row.append("N/A")
        
        md_content.append("| " + " | ".join(setting_row) + " |")
    
    md_content.append("")
    
    # For ablation studies
    if "ablation_analysis" in results:
        reference = results["ablation_analysis"]["reference_setting"]
        effect_sizes = results["ablation_analysis"]["effect_sizes"]
        
        md_content.append("## Ablation Analysis")
        md_content.append("")
        md_content.append(f"Reference setting: {reference}")
        md_content.append("")
        
        # Group by varied parameter
        param_groups = defaultdict(list)
        for setting, data in effect_sizes.items():
            param = data.get('varied_parameter', 'unknown')
            param_groups[param].append((setting, data))
        
        for param, settings in param_groups.items():
            # Skip if only one setting with this parameter
            if len(settings) <= 1:
                continue
                
            md_content.append(f"### Parameter: {param}")
            md_content.append("")
            
            # Table header
            header = ["Setting"]
            for metric in metrics:
                header.extend([f"{metric}", "Effect Size", "p-value"])
            
            md_content.append("| " + " | ".join(header) + " |")
            md_content.append("| " + " | ".join(["---" for _ in header]) + " |")
            
            # Reference setting row
            ref_row = [f"{param}=Reference"]
            ref_results = results["results"][reference]
            
            for metric in metrics:
                for attr in attributes:
                    if attr in ref_results["statistics"]:
                        stats = ref_results["statistics"][attr].get(metric, {})
                        mean = stats.get("mean")
                        
                        if mean is not None:
                            ref_row.append(f"{mean:.2f}")
                        else:
                            ref_row.append("N/A")
                        
                        # Empty cells for effect size and p-value (reference)
                        ref_row.append("--")
                        ref_row.append("--")
            
            md_content.append("| " + " | ".join(ref_row) + " |")
            
            # Other settings
            for setting_name, data in settings:
                setting_val = setting_name.split(f"{param}=")[1].split("_")[0] if f"{param}=" in setting_name else "Unknown"
                row = [f"{param}={setting_val}"]
                
                for metric in metrics:
                    for attr in attributes:
                        if attr in data["effects"]:
                            effect_data = data["effects"][attr].get(metric, {})
                            comp_mean = effect_data.get("comparison_mean")
                            effect = effect_data.get("effect_size")
                            p_val = effect_data.get("p_value")
                            
                            # Append mean
                            if comp_mean is not None:
                                row.append(f"{comp_mean:.2f}")
                            else:
                                row.append("N/A")
                            
                            # Append effect size
                            if effect is not None:
                                # Add significance markers
                                effect_str = f"{effect:.2f}"
                                if p_val is not None and p_val < 0.05:
                                    effect_str += "*"
                                if p_val is not None and p_val < 0.01:
                                    effect_str += "*"
                                if p_val is not None and p_val < 0.001:
                                    effect_str += "*"
                                row.append(effect_str)
                            else:
                                row.append("--")
                            
                            # Append p-value
                            if p_val is not None:
                                if p_val < 0.001:
                                    row.append("<0.001")
                                else:
                                    row.append(f"{p_val:.3f}")
                            else:
                                row.append("--")
                
                md_content.append("| " + " | ".join(row) + " |")
            
            md_content.append("")
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    
    print(f"Markdown report saved to {filename}")


def process_and_export_results(original_results, settings, attributes, base_filename):
    """
    Process results and export to multiple formats
    
    Args:
        original_results: Original results dictionary
        settings: Settings used for the experiments
        attributes: List of attributes analyzed
        base_filename: Base name for output files
    """
    # Enhance results with statistics
    enhanced = enhance_results(original_results, settings, attributes)
    
    # Save enhanced JSON
    json_filename = f"{base_filename}.json"
    save_enhanced_results(enhanced, json_filename)
    
    # Export to CSV
    csv_filename = f"{base_filename}.csv"
    export_to_csv(enhanced, csv_filename)
    
    # Export to LaTeX
    latex_base = f"{base_filename}"
    export_to_latex(enhanced, latex_base + ".tex")
    
    # Export to Markdown
    md_filename = f"{base_filename}.md"
    export_to_markdown(enhanced, md_filename)
    
    print(f"All exports completed successfully")
    return enhanced