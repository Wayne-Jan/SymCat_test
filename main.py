# main.py
# Main entry point for CPRFL_SymCat project

import os
# Set tokenizers parallelism to false to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import json
import copy
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

# Import components
from src.models.model import MultiCPRFL, VisualSemanticInteraction, MultiPromptInitializer, FiLMFusion
# Note: compute_multilabel_metrics is not available in model_components
from src.models.baseline import BaselineClassifier
from src.data_loader.data_utils import load_index, convert_to_multihot, compute_head_medium_tail, attributes
from src.helper_functions.train_eval import train_multi_cprfl, initialize_multi_category_embeddings, data_key_map
from src.helper_functions.train_eval import evaluate_multi_cprfl
from src.helper_functions.train_baseline import train_baseline_classifier, evaluate_baseline_classifier
from src.helper_functions.zero_shot import evaluate_clip_zero_shot
from src.loss_functions.loss import get_loss_function, get_contrastive_loss
from src.utils.util import create_progress_bar

# Import configuration
import config

console = Console()

class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy data types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def gen_setting_name(s):
    """
    Generate a unique setting name string based on the settings dictionary.
    """
    name = (
        f"VSI={s.get('use_vsi', 'Def')}_Fusion={s.get('fusion_mode', 'Def')}_Loss={s.get('loss_type', 'Def')}_"
        f"CLT={s.get('contrastive_loss_type', 'Def')}_L={s.get('lambda_start', 'Def')}-{s.get('lambda_end', 'Def')}_"
        f"B={s.get('beta_prompt_separation', 'Def')}_K={s.get('contrastive_top_k_negatives', 'None')}_"
        f"PB={s.get('positive_weighting_beta', 'None')}"
    )
    return name

def apply_setting(base, key, val):
    """
    Modify a specific key-value in the base setting, update related settings if needed,
    and generate a new setting name.
    """
    new_s = base.copy()
    new_s[key] = val

    # Update related settings
    if key == "contrastive_top_k_negatives" and val is not None:
        new_s["contrastive_loss_type"] = "image_text_v2"
    if key == "positive_weighting_beta" and val is not None:
        new_s["contrastive_loss_type"] = "image_text_v2"
    if key == "lambda_contrastive_weight":
        new_s["lambda_start"] = val
        new_s["lambda_end"] = val
        new_s["lambda_schedule"] = "linear"

    new_s["setting_name"] = gen_setting_name(new_s)
    return new_s

def calculate_average_metrics(results, attributes):
    """
    Calculate average metrics across multiple repetitions.
    """
    avg_metrics = {}
    for attr in attributes:
        attr_metrics = {
            "overall_mAP": {"values": [], "mean": None, "std": None},
            "overall_f1": {"values": [], "mean": None, "std": None},
            "overall_acc": {"values": [], "mean": None, "std": None},
            "head_mAP": {"values": [], "mean": None, "std": None},
            "medium_mAP": {"values": [], "mean": None, "std": None},
            "tail_mAP": {"values": [], "mean": None, "std": None},
            "head_f1": {"values": [], "mean": None, "std": None},
            "medium_f1": {"values": [], "mean": None, "std": None},
            "tail_f1": {"values": [], "mean": None, "std": None}
        }

        for rep_result in results:
            if attr not in rep_result.get("results", {}):
                continue

            rep_data = rep_result["results"][attr]

            # Process metrics
            if "overall_mAP" in rep_data:
                attr_metrics["overall_mAP"]["values"].append(rep_data["overall_mAP"])
            if "overall_acc" in rep_data:
                attr_metrics["overall_acc"]["values"].append(rep_data["overall_acc"])
            if "overall_f1" in rep_data:
                attr_metrics["overall_f1"]["values"].append(rep_data["overall_f1"])

            # Process type metrics (head/medium/tail)
            if "type_metrics" in rep_data and isinstance(rep_data["type_metrics"], dict):
                type_metrics = rep_data["type_metrics"]
                for type_key in ["head", "medium", "tail"]:
                    if type_key in type_metrics:
                        if "mAP" in type_metrics[type_key]:
                            attr_metrics[f"{type_key}_mAP"]["values"].append(type_metrics[type_key]["mAP"])
                        if "f1" in type_metrics[type_key]:
                            attr_metrics[f"{type_key}_f1"]["values"].append(type_metrics[type_key]["f1"])

        # Calculate means and std devs
        for metric_key, metric_data in attr_metrics.items():
            values = metric_data["values"]
            # Filter out None values before calculating statistics
            valid_values = [v for v in values if v is not None]
            if valid_values:
                metric_data["mean"] = np.mean(valid_values)
                metric_data["std"] = np.std(valid_values)
            else:
                metric_data["mean"] = None
                metric_data["std"] = None

        avg_metrics[attr] = attr_metrics

    return avg_metrics

def display_average_results(metrics, attributes):
    """
    Display average results in a table format.
    """
    for attr in attributes:
        if attr not in metrics:
            continue

        table = Table(title=f"{attr} Metrics")
        table.add_column("Metric")
        table.add_column("Mean")
        table.add_column("Std Dev")

        attr_metrics = metrics[attr]
        metric_order = [
            "overall_mAP", "overall_f1", "overall_acc",
            "head_mAP", "medium_mAP", "tail_mAP",
            "head_f1", "medium_f1", "tail_f1",
        ]

        for metric_key in metric_order:
            if metric_key in attr_metrics:
                metric_data = attr_metrics[metric_key]
                if metric_data["mean"] is not None:
                    mean_str = f"{metric_data['mean']:.2f}%"
                    std_str = f"±{metric_data['std']:.2f}" if metric_data["std"] is not None else "N/A"
                    table.add_row(metric_key, mean_str, std_str)

        console.print(table)

def process_label_tensor(raw_labels_or_list, attr, idx_to_label_dict, label_to_idx_dict):
    """
    Process label tensor or list and convert to multi-hot encoding.
    Can handle items that are single IDs or lists of IDs.
    """
    if attr not in label_to_idx_dict or attr not in idx_to_label_dict:
        return torch.empty((0, 0), dtype=torch.long)

    num_classes = len(label_to_idx_dict[attr])
    if num_classes == 0:
        if isinstance(raw_labels_or_list, list):
            return torch.empty((len(raw_labels_or_list), 0), dtype=torch.long)
        else:
            return torch.empty((0, 0), dtype=torch.long)

    multi_hot_list = []
    if isinstance(raw_labels_or_list, torch.Tensor):
        label_list = raw_labels_or_list.tolist()
    elif isinstance(raw_labels_or_list, list):
        label_list = raw_labels_or_list
    else:
        return torch.empty((0, num_classes), dtype=torch.long)

    for label_item in label_list:
        vec = torch.zeros(num_classes, dtype=torch.long)
        if isinstance(label_item, list):
            current_labels = label_item
        elif label_item is not None:
            current_labels = [label_item]
        else:
            current_labels = []

        processed_ids = set()
        for label_id in current_labels:
            label_id_str = str(label_id)
            found_label_str = None

            # Try with original key
            if label_id in idx_to_label_dict[attr]:
                found_label_str = idx_to_label_dict[attr][label_id]
            elif label_id_str in idx_to_label_dict[attr]:
                found_label_str = idx_to_label_dict[attr][label_id_str]
                label_id = label_id_str

            if found_label_str is not None:
                if found_label_str not in processed_ids:
                    try:
                        single_hot = convert_to_multihot(found_label_str, label_to_idx_dict[attr])
                        vec = vec | single_hot
                        processed_ids.add(found_label_str)
                    except KeyError:
                        pass
                    except Exception as e:
                        console.print(f"[red]Error: Processing label '{found_label_str}': {e}[/red]")
        multi_hot_list.append(vec)

    if not multi_hot_list:
        return torch.empty((0, num_classes), dtype=torch.long)

    try:
        return torch.stack(multi_hot_list, dim=0)
    except Exception as e:
        console.print(f"[red]Error: Stacking multi-hot vectors: {e}[/red]")
        return torch.empty((0, num_classes), dtype=torch.long)

def main():
    """Main training function with interactive options"""
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Using device: {device}[/bold]")

    # 1) Choose model type
    console.print("[bold]Choose model type or evaluation method:[/bold]")
    console.print("1) Baseline classifier")
    console.print("2) Multi-modal model (MultiCPRFL)")
    console.print("3) Zero-Shot CLIP evaluation")
    model_type = input(">> ")

    is_baseline = (model_type == "1")
    is_zero_shot = (model_type == "3")

    if is_zero_shot:
        console.print("[bold blue]Will use Zero-Shot CLIP for direct evaluation (no training needed)[/bold blue]")
    elif is_baseline:
        console.print("[bold blue]Will train a pure image-based baseline classifier[/bold blue]")
    else:
        console.print("[bold blue]Will train a complete multi-modal model (MultiCPRFL)[/bold blue]")

    # Default training parameters
    default_epochs = 150
    default_repetitions = 3
    default_patience = 30  # Increased default patience
    default_map_weight = 0.5
    hidden_dim = 768

    # Load CLIP model for text embeddings
    try:
        clip_model = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
        clip_model.eval()
        console.print(f"[bold green]Loaded Jina-CLIP-v2 model[/bold green]")
    except Exception as e:
        console.print(f"[red]Error loading SentenceTransformer: {e}[/red]")
        return

    # Load data
    console.print("[bold]Loading data...[/bold]")
    label_to_idx_dict, idx_to_label_dict, index_data = load_index(config.symcat_json_path)

    # Calculate head/medium/tail statistics
    head_medium_tail = compute_head_medium_tail(config.nchu_sorted_data_path)

    # Load embeddings
    train_data = torch.load(config.train_embeddings_path)
    val_data = torch.load(config.val_embeddings_path)
    test_data = torch.load(config.test_embeddings_path)
    
    console.print(f"[bold green]Loaded embeddings: Train={train_data['image_embeddings'].size(0)}, "
                 f"Val={val_data['image_embeddings'].size(0)}, Test={test_data['image_embeddings'].size(0)}[/bold green]")
    
    # Process labels for each attribute
    console.print("[bold]Processing labels...[/bold]")
    train_labels_dict = {}
    val_labels_dict = {}
    test_labels_dict = {}
    
    for attr in attributes:
        console.print(f"  Processing {attr} labels...")
        key_name = data_key_map.get(attr, f"{attr}_labels")

        # Train data
        if key_name not in train_data:
            console.print(f"[red]Error: Training data missing key '{key_name}' (for attribute '{attr}')")
            return
        train_labels_dict[attr] = process_label_tensor(train_data[key_name], attr, idx_to_label_dict, label_to_idx_dict)

        # Validation data
        if key_name not in val_data:
            console.print(f"[red]Error: Validation data missing key '{key_name}' (for attribute '{attr}')")
            return
        val_labels_dict[attr] = process_label_tensor(val_data[key_name], attr, idx_to_label_dict, label_to_idx_dict)

        # Test data
        if key_name not in test_data:
            console.print(f"[red]Error: Test data missing key '{key_name}' (for attribute '{attr}')")
            return
        test_labels_dict[attr] = process_label_tensor(test_data[key_name], attr, idx_to_label_dict, label_to_idx_dict)
    
    # Set training parameters based on model type
    if is_zero_shot:
        # Implement Zero-Shot evaluation
        console.print("[bold]Running Zero-Shot CLIP evaluation...[/bold]")
        # Implement zero-shot evaluation
        results = evaluate_clip_zero_shot(
            attributes=attributes,
            label_to_idx_dict=label_to_idx_dict,
            idx_to_label_dict=idx_to_label_dict,
            test_data=test_data,
            test_labels_dict=test_labels_dict,
            index_data=index_data,
            clip_model=clip_model,
            device=device
        )
        
        # Create directory for results if it doesn't exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare the results for JSON export
        setting_name = "zero_shot_clip"
        setting_params = {
            "model_type": "zero_shot",
            "description": "Zero-Shot CLIP evaluation without any training"
        }
        
        # Structure the results like the other options
        result_entry = {"repetition": 1, "results": results}
        setting_results_list = [result_entry]
        
        # Calculate metrics
        avg_metrics = calculate_average_metrics(setting_results_list, attributes)
        
        # Create final results structure
        final_results_to_save = {
            setting_name: {
                "setting_params": setting_params,
                "avg_metrics": avg_metrics,
                "detailed_results": setting_results_list
            }
        }
        
        # Save results to JSON file
        results_file = f"{results_dir}/zero_shot_clip_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results_to_save, f, indent=2, cls=NumpyEncoder)
        console.print(f"[green]Zero-shot evaluation results saved to {results_file}[/green]")
        
        return
    elif is_baseline:
        # Setup baseline classifier parameters
        console.print("[bold]Select loss function for baseline classifier:[/bold]")
        console.print("1) BCE (Binary Cross Entropy)")
        console.print("2) ASL (Asymmetric Loss)")
        loss_choice = input(">> ")

        if loss_choice == "2":
            loss_type = "asl"
            console.print("[bold]ASL parameters (can use defaults):[/bold]")
            gamma_neg_str = input("[bold]gamma_neg [default: 4]:[/bold] >> ")
            gamma_neg = float(gamma_neg_str) if gamma_neg_str else 4.0
            
            gamma_pos_str = input("[bold]gamma_pos [default: 0]:[/bold] >> ")
            gamma_pos = float(gamma_pos_str) if gamma_pos_str else 0.0
            
            clip_str = input("[bold]clip value [default: 0.05]:[/bold] >> ")
            clip_value = float(clip_str) if clip_str else 0.05
        else:
            loss_type = "bce"
            gamma_neg = 4.0
            gamma_pos = 0.0
            clip_value = 0.05

        # Get epochs for baseline training
        epochs_str = input(f"[bold]Number of epochs [default: {default_epochs}]:[/bold] >> ")
        epochs = int(epochs_str) if epochs_str else default_epochs
        epochs = max(1, epochs)
        
        # Ask for number of repetitions
        console.print(f"[bold]Number of repetitions [default: {default_repetitions}]:[/bold] (How many times to repeat training for statistical significance)")
        repetitions_str = input(">> ")
        repetitions = int(repetitions_str) if repetitions_str else default_repetitions
        repetitions = max(1, repetitions)

        # Get class counts for baseline classifier
        num_classes_dict = {attr: len(label_to_idx_dict[attr]) for attr in attributes}

        # Implement baseline classifier training
        console.print("[bold]Training baseline classifier...[/bold]")
        # Create BaselineClassifier model
        model = BaselineClassifier(
            attributes=attributes,
            num_classes_dict=num_classes_dict,
            image_input_dim=train_data['image_embeddings'].size(1),
            hidden_dim=hidden_dim
        )
        model = model.to(device)

        # Get early stopping configuration
        patience_str = input(f"[bold]Early Stopping Patience [default: {default_patience}]:[/bold] (Number of epochs to wait for improvement before stopping) >> ")
        early_stopping_patience_value = int(patience_str) if patience_str else default_patience
        early_stopping_patience_value = max(0, early_stopping_patience_value)

        console.print(f"[bold]mAP weight (0.0-1.0) [default: {default_map_weight}]:[/bold] (Weight of mAP in the combined score; 0.0 = only F1, 1.0 = only mAP)")
        map_weight_str = input(">> ")
        try:
            map_weight_value = float(map_weight_str) if map_weight_str else default_map_weight
            map_weight_value = max(0.0, min(1.0, map_weight_value))
        except ValueError:
            map_weight_value = default_map_weight
            console.print(f"[yellow]Invalid value, using default {default_map_weight}[/yellow]")

        # Default to yes for combined score
        console.print(f"[bold]Use combined score (mAP + F1) for model selection? (y/n) [default: y]:[/bold] (If 'y', save model with best combined score; if 'n', save model with best mAP only)")
        save_best_combined_str = input(">> ")
        save_best_combined_value = True
        if save_best_combined_str and save_best_combined_str.lower().startswith('n'):
            save_best_combined_value = False

        if save_best_combined_value:
            console.print(f"Model selection strategy: Combined score (mAP weight: {map_weight_value:.2f}, F1 weight: {1 - map_weight_value:.2f})")
        else:
            console.print("Model selection strategy: mAP only")
            
        # Create setting name and params for the baseline model
        setting_name = f"baseline_{loss_type}_e{epochs}"
        setting_params = {
            "model_type": "baseline",
            "loss_type": loss_type,
            "epochs": epochs,
            "repetitions": repetitions,
            "early_stopping_patience": early_stopping_patience_value,
            "map_weight": map_weight_value,
            "save_best_combined": save_best_combined_value,
            "gamma_neg": gamma_neg if loss_type == "asl" else None,
            "gamma_pos": gamma_pos if loss_type == "asl" else None,
            "clip_value": clip_value if loss_type == "asl" else None
        }
        
        # Create directory for results if it doesn't exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize list to store results for each repetition
        setting_results_list = []
        
        # Run for multiple repetitions
        for rep in range(1, repetitions + 1):
            console.print(f"[bold cyan]-- Baseline Classifier, Repetition {rep}/{repetitions} --[/bold cyan]")
            
            # Create a new model instance for each repetition
            model = BaselineClassifier(
                attributes=attributes,
                num_classes_dict=num_classes_dict,
                image_input_dim=train_data['image_embeddings'].size(1),
                hidden_dim=hidden_dim
            )
            model = model.to(device)
            
            # Train model
            console.print(f"[bold]Training baseline classifier (rep {rep}/{repetitions})...[/bold]")
            trained_model = train_baseline_classifier(
                model=model,
                attributes=attributes,
                train_data=train_data,
                val_data=val_data,
                train_labels_dict=train_labels_dict,
                val_labels_dict=val_labels_dict,
                label_to_idx_dict=label_to_idx_dict,
                idx_to_label_dict=idx_to_label_dict,
                device=device,
                epochs=epochs,
                batch_size=config.batch_size,
                loss_type=loss_type,
                lr=config.learning_rate,
                gamma_neg=gamma_neg,
                gamma_pos=gamma_pos,
                clip_value=clip_value,
                early_stopping_patience=early_stopping_patience_value,
                map_weight=map_weight_value,
                save_best_combined=save_best_combined_value,
                test_data=test_data,
                test_labels_dict=test_labels_dict
            )
            
            # Evaluate on test set
            console.print(f"[bold]Evaluating baseline classifier (rep {rep}/{repetitions}) on test set...[/bold]")
            test_results = evaluate_baseline_classifier(
                model=trained_model,
                attributes=attributes,
                test_data=test_data,
                test_labels_dict=test_labels_dict,
                label_to_idx_dict=label_to_idx_dict,
                idx_to_label_dict=idx_to_label_dict,
                device=device,
                batch_size=config.batch_size
            )
            
            # Save model for this repetition
            model_save_path = f"{results_dir}/baseline_model_{loss_type}_e{epochs}_rep{rep}.pth"
            torch.save(trained_model.state_dict(), model_save_path)
            console.print(f"[green]Model saved to {model_save_path}[/green]")
            
            # Store this repetition's results
            result_entry = {"repetition": rep, "results": test_results}
            setting_results_list.append(result_entry)
        
        # Calculate average metrics across repetitions
        avg_metrics = calculate_average_metrics(setting_results_list, attributes)
        
        # Format and display the average results
        console.print(f"\n[bold magenta]===== Baseline Classifier Results ({repetitions} repetitions) =====\n[/bold magenta]")
        display_average_results(avg_metrics, attributes)
        
        # Store results in the same format as multi-modal
        final_results_to_save = {
            setting_name: {
                "setting_params": setting_params,
                "avg_metrics": avg_metrics,
                "detailed_results": setting_results_list
            }
        }
        
        # Save results to JSON file
        results_file = f"{results_dir}/baseline_{loss_type}_e{epochs}_r{repetitions}_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results_to_save, f, indent=2, cls=NumpyEncoder)
        console.print(f"[green]All results saved to {results_file}[/green]")
        
        return
    else:
        # Setup multi-modal model parameters
        default_lambda_start = 0.0  # Changed to 0 as default
        default_lambda_end = 0.0    # Changed to 0 as default
        default_lambda_schedule = "linear"
        contrastive_loss_types_available = ["supcon", "mse", "image_text_v2", "prototype"]
        default_contrastive_loss_type = "image_text_v2"
        default_beta_prompt_separation = 0.0
        default_top_k_negatives = None
        default_positive_weighting_beta = 0.999
        default_loss_type = "bce"   # Changed default to BCE

        # Ask for contrastive loss type
        console.print(f"[bold]Choose contrastive loss (1: supcon, 2: mse, 3: image_text_v2, 4: prototype) [default: {default_contrastive_loss_type}]:[/bold]")
        clt_str = input(">> ")
        clt_map = {
            "1": "supcon",
            "2": "mse",
            "3": "image_text_v2",
            "4": "prototype"
        }
        selected_contrastive_loss_type = clt_map.get(clt_str, default_contrastive_loss_type)

        # Ask for ablation
        console.print("[bold]Run ablation test? (y/n)[/bold]")
        do_ablation = input(">> ").lower().startswith('y')

        # Initialize default parameters
        lambda_contrastive_start = default_lambda_start
        lambda_contrastive_end = default_lambda_end
        lambda_schedule = default_lambda_schedule
        beta_prompt_separation_value = default_beta_prompt_separation
        top_k_negatives_value = default_top_k_negatives
        positive_weighting_beta_value = default_positive_weighting_beta
        use_vsi = True
        fusion_mode = "film"
        loss_type = default_loss_type  # Using BCE as default
        contrastive_loss_type = selected_contrastive_loss_type
        repetitions = default_repetitions  # Initialize repetitions variable

        # Initialize experiment settings
        experiment_settings = []

        # Define the ablation key mapping
        ablation_key_map = {
            "VSI": "use_vsi",
            "Fusion Mode": "fusion_mode",
            "Loss Type": "loss_type",
            "Contrastive Weight": "lambda_contrastive_weight",
            "Contrastive Loss Type": "contrastive_loss_type",
            "Prompt Separation Weight": "beta_prompt_separation",
            "Hard Negative K": "contrastive_top_k_negatives",
            "Positive Weighting Beta": "positive_weighting_beta"
        }

        if do_ablation:
            console.print("[bold]Choose ablation parameter:[/bold]")
            console.print("1) VSI")
            console.print("2) Fusion Mode")
            console.print("3) Loss Type (Classification)")
            console.print("4) Contrastive Weight (Lambda - fixed value)")
            console.print("5) Contrastive Loss Type")
            console.print("6) Prompt Separation Weight (Beta)")
            console.print("7) Hard Negative K (for image_text_v2)")
            console.print("8) Positive Weighting Beta (for image_text_v2)")
            console.print("9) Contrastive Learning Comparison (Loss Type + Weight)")
            ablation_choice = input(">> ")

            # Set parameter name and values based on choice
            param_name = None
            param_values = []

            if ablation_choice == "1":
                param_name = "VSI"
                param_values = [True, False]
            elif ablation_choice == "2":
                param_name = "Fusion Mode"
                param_values = ["random", "clip", "concat", "film"]
            elif ablation_choice == "3":
                param_name = "Loss Type"
                param_values = ["bce", "asl", "dbfocal", "mls"]
            elif ablation_choice == "4":
                param_name = "Contrastive Weight"
                param_values = [0.0, 0.1, 0.3, 0.5, 1.0]
            elif ablation_choice == "5":
                param_name = "Contrastive Loss Type"
                param_values = ["supcon", "mse", "image_text_v2", "prototype"]
            elif ablation_choice == "6":
                param_name = "Prompt Separation Weight"
                param_values = [0.0, 0.5, 1.0]
            elif ablation_choice == "7":
                param_name = "Hard Negative K"
                param_values = [None, 5, 10, 20]
                if selected_contrastive_loss_type not in ["image_text_v2"]:
                    console.print("[yellow]Warning: Hard Negative K typically works with image_text_v2.[/yellow]")
            elif ablation_choice == "8":
                param_name = "Positive Weighting Beta"
                param_values = [None, 0.9, 0.99, 0.999, 0.9999]
                if selected_contrastive_loss_type not in ["image_text_v2"]:
                    console.print("[yellow]Warning: Positive Weighting Beta typically works with image_text_v2.[/yellow]")
            elif ablation_choice == "9":
                param_name = "Contrastive Loss Type"
                param_values = ["supcon", "image_text_v2", "prototype"]
                console.print("[bold green]Running focused contrastive loss comparison...[/bold green]")
                console.print("[bold]Testing the three most promising contrastive loss types with fixed parameters:[/bold]")
                console.print("1. SupCon: Standard supervised contrastive loss")
                console.print("2. Image-Text V2: Cross-modal contrastive with hard negative mining")
                console.print("3. Prototype: Memory-bank approach with class prototype comparisons")
            else:
                console.print("[red]Invalid choice, will use default values.[/red]")
                do_ablation = False

            # Ask for other parameters to create the base settings
            console.print("[bold]Set base parameters for all ablation experiments:[/bold]")

            if ablation_choice == "9":
                # For contrastive loss comparison, use fixed parameters (VSI=True, FiLM fusion, ASL loss)
                console.print("[bold cyan]Using recommended fixed parameters for contrastive loss comparison:[/bold cyan]")
                console.print("- VSI: True (Visual-Semantic Interaction enabled)")
                console.print("- Fusion Mode: film (FiLM fusion)")
                console.print("- Loss Type: asl (Asymmetric Loss)")
                console.print("- Contrastive Weight: 0.3 (balanced weight)")

                use_vsi = True
                fusion_mode = "film"
                loss_type = "asl"
                lambda_contrastive_start = 0.3
                lambda_contrastive_end = 0.3
            else:
                # Standard parameter selection for other ablation types
                if param_name != "VSI":
                    vsi_in = input(f"[bold]Use VSI (y/n) [default: y]:[/bold] >> ")
                    use_vsi = False if vsi_in.lower().startswith('n') else True

                if param_name != "Fusion Mode":
                    fm_map = {"1": "random", "2": "clip", "3": "concat", "4": "film"}
                    fm_in = input(f"[bold]Fusion Mode (1: random, 2: clip, 3: concat, 4: film) [default: film]:[/bold] >> ")
                    fusion_mode = fm_map.get(fm_in, "film")

                if param_name != "Loss Type":
                    loss_map = {"1": "bce", "2": "asl", "3": "dbfocal", "4": "mls"}
                    loss_in = input(f"[bold]Loss type (1: bce, 2: asl, 3: dbfocal, 4: mls) [default: {default_loss_type}]:[/bold] >> ")
                    loss_type = loss_map.get(loss_in, default_loss_type)

                if param_name != "Contrastive Weight":
                    start_lambda_str = input(f"[bold]Contrastive start weight [default: {default_lambda_start}]:[/bold] >> ")
                    try:
                        lambda_contrastive_start = float(start_lambda_str) if start_lambda_str else default_lambda_start
                    except ValueError:
                        lambda_contrastive_start = default_lambda_start

                    end_lambda_str = input(f"[bold]Contrastive end weight [default: {default_lambda_end}]:[/bold] >> ")
                    try:
                        lambda_contrastive_end = float(end_lambda_str) if end_lambda_str else default_lambda_end
                    except ValueError:
                        lambda_contrastive_end = default_lambda_end

            if ablation_choice == "9":
                # For contrastive loss comparison, use optimized fixed parameters for Hard Neg K and Positive Beta
                beta_prompt_separation_value = 0.0
                # For Image-Text V2, we'll use the optimal parameters in the ablation
                top_k_negatives_value = 10 if selected_contrastive_loss_type == "image_text_v2" else None
                positive_weighting_beta_value = 0.999 if selected_contrastive_loss_type == "image_text_v2" else None
                console.print("- Prompt Separation Beta: 0.0")
                console.print("- Hard Negative K: 10 (for Image-Text V2)")
                console.print("- Positive Weighting Beta: 0.999 (for Image-Text V2)")
            else:
                # Standard parameter selection for other ablation types
                if param_name != "Prompt Separation Weight":
                    beta_str = input(f"[bold]Prompt separation weight Beta [default: {default_beta_prompt_separation}]:[/bold] >> ")
                    try:
                        beta_prompt_separation_value = float(beta_str) if beta_str else default_beta_prompt_separation
                    except ValueError:
                        beta_prompt_separation_value = default_beta_prompt_separation

                if param_name != "Hard Negative K" and param_name != "Positive Weighting Beta" and selected_contrastive_loss_type == "image_text_v2":
                    k_str = input(f"[bold]Hard Negative K (enter number or leave empty for all) [default: None]:[/bold] >> ")
                    try:
                        top_k_negatives_value = int(k_str) if k_str else default_top_k_negatives
                    except ValueError:
                        top_k_negatives_value = default_top_k_negatives
                    if top_k_negatives_value is not None and top_k_negatives_value <= 0:
                        top_k_negatives_value = None

                    pos_beta_str = input(f"[bold]Positive weighting Beta (0.9-0.9999, or empty for none) [default: {default_positive_weighting_beta}]:[/bold] >> ")
                    try:
                        positive_weighting_beta_value = float(pos_beta_str) if pos_beta_str else default_positive_weighting_beta
                    except ValueError:
                        positive_weighting_beta_value = default_positive_weighting_beta

            # Create base setting
            base_setting = {
                "use_vsi": use_vsi,
                "fusion_mode": fusion_mode,
                "loss_type": loss_type,
                "contrastive_loss_type": contrastive_loss_type,
                "lambda_start": lambda_contrastive_start,
                "lambda_end": lambda_contrastive_end,
                "lambda_schedule": lambda_schedule,
                "beta_prompt_separation": beta_prompt_separation_value,
                "contrastive_top_k_negatives": top_k_negatives_value,
                "positive_weighting_beta": positive_weighting_beta_value
            }

            # Generate name for the base setting
            base_setting["setting_name"] = gen_setting_name(base_setting)

            # Create experiment settings for ablation
            if param_name in ablation_key_map:
                ablation_key = ablation_key_map[param_name]
                for val in param_values:
                    setting = apply_setting(base_setting, ablation_key, val)
                    experiment_settings.append(setting)
                    console.print(f"[blue]Added ablation setting: {setting['setting_name']}[/blue]")
            else:
                console.print(f"[red]Unknown ablation parameter '{param_name}', will use base settings.[/red]")
                experiment_settings = [base_setting]
        else:
            # Ask for individual parameters when not in ablation mode
            vsi_in = input(f"[bold]Use VSI (y/n) [default: y]:[/bold] >> ")
            use_vsi = False if vsi_in.lower().startswith('n') else True

            fm_map = {"1": "random", "2": "clip", "3": "concat", "4": "film"}
            fm_in = input(f"[bold]Fusion Mode (1: random, 2: clip, 3: concat, 4: film) [default: film]:[/bold] >> ")
            fusion_mode = fm_map.get(fm_in, "film")

            loss_map = {"1": "bce", "2": "asl", "3": "dbfocal", "4": "mls"}
            loss_in = input(f"[bold]Loss type (1: bce, 2: asl, 3: dbfocal, 4: mls) [default: {default_loss_type}]:[/bold] >> ")
            loss_type = loss_map.get(loss_in, default_loss_type)

            start_lambda_str = input(f"[bold]Contrastive start weight [default: {default_lambda_start}]:[/bold] >> ")
            try:
                lambda_contrastive_start = float(start_lambda_str) if start_lambda_str else default_lambda_start
            except ValueError:
                lambda_contrastive_start = default_lambda_start

            end_lambda_str = input(f"[bold]Contrastive end weight [default: {default_lambda_end}]:[/bold] >> ")
            try:
                lambda_contrastive_end = float(end_lambda_str) if end_lambda_str else default_lambda_end
            except ValueError:
                lambda_contrastive_end = default_lambda_end

            schedule_str = input(f"[bold]Weight schedule (1:lin, 2:cos, 3:exp) [default: 1]:[/bold] >> ")
            schedule_map = {"1": "linear", "2": "cosine", "3": "exp"}
            lambda_schedule = schedule_map.get(schedule_str, "linear")

            beta_str = input(f"[bold]Prompt separation weight Beta [default: {default_beta_prompt_separation}]:[/bold] >> ")
            try:
                beta_prompt_separation_value = float(beta_str) if beta_str else default_beta_prompt_separation
            except ValueError:
                beta_prompt_separation_value = default_beta_prompt_separation

            if selected_contrastive_loss_type == 'image_text_v2':
                k_str = input(f"[bold]Hard Negative K (enter number or leave empty for all) [default: None]:[/bold] >> ")
                try:
                    top_k_negatives_value = int(k_str) if k_str else default_top_k_negatives
                except ValueError:
                    top_k_negatives_value = default_top_k_negatives
                if top_k_negatives_value is not None and top_k_negatives_value <= 0:
                    top_k_negatives_value = None

                pos_beta_str = input(f"[bold]Positive weighting Beta (0.9-0.9999, or empty for none) [default: {default_positive_weighting_beta}]:[/bold] >> ")
                try:
                    positive_weighting_beta_value = float(pos_beta_str) if pos_beta_str else default_positive_weighting_beta
                except ValueError:
                    positive_weighting_beta_value = default_positive_weighting_beta

            # Create a single setting for non-ablation mode
            single_setting = {
                "use_vsi": use_vsi,
                "fusion_mode": fusion_mode,
                "loss_type": loss_type,
                "contrastive_loss_type": contrastive_loss_type,
                "lambda_start": lambda_contrastive_start,
                "lambda_end": lambda_contrastive_end,
                "lambda_schedule": lambda_schedule,
                "beta_prompt_separation": beta_prompt_separation_value,
                "contrastive_top_k_negatives": top_k_negatives_value,
                "positive_weighting_beta": positive_weighting_beta_value
            }

            # Generate name for the setting
            single_setting["setting_name"] = gen_setting_name(single_setting)
            experiment_settings = [single_setting]

        # Common training parameters
        epochs_str = input(f"[bold]Number of epochs [default: {default_epochs}]:[/bold] >> ")
        epochs = int(epochs_str) if epochs_str else default_epochs
        epochs = max(1, epochs)

        # Ask for number of repetitions (moved here after epochs)
        console.print(f"[bold]Number of repetitions [default: {default_repetitions}]:[/bold] (How many times to repeat each experiment for statistical significance)")
        repetitions_str = input(">> ")
        repetitions = int(repetitions_str) if repetitions_str else default_repetitions
        repetitions = max(1, repetitions)

        patience_str = input(f"[bold]Early Stopping Patience [default: {default_patience}]:[/bold] (Number of epochs to wait for improvement before stopping) >> ")
        early_stopping_patience_value = int(patience_str) if patience_str else default_patience
        early_stopping_patience_value = max(0, early_stopping_patience_value)

        console.print(f"[bold]mAP weight (0.0-1.0) [default: {default_map_weight}]:[/bold] (Weight of mAP in the combined score; 0.0 = only F1, 1.0 = only mAP)")
        map_weight_str = input(">> ")
        try:
            map_weight_value = float(map_weight_str) if map_weight_str else default_map_weight
            map_weight_value = max(0.0, min(1.0, map_weight_value))
        except ValueError:
            map_weight_value = default_map_weight
            console.print(f"[yellow]Invalid value, using default {default_map_weight}[/yellow]")

        # Default to yes for combined score
        console.print(f"[bold]Use combined score (mAP + F1) for model selection? (y/n) [default: y]:[/bold] (If 'y', save model with best combined score; if 'n', save model with best mAP only)")
        save_best_combined_str = input(">> ")
        save_best_combined_value = True
        if save_best_combined_str and save_best_combined_str.lower().startswith('n'):
            save_best_combined_value = False

        if save_best_combined_value:
            console.print(f"Model selection strategy: Combined score (mAP weight: {map_weight_value:.2f}, F1 weight: {1 - map_weight_value:.2f})")
        else:
            console.print("Model selection strategy: mAP only")

        # Initialize text embeddings
        categories_dict = {attr: list(label_to_idx_dict[attr].keys()) for attr in attributes}
        text_embeddings_dict = initialize_multi_category_embeddings(
            attributes, categories_dict, clip_model, device, index_data
        )

        # Get class counts for each attribute
        num_classes_dict = {attr: len(label_to_idx_dict[attr]) for attr in attributes}

        # Create results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # Store final results
        final_results_to_save = {}

        # Run experiments
        console.print(f"\n[blue]Will run {len(experiment_settings)} experiment settings.[/blue]")

        for setting in experiment_settings:
            current_setting_name = setting["setting_name"]
            console.print(f"\n[bold cyan]===== Starting setting: {current_setting_name} =====\n[/bold cyan]")
            setting_results_list = []

            # Run each setting for the specified number of repetitions
            for rep in range(1, repetitions + 1):
                console.print(f"[bold cyan]-- Setting: {current_setting_name}, Repetition {rep}/{repetitions} --[/bold cyan]")

                # Create model for this repetition
                console.print("[bold]Creating model...[/bold]")

                # Determine input dimensions
                image_input_dim = train_data['image_embeddings'].size(1)
                text_embed_dim = next(iter(text_embeddings_dict.values())).size(1)

                # Create model based on configuration
                model = MultiCPRFL(
                    attributes=attributes,
                    text_embed_dim=text_embed_dim,
                    prompt_dim=config.feature_d,
                    hidden_dim=config.dim_head,
                    num_classes_dict=num_classes_dict,
                    image_input_dim=image_input_dim,
                    use_vsi=setting["use_vsi"],
                    fusion_mode=setting["fusion_mode"],
                    use_contrastive=True,
                    dropout_rate=0.2,
                    vsi_num_heads=config.head_num
                )

                # Move model to device
                model = model.to(device)

                # Summary of model configuration for this repetition
                console.print(f"[bold]Model Configuration:[/bold]")
                console.print(f"  Fusion Mode: {setting['fusion_mode']}")
                console.print(f"  VSI: {setting['use_vsi']}")
                console.print(f"  Loss Type: {setting['loss_type']}")
                console.print(f"  Contrastive Loss: {setting['contrastive_loss_type']}")

                # Train the model
                console.print("[bold]Starting training...[/bold]")
                train_result = train_multi_cprfl(
                    model=model,
                    attributes=attributes,
                    train_data=train_data,
                    val_data=val_data,
                    train_labels_dict=train_labels_dict,
                    val_labels_dict=val_labels_dict,
                    label_to_idx_dict=label_to_idx_dict,
                    idx_to_label_dict=idx_to_label_dict,
                    index_data=index_data,
                    clip_model=clip_model,
                    device=device,
                    test_data=test_data,
                    test_labels_dict=test_labels_dict,
                    epochs=epochs,
                    batch_size=config.batch_size,
                    loss_type=setting["loss_type"],
                    contrastive_loss_type=setting["contrastive_loss_type"],
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    lr_scheduler_type="cosine",
                    warmup_epochs=config.warmup_epochs,
                    lambda_contrastive_start=setting["lambda_start"],
                    lambda_contrastive_end=setting["lambda_end"],
                    lambda_schedule=setting["lambda_schedule"],
                    contrastive_temperature=config.contrastive_temperature,
                    beta_prompt_separation=setting["beta_prompt_separation"],
                    contrastive_top_k_negatives=setting["contrastive_top_k_negatives"],
                    positive_weighting_beta=setting["positive_weighting_beta"],
                    early_stopping_patience=early_stopping_patience_value,
                    map_weight=map_weight_value,
                    save_best_combined=save_best_combined_value,
                    fusion_mode=setting["fusion_mode"],
                    use_vsi=setting["use_vsi"],
                    track_losses=True
                )

                # Check if we got a tuple back (model, loss_tracking_data)
                if isinstance(train_result, tuple) and len(train_result) == 2:
                    model, loss_tracking_data = train_result
                else:
                    model = train_result
                    loss_tracking_data = None

                # Evaluate the model
                console.print(f"[blue]-- Setting: {current_setting_name}, Rep {rep} evaluating... --[/blue]")
                rep_eval_results = evaluate_multi_cprfl(
                    model=model,
                    attributes=attributes,
                    test_data=test_data,
                    test_labels_dict=test_labels_dict,
                    label_to_idx_dict=label_to_idx_dict,
                    idx_to_label_dict=idx_to_label_dict,
                    index_data=index_data,
                    clip_model=clip_model,
                    device=device,
                    batch_size=config.batch_size
                )

                # Save the evaluation results
                if isinstance(rep_eval_results, dict):
                    result_entry = {"repetition": rep, "results": rep_eval_results}

                    # Add loss tracking data if available
                    if loss_tracking_data is not None:
                        result_entry["training_data"] = loss_tracking_data

                    setting_results_list.append(result_entry)
                else:
                    console.print(f"[yellow]Warning: Rep {rep} evaluation didn't return valid results.[/yellow]")

                # Save the model
                model_save_path = f"{results_dir}/model_{current_setting_name}_rep{rep}.pth"
                torch.save(model.state_dict(), model_save_path)
                console.print(f"[green]Model saved to {model_save_path}[/green]")

            # Calculate and display average results for this setting
            if setting_results_list:
                avg_metrics = calculate_average_metrics(setting_results_list, attributes)
                console.print(f"\n[bold magenta]===== Setting: {current_setting_name} average results ({repetitions} repetitions) =====\n[/bold magenta]")
                display_average_results(avg_metrics, attributes)

                # Save results to our final collection
                final_results_to_save[current_setting_name] = {
                    "setting_params": setting,
                    "avg_metrics": avg_metrics,
                    "detailed_results": setting_results_list
                }
            else:
                console.print(f"[yellow]Setting '{current_setting_name}' has no valid evaluation results.[/yellow]")

        # If we ran ablation studies, compare the results
        if do_ablation and len(experiment_settings) > 1:
            console.print(f"\n[bold magenta]===== Ablation Study Results: {param_name} =====\n[/bold magenta]")

            # Create a comparison table for each attribute
            for attr in attributes:
                table = Table(title=f"{attr} with different {param_name} values")
                table.add_column(param_name)
                table.add_column("Overall mAP")
                table.add_column("Overall F1")
                table.add_column("Head mAP")
                table.add_column("Medium mAP")
                table.add_column("Tail mAP")
                table.add_column("Head F1")
                table.add_column("Medium F1")
                table.add_column("Tail F1")

                for setting in experiment_settings:
                    setting_name = setting["setting_name"]
                    if setting_name in final_results_to_save:
                        results = final_results_to_save[setting_name]

                        # Determine the param value to display
                        if param_name == "VSI":
                            param_val = str(setting["use_vsi"])
                        elif param_name == "Fusion Mode":
                            param_val = setting["fusion_mode"]
                        elif param_name == "Loss Type":
                            param_val = setting["loss_type"]
                        elif param_name == "Contrastive Weight":
                            param_val = f"{setting['lambda_start']}-{setting['lambda_end']}"
                        elif param_name == "Contrastive Loss Type":
                            param_val = setting["contrastive_loss_type"]
                        elif param_name == "Prompt Separation Weight":
                            param_val = str(setting["beta_prompt_separation"])
                        elif param_name == "Hard Negative K":
                            param_val = str(setting["contrastive_top_k_negatives"])
                        elif param_name == "Positive Weighting Beta":
                            param_val = str(setting["positive_weighting_beta"])
                        else:
                            param_val = "unknown"

                        # Get metrics for this attribute
                        if attr in results["avg_metrics"]:
                            metrics = results["avg_metrics"][attr]

                            # Extract values with proper formatting
                            overall_map = f"{metrics['overall_mAP']['mean']:.2f}% ±{metrics['overall_mAP']['std']:.2f}" if metrics['overall_mAP']['mean'] is not None else "N/A"
                            overall_f1 = f"{metrics['overall_f1']['mean']:.2f}% ±{metrics['overall_f1']['std']:.2f}" if metrics['overall_f1']['mean'] is not None else "N/A"

                            # Head/Medium/Tail mAP
                            head_map = f"{metrics['head_mAP']['mean']:.2f}% ±{metrics['head_mAP']['std']:.2f}" if metrics['head_mAP']['mean'] is not None else "N/A"
                            medium_map = f"{metrics['medium_mAP']['mean']:.2f}% ±{metrics['medium_mAP']['std']:.2f}" if metrics['medium_mAP']['mean'] is not None else "N/A"
                            tail_map = f"{metrics['tail_mAP']['mean']:.2f}% ±{metrics['tail_mAP']['std']:.2f}" if metrics['tail_mAP']['mean'] is not None else "N/A"

                            # Head/Medium/Tail F1
                            head_f1 = f"{metrics['head_f1']['mean']:.2f}% ±{metrics['head_f1']['std']:.2f}" if metrics['head_f1']['mean'] is not None else "N/A"
                            medium_f1 = f"{metrics['medium_f1']['mean']:.2f}% ±{metrics['medium_f1']['std']:.2f}" if metrics['medium_f1']['mean'] is not None else "N/A"
                            tail_f1 = f"{metrics['tail_f1']['mean']:.2f}% ±{metrics['tail_f1']['std']:.2f}" if metrics['tail_f1']['mean'] is not None else "N/A"

                            table.add_row(param_val, overall_map, overall_f1, head_map, medium_map, tail_map, head_f1, medium_f1, tail_f1)

                console.print(table)

        # Save all results to a JSON file
        if do_ablation:
            if 'ablation_choice' in locals() and ablation_choice == "9":
                results_file = f"results/ablation_Contrastive_Loss_Comparison_e{epochs}_r{repetitions}_results.json"
            else:
                results_file = f"results/ablation_{param_name.replace(' ', '_')}_e{epochs}_r{repetitions}_results.json"
        else:
            results_file = f"results/{current_setting_name}_e{epochs}_r{repetitions}_results.json"

        with open(results_file, 'w') as f:
            json.dump(final_results_to_save, f, indent=2, cls=NumpyEncoder)
        console.print(f"[green]All results saved to {results_file}[/green]")

if __name__ == "__main__":
    main()