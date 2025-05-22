# train.py
# Main training script for CPRFL model

import os
# Set tokenizers parallelism to false to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import json
import torch.cuda.amp as amp
from rich.console import Console
from sentence_transformers import SentenceTransformer

# Import configuration
import config

# Import components
from src.models.model import MultiCPRFL
from src.models.baseline import BaselineClassifier
from src.data_loader.data_utils import load_index, convert_to_multihot, compute_head_medium_tail, attributes
from src.helper_functions.train_eval import train_multi_cprfl, initialize_multi_category_embeddings, data_key_map
from src.helper_functions.train_baseline import train_baseline_classifier
from src.loss_functions.loss import get_loss_function, get_contrastive_loss
from src.utils.util import create_progress_bar

console = Console()

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
    """Main training function"""
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Using device: {device}[/bold]")

    # Load CLIP model for text embeddings
    clip_model = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
    clip_model.eval()
    console.print(f"[bold green]Loaded Jina-CLIP-v2 model[/bold green]")

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

    for attr in config.attributes:
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
    
    # Initialize text embeddings
    categories_dict = {attr: list(label_to_idx_dict[attr].keys()) for attr in config.attributes}
    text_embeddings_dict = initialize_multi_category_embeddings(
        config.attributes, categories_dict, clip_model, device, index_data
    )
    
    # Get class counts for each attribute
    num_classes_dict = {attr: len(label_to_idx_dict[attr]) for attr in config.attributes}
    
    # Create model
    console.print("[bold]Creating model...[/bold]")
    
    # Determine input dimensions
    image_input_dim = train_data['image_embeddings'].size(1)
    text_embed_dim = next(iter(text_embeddings_dict.values())).size(1)
    
    # Create model based on configuration
    model = MultiCPRFL(
        attributes=config.attributes,
        text_embed_dim=text_embed_dim,
        prompt_dim=config.feature_d,
        hidden_dim=config.dim_head,
        num_classes_dict=num_classes_dict,
        image_input_dim=image_input_dim,
        use_vsi=config.use_vsi,
        fusion_mode=config.fusion_mode,
        use_contrastive=True,
        dropout_rate=0.2,
        vsi_num_heads=config.head_num
    )
    
    # Move model to device
    model = model.to(device)
    
    # Summary of model configuration
    console.print(f"[bold]Model Configuration:[/bold]")
    console.print(f"  Fusion Mode: {config.fusion_mode}")
    console.print(f"  VSI: {config.use_vsi}")
    console.print(f"  Loss Type: {config.loss_type}")
    console.print(f"  Contrastive Loss: {config.contrastive_loss_type}")
    
    # Train the model
    console.print("[bold]Starting training...[/bold]")
    train_multi_cprfl(
        model=model,
        attributes=config.attributes,
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
        epochs=config.epochs,
        batch_size=config.batch_size,
        loss_type=config.loss_type,
        contrastive_loss_type=config.contrastive_loss_type,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type="cosine",
        warmup_epochs=config.warmup_epochs,
        lambda_contrastive_start=config.lambda_contrastive_start,
        lambda_contrastive_end=config.lambda_contrastive_end,
        lambda_schedule=config.lambda_schedule,
        contrastive_temperature=config.contrastive_temperature,
        beta_prompt_separation=config.beta_prompt_separation,
        positive_weighting_beta=config.positive_weighting_beta,
        early_stopping_patience=config.early_stopping_patience,
        fusion_mode=config.fusion_mode,
        use_vsi=config.use_vsi
    )

if __name__ == "__main__":
    main()