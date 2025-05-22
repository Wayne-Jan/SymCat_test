# inference.py
# Inference script for CPRFL model

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table

# Import configuration
import config

# Import components
from src.models.model import MultiCPRFL
from src.models.model_components import compute_multilabel_metrics
from src.data_loader.data_utils import load_index, convert_to_multihot
from src.helper_functions.train_eval import initialize_multi_category_embeddings

console = Console()

def load_model(model_path, device):
    """Load a trained model from the given path"""
    # Load index and categories
    label_to_idx_dict, idx_to_label_dict, index_data = load_index(config.symcat_json_path)
    
    # Get class counts for each attribute
    num_classes_dict = {attr: len(label_to_idx_dict[attr]) for attr in config.attributes}
    
    # Load CLIP model for text embeddings
    clip_model = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
    clip_model.eval()
    
    # Initialize text embeddings
    categories_dict = {attr: list(label_to_idx_dict[attr].keys()) for attr in config.attributes}
    text_embeddings_dict = initialize_multi_category_embeddings(
        config.attributes, categories_dict, clip_model, device, index_data
    )
    
    # Determine dimensions
    text_embed_dim = next(iter(text_embeddings_dict.values())).size(1)
    
    # Create model
    model = MultiCPRFL(
        attributes=config.attributes,
        text_embed_dim=text_embed_dim,
        prompt_dim=config.feature_d,
        hidden_dim=config.dim_head,
        num_classes_dict=num_classes_dict,
        image_input_dim=2048,  # Default CLIP dimension
        use_vsi=config.use_vsi,
        fusion_mode=config.fusion_mode,
        use_contrastive=True,
        dropout_rate=0.2,
        vsi_num_heads=config.head_num
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, text_embeddings_dict, label_to_idx_dict, idx_to_label_dict

def preprocess_image(image_path, device):
    """Preprocess an image and extract CLIP features"""
    # Load CLIP model
    clip_model = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
    clip_model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Extract features using CLIP
    with torch.no_grad():
        image_features = clip_model.encode([image])
        image_features = torch.tensor(image_features).to(device)
    
    return image_features

def inference(model, image_features, text_embeddings_dict, idx_to_label_dict, threshold=0.5):
    """Run inference on the given image features"""
    with torch.no_grad():
        # Forward pass
        logits_dict, _, _ = model(image_features, text_embeddings_dict)
        
        # Get predictions for each attribute
        predictions = {}
        for attr in config.attributes:
            if attr in logits_dict:
                logits = logits_dict[attr]
                probs = torch.sigmoid(logits)
                
                # Get top predictions
                values, indices = torch.topk(probs, k=min(5, probs.size(1)))
                indices = indices.squeeze().cpu().numpy()
                values = values.squeeze().cpu().numpy()
                
                # Convert indices to labels
                labels = []
                scores = []
                for idx, val in zip(indices, values):
                    if val >= threshold:
                        label = idx_to_label_dict[attr][idx]
                        labels.append(label)
                        scores.append(float(val))
                
                predictions[attr] = list(zip(labels, scores))
        
        return predictions

def main():
    parser = argparse.ArgumentParser(description="Run inference with CPRFL model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    args = parser.parse_args()
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Using device: {device}[/bold]")
    
    # Load model
    console.print("[bold]Loading model...[/bold]")
    model, text_embeddings_dict, label_to_idx_dict, idx_to_label_dict = load_model(args.model_path, device)
    
    # Preprocess image
    console.print(f"[bold]Processing image: {args.image_path}[/bold]")
    image_features = preprocess_image(args.image_path, device)
    
    # Run inference
    console.print("[bold]Running inference...[/bold]")
    predictions = inference(model, image_features, text_embeddings_dict, idx_to_label_dict, args.threshold)
    
    # Display results
    console.print("[bold green]Predictions:[/bold green]")
    for attr, preds in predictions.items():
        table = Table(title=f"{attr} Predictions")
        table.add_column("Label", style="cyan")
        table.add_column("Confidence", style="green")
        
        for label, score in preds:
            table.add_row(label, f"{score:.4f}")
        
        console.print(table)

if __name__ == "__main__":
    main()