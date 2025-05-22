# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SymCat is a plant disease categorization and classification project based on a multi-modal approach using both image and text data. The project combines vision-semantic information to create a robust classification system for plant diseases, categorized by crops, plant parts, symptom categories, and symptom tags.

Key features:
- Multiple classification approaches including baseline classifiers, multi-modal models, and zero-shot CLIP evaluation
- Support for different loss functions (BCE, ASL, DBFocal, etc.)
- Contrastive learning mechanisms with various strategies (SupCon, MSE, Image-Text, Image-Text V2)
- Bottleneck architectures for both visual and text processing paths
- Head/Medium/Tail analysis for handling class imbalance 
- Visual-Semantic Interaction (VSI) for enhanced feature representation
- Support for different fusion modes (Random, CLIP, Concat, Weighted, FiLM)

## Environment Setup

The codebase is built with PyTorch and uses the following key dependencies:
- PyTorch
- scikit-learn
- SentenceTransformer (with JINA CLIP-v2 model as the default text encoder)
- Rich (for console output formatting)

### Docker Setup

The project includes Docker support for consistent development:

```bash
# Build the container
docker-compose build

# Start the container in detached mode
docker-compose up -d

# Enter the running container
docker exec -it cprfl_symcat-symcat-1 bash

# Stop the container
docker-compose down
```

## Architecture Components

The project consists of several main components:

1. **BaselineClassifier**: A pure image-based classifier that doesn't use text embeddings
2. **MultiCPRFL**: Multi-modal model with Visual-Semantic Interaction (VSI)
   - Supports bottleneck architectures for dimensionality reduction
   - Can be configured with different fusion modes
   - Includes contrastive learning capabilities
3. **Loss functions**: 
   - Classification losses: BCE, ASL (Asymmetric Loss), DBFocal, MLS
   - Contrastive losses: SupCon, MSE, Image-Text, Image-Text V2 (with hard negative mining)
   - Prompt separation loss for more distinct class representations
4. **CLIP zero-shot**: Zero-shot evaluation using CLIP embeddings

## Data Structure

The project uses the following data components:
- `SymCat.json`: Main index file containing categories for crops, plant parts, symptom categories, and symptom tags
- `embeddings_train_sym.pt`, `embeddings_val_sym.pt`, `embeddings_test_sym.pt`: Pre-extracted CLIP image embeddings
- `NCHU_sorted_data.json`: Used for computing head/medium/tail statistics for class imbalance analysis

## Commands and Operations

### Running Training

The main entry point is `main.py` which provides an interactive interface for model training. The general workflow is:

```bash
python main.py
```

The script will prompt for configuration options:
1. Choose model type (Baseline, MultiCPRFL, or Zero-Shot)
2. Configure model parameters and training settings
3. Execute training and evaluation

### Training Modes

1. **Baseline Classifier**:
```bash
python main.py  # Then select option 1
```

2. **MultiCPRFL (Multi-modal model)**:
```bash
python main.py  # Then select option 2
```

3. **Zero-Shot CLIP Evaluation**:
```bash
python main.py  # Then select option 3
```

### Key Configuration Options

When running `main.py`, you'll be prompted to configure:

- Loss type (BCE, ASL, DBFocal, MLS)
- Fusion mode (random, clip, concat, weighted, film)
- Contrastive learning options (supcon, mse, image_text, image_text_v2) 
- VSI (Visual Semantic Interaction) usage
- Training parameters (epochs, learning rate, early stopping)
- Bottleneck dimensions for visual and text features
- Evaluation metrics weighting (mAP vs F1 balance)
- Early stopping patience
- Hard negative mining parameters for contrastive learning
- Positive weighting beta for frequency-based class weighting

### Ablation Studies

The system supports running ablation studies by setting:
```bash
python main.py  # Then select option 2, then answer 'y' to "ablation test"
```

This allows systematically testing different configurations such as:
- Visual Semantic Interaction (VSI)
- Fusion modes
- Loss types
- Contrastive learning weights and strategies
- Bottleneck dimensions

## Project Files and Structure

### Main Components
- **main.py**: Main entry point for training and evaluation
- **model_components.py**: Model architecture definitions including MultiCPRFL, VSI, and fusion modules
- **train_eval.py**: Training and evaluation functions for MultiCPRFL models
- **train_baseline.py**: Training and evaluation for BaselineClassifier
- **losses.py**: Implementation of different loss functions including ASL, DBFocal and contrastive losses
- **data_utils.py**: Data loading and processing utilities including multi-hot label conversion
- **BaselineClassifier.py**: Implementation of the baseline classifier
- **clip_zero_shot.py**: Zero-shot CLIP evaluation implementation

### Core Architectural Elements
- **VisualSemanticInteraction**: Attention-based module that refines visual features using text prompts
- **MultiPromptInitializer**: Converts text embeddings to prompts with bottleneck structure
- **FiLMFusion**: Feature-wise Linear Modulation for fusion of random and CLIP prompts
- **ContrastiveProjector**: Projects features for contrastive learning

## Model Training Strategy

1. Load pretrained Jina-CLIP-v2 model for text embeddings (default text encoder with 1024-dimensional embeddings)
2. Process image embeddings and multi-hot labels
3. Initialize model with selected parameters
4. Train with selected loss function and learning rate schedule
5. Calculate combined loss (classification + contrastive + prompt separation)
6. Evaluate using mAP, F1 and perform H/M/T analysis
7. Save the best model based on validation performance (either mAP or combined mAP+F1 score)

## Troubleshooting

Common issues:
- Missing data files: Ensure SymCat.json and embedding files are in the correct location
- GPU memory limitations: Reduce batch size or model dimensions
- Long training times: Consider using ablation studies with fewer epochs for exploration
- SentenceTransformer issues: Make sure Jina-CLIP-v2 model is properly installed with `trust_remote_code=True` option

## Development Tips

- When implementing new loss functions, add them to both the `get_loss_function` factory in losses.py and update the relevant options in main.py
- For adding new fusion modes, update both the MultiCPRFL class and the fusion mode options in main.py
- When working with VSI, note that visual features are used as Query while prompts are used as Key/Value in the attention mechanism
- Results are saved as JSON files with naming conventions including the key parameters