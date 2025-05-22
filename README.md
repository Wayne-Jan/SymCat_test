# CPRFL - Category-Prompt Refined Feature Learning

Implementation of the Category-Prompt Refined Feature Learning (CPRFL) for Long-Tailed Multi-Label Image Classification, applied to plant disease classification.

## Project Overview

CPRFL is a novel approach for multi-label image classification that leverages CLIP's text encoder to extract category semantics and establish semantic correlations between head and tail classes. The method initializes category-prompts from the pretrained CLIP's embeddings and decouples category-specific visual representations through interaction with visual features.

Key components:
- **Prompt Initialization (PI)**: Initializes category-prompts from CLIP's text embeddings
- **Visual-Semantic Interaction (VSI)**: Decouples category-specific visual representations through interaction with visual features
- **FiLM Fusion**: Feature-wise Linear Modulation for fusion of random and CLIP prompts
- **Asymmetric Loss**: Suppresses negative samples across all classes to enhance head-to-tail recognition performance

## Project Structure

```
├── config.py              # Configuration settings
├── train.py               # Training script
├── inference.py           # Inference script
├── src/
│   ├── data_loader/       # Data loading utilities
│   ├── helper_functions/  # Helper functions for training/evaluation
│   ├── loss_functions/    # Loss function implementations
│   └── models/            # Model architectures
├── utils/                 # Utility scripts
└── data/                  # Data files
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cprfl.git
cd cprfl

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python train.py
```

### Inference

To run inference on an image:

```bash
python inference.py --model_path path/to/model.pth --image_path path/to/image.jpg
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This code is based on the paper:
"Category-Prompt Refined Feature Learning for Long-Tailed Multi-Label Image Classification"