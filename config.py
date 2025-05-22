# config.py
# Configuration file for the CPRFL model

# Model Architecture Settings
head_num = 8  # Number of attention heads in VSI
depth = 1  # Depth of the model
feature_d = 2048  # Feature dimension
dim_head = 512  # Dimension per attention head
mlp_dim = 2048  # MLP layer dimension

# Model Flags
label_emb_drop = False  # Label embedding dropout
nonlinear = True  # Enable non-linear transformations
no_cls_grad = False  # Gradient settings for classification layer
expand = 0.5  # Expansion factor

# Training Settings
batch_size = 192
learning_rate = 5e-5
weight_decay = 1e-4
epochs = 150
warmup_epochs = 5
early_stopping_patience = 10

# Loss Settings
loss_type = "asl"  # Options: "bce", "asl", "dbfocal", "mls"
contrastive_loss_type = "image_text_v2"  # Options: "mse", "supcon", "image_text_v2", "prototype"
lambda_contrastive_start = 0.2  # Initial contrastive loss weight
lambda_contrastive_end = 0.8  # Final contrastive loss weight
lambda_schedule = "linear"  # Options: "linear", "constant", "cosine"
contrastive_temperature = 0.1
beta_prompt_separation = 0.0  # Prompt separation loss weight
positive_weighting_beta = 0.999  # Beta for class-balanced weighting

# VSI and Fusion Settings
use_vsi = True  # Whether to use Visual-Semantic Interaction
fusion_mode = "film"  # Options: "random", "clip", "concat", "film"

# Dataset Paths
symcat_json_path = "SymCat.json"
nchu_sorted_data_path = "NCHU_sorted_data_fixed.json"
train_embeddings_path = "embeddings_train_sym.pt"
val_embeddings_path = "embeddings_val_sym.pt"
test_embeddings_path = "embeddings_test_sym.pt"

# Generate dynamic model name based on configuration
model_name = f"model_VSI={use_vsi}_Fusion={fusion_mode}_Loss={loss_type}_CLT={contrastive_loss_type}_LCL={lambda_contrastive_start}-{lambda_contrastive_end}_PB={positive_weighting_beta}"

# Attributes to process
attributes = ['crop', 'part', 'symptomCategories', 'symptomTags']