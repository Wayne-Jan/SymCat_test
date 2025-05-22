# src/loss_functions/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 分類損失函數 (保持不變 + 新增 DBFocalLoss) ---
class AsymmetricLossOptimized(nn.Module):
    """
    ASL: Asymmetric Loss for Multi-Label Classification.
    論文: https://arxiv.org/abs/2009.14119
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """
        Args:
            x: input logits (N, C)
            y: targets (multi-hot) (N, C)
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 基本 BCE Loss
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    pt0 = xs_pos * y
                    pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
                    pt = pt0 + pt1
                    one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                    one_sided_w = torch.pow(1 - pt, one_sided_gamma)
                    loss = loss * one_sided_w
            else:
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
                loss *= one_sided_w

        return -loss.sum() / x.size(0)


class DBFocalLoss(nn.Module):
    """
    Distribution-Balanced Focal Loss.
    論文: https://arxiv.org/abs/2004.10752
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean', weight=None):
        super(DBFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, targets):
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)  # 對正負樣本都計算
        focal_weight = (1 - pt).pow(self.gamma)

        # Alpha 平衡
        alpha_weight = torch.ones_like(targets)
        if isinstance(self.alpha, float) and self.alpha >= 0:
            alpha_weight = alpha_weight * (1 - self.alpha)
            alpha_weight = alpha_weight + targets * (2 * self.alpha - 1)

        loss = alpha_weight * focal_weight * bce_loss

        # 類別權重 (若有提供)
        if self.weight is not None:
            class_weight = self.weight.unsqueeze(0).to(logits.device)  # [1, num_classes]
            loss = loss * class_weight

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# --- 對比學習輔助函數 (保持不變) ---
def compute_label_overlap(labels):
    """
    計算批次中所有樣本對之間的標籤重疊率
    labels: [batch_size, num_classes] 多熱編碼的標籤
    返回: [batch_size, batch_size] 的重疊率矩陣 (值域 0-1)
    """
    batch_size = labels.size(0)
    if batch_size == 0:
        return torch.zeros((0, 0), device=labels.device)

    labels = labels.float()
    intersection = torch.matmul(labels, labels.t())
    label_counts = torch.sum(labels, dim=1, keepdim=True)
    min_counts = torch.minimum(label_counts, label_counts.t())

    overlap_ratio = torch.zeros_like(intersection)
    valid_mask = min_counts > 0
    overlap_ratio[valid_mask] = intersection[valid_mask] / min_counts[valid_mask]

    return overlap_ratio


# --- 原本的對比學習損失 (MSE / SupCon / ImageText) ---
class ContrastiveLossMSE(nn.Module):
    """
    對比學習損失 - MSE 風格
    目標：讓特徵相似度匹配標籤重疊度。
    """
    def __init__(self, temperature=0.1):
        super(ContrastiveLossMSE, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels_dict, attributes):
        device = features.device
        batch_size = features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        total_loss = 0.0
        num_valid_attrs = 0
        norm_features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_features, norm_features.t())
        mask_off_diagonal = torch.ones_like(similarity_matrix) - torch.eye(batch_size, device=device)

        for attr in attributes:
            if attr not in labels_dict:
                continue
            labels = labels_dict[attr]
            if labels is None or labels.numel() == 0:
                continue

            overlap_matrix = compute_label_overlap(labels)
            scaled_similarity = similarity_matrix / self.temperature
            scaled_overlap = overlap_matrix / self.temperature
            loss = torch.sum(((scaled_similarity - scaled_overlap) ** 2) * mask_off_diagonal)

            num_pairs = batch_size * (batch_size - 1)
            if num_pairs > 0:
                loss = loss / num_pairs
            else:
                loss = torch.tensor(0.0, device=device)

            total_loss += loss
            num_valid_attrs += 1

        if num_valid_attrs == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / num_valid_attrs


class SupConStyleLoss(nn.Module):
    """
    對比學習損失 - Supervised Contrastive (SupCon) 風格
    目標：拉近標籤重疊度高的樣本對，推開標籤重疊度低的樣本對。
    """
    def __init__(self, temperature=0.1, base_temperature=None):
        super(SupConStyleLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature if base_temperature is not None else self.temperature

    def forward(self, features, labels_dict, attributes):
        device = features.device
        batch_size = features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        total_loss = 0.0
        num_valid_attrs = 0
        norm_features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_features, norm_features.t())
        mask_off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        for attr in attributes:
            if attr not in labels_dict:
                continue
            labels = labels_dict[attr]
            if labels is None or labels.numel() == 0:
                continue

            overlap_matrix = compute_label_overlap(labels)
            logits = similarity_matrix / self.temperature
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            logits = logits - logits_max.detach()

            exp_logits = torch.exp(logits)
            log_prob_denominator = torch.log((exp_logits * mask_off_diagonal.float()).sum(1, keepdim=True) + 1e-8)
            log_prob = logits - log_prob_denominator

            overlap_matrix_masked = overlap_matrix * mask_off_diagonal.float()
            sum_overlap = torch.sum(overlap_matrix_masked, dim=1)
            has_positive_pairs = sum_overlap > 1e-8
            numerator = torch.sum(overlap_matrix_masked * log_prob, dim=1)

            loss_per_sample = torch.zeros_like(sum_overlap)
            loss_per_sample[has_positive_pairs] = -numerator[has_positive_pairs] / sum_overlap[has_positive_pairs]

            if has_positive_pairs.sum() > 0:
                attr_loss = loss_per_sample[has_positive_pairs].mean()
                total_loss += attr_loss
                num_valid_attrs += 1

        if num_valid_attrs == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / num_valid_attrs


class ImageTextContrastiveLoss(nn.Module):
    """
    Image-Text 對比學習損失 (CLIP 風格) - 已由 V2 版本替代，此為保留版本
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, prompts_dict, labels_dict, attributes):
        total_loss = 0.0
        num_valid_attrs = 0
        device = image_features.device
        batch_size = image_features.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # 圖像特徵正規化
        z_img = F.normalize(image_features, p=2, dim=1)

        for attr in attributes:
            if attr not in prompts_dict or attr not in labels_dict:
                continue
            P_attr = prompts_dict[attr]
            labels_attr = labels_dict[attr].float()

            if P_attr is None or P_attr.numel() == 0 or labels_attr is None or labels_attr.numel() == 0:
                continue

            num_classes_attr = P_attr.size(0)
            if labels_attr.size(1) != num_classes_attr:
                continue

            # 提示詞正規化
            z_prompts_attr = F.normalize(P_attr, p=2, dim=1)

            # 相似度 (B x num_classes)
            sim = z_img @ z_prompts_attr.t()
            sim = sim / self.temperature

            # softmax over prompts
            log_prob = F.log_softmax(sim, dim=1)

            # 正樣本總數 (每個 image)
            num_positives = labels_attr.sum(dim=1)
            valid_normalization_mask = num_positives > 0

            # 計算各正樣本對應的 log_prob 之和
            sum_log_prob_pos = (log_prob * labels_attr).sum(dim=1)

            loss_per_image = torch.zeros_like(num_positives, device=device)
            loss_per_image[valid_normalization_mask] = -sum_log_prob_pos[valid_normalization_mask] / num_positives[valid_normalization_mask]

            # 只對有正樣本的圖像取平均
            if valid_normalization_mask.sum() > 0:
                attr_loss = loss_per_image[valid_normalization_mask].mean()
                total_loss += attr_loss
                num_valid_attrs += 1

        if num_valid_attrs == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / num_valid_attrs


# --- 現在作為預設的 Image-Text 對比損失 ---
class ImageTextContrastiveLossV2(nn.Module):
    """
    Image-Text 對比損失 V2 (向量化版本):
    - 可選的 Hard Negative Mining。
    - 可選的基於頻率的正樣本加權。
    """
    def __init__(self, temperature=0.1, top_k_negatives=None, class_weights_dict=None):
        super().__init__()
        self.temperature = temperature
        self.top_k_negatives = top_k_negatives
        self.class_weights_dict = class_weights_dict if class_weights_dict is not None else {}
        self.eps = 1e-8 # Epsilon for numerical stability

        if self.top_k_negatives is not None and self.top_k_negatives <= 0:
            print("警告: top_k_negatives <= 0，將使用所有負樣本。")
            self.top_k_negatives = None

    def forward(self, image_features, prompts_dict, labels_dict, attributes):
        total_loss = 0.0
        num_valid_attrs = 0
        device = image_features.device
        batch_size = image_features.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        z_img = F.normalize(image_features, p=2, dim=1) # [B, D]

        for attr in attributes:
            if attr not in prompts_dict or attr not in labels_dict: continue
            P_attr = prompts_dict[attr]         # [C, D]
            labels_attr = labels_dict[attr].float() # [B, C]
            if P_attr is None or P_attr.numel() == 0 or labels_attr is None or labels_attr.numel() == 0: continue
            num_classes_attr = P_attr.size(0)
            if labels_attr.size(1) != num_classes_attr: continue

            z_prompts_attr = F.normalize(P_attr, p=2, dim=1) # [C, D]

            # --- 1. 計算圖像-提示詞相似度 ---
            sim = (z_img @ z_prompts_attr.t()) / self.temperature # [B, C]

            # --- 2. 準備 Mask ---
            pos_mask = (labels_attr > 0) # [B, C], True for positive pairs
            neg_mask = ~pos_mask         # [B, C], True for negative pairs

            # --- 3. Hard Negative Mining (如果啟用) ---
            logits_for_softmax = sim.clone() # Start with all logits
            if self.top_k_negatives is not None:
                # Fill positive similarities with -inf so they won't be selected by topk
                sim_only_neg = sim.masked_fill(pos_mask, -float('inf'))
                # Find top-k negative similarities for each image
                # Note: k must be <= number of actual negatives
                actual_k = min(self.top_k_negatives, neg_mask.sum(dim=1).min().item()) # Use the minimum available negatives if k is too large
                if actual_k > 0 : # Proceed only if k is valid
                  hard_neg_sim, _ = torch.topk(sim_only_neg, k=actual_k, dim=1, largest=True) # [B, K]
                  # Create a mask for the denominator: positives + hard negatives
                  # We need to be careful here. The easiest way is often to calculate
                  # the full logsumexp and then subtract the easy negatives.
                  # Alternative: Construct the logits for softmax carefully.
                  # Let's stick to constructing the logits for softmax directly for now.

                  # Create a combined logit tensor [B, C] where easy negatives are masked out
                  is_hard_neg_mask = torch.zeros_like(sim, dtype=torch.bool)
                  # Get indices of top k negatives. This part is tricky to vectorize perfectly without loops or advanced indexing.
                  # Let's use topk indices for a simplified (potentially slightly less accurate) approach: gather hard neg logits.
                  # A more robust (but complex) way would involve scatter/masking based on indices.

                  # Simplified approach: just use pos + topk neg for logsumexp
                  # This requires careful handling if num_pos + k != C
                  # Let's refine: Calculate logsumexp over all, then adjust?
                  # Or, reconstruct the tensor for logsumexp:
                  logits_for_softmax = torch.full_like(sim, -float('inf')) # Mask all initially
                  logits_for_softmax = logits_for_softmax.masked_scatter_(pos_mask, sim[pos_mask]) # Fill positives
                  # How to efficiently scatter the hard negatives?
                  # This vectorization is non-trivial. Let's fall back to a slightly less efficient but vectorized masking approach.

                  # Get the similarity threshold for top-k negatives for each row
                  if hard_neg_sim.numel() > 0: # Check if topk returned anything
                      min_hard_neg_sim = hard_neg_sim[:, -1].unsqueeze(1) # [B, 1] Threshold for each image
                      # Mask negatives that are easier than the k-th hardest negative
                      is_easy_neg = neg_mask & (sim < min_hard_neg_sim)
                      logits_for_softmax = logits_for_softmax.masked_fill(is_easy_neg, -float('inf'))
                  # else: if k=0 or no negatives, logits_for_softmax keeps only positives.

            # --- 4. 計算 LogSumExp ---
            # LogSumExp over positives and selected negatives
            logsumexp_val = torch.logsumexp(logits_for_softmax.masked_fill(~pos_mask & (logits_for_softmax == -float('inf')), 0.0), dim=1) # [B], mask -inf before logsumexp

            # --- 5. 計算正樣本 Log Probability ---
            # log_prob = sim - logsumexp_val.unsqueeze(1) # [B, C]
            # Select log_probs for positive pairs
            log_prob_pos = sim[pos_mask] - logsumexp_val.repeat_interleave(pos_mask.sum(dim=1)) # [Total Positives in Batch]

            # --- 6. 正樣本加權 ---
            num_pos_per_image = pos_mask.sum(dim=1) # [B]
            if attr in self.class_weights_dict and self.class_weights_dict[attr] is not None:
                weights = self.class_weights_dict[attr].to(device)
                if weights.size(0) == num_classes_attr:
                    # Get weights corresponding to the positive samples
                    # Need to index weights based on positive indices for each image
                    pos_indices = torch.where(pos_mask)[1] # Get column indices of positives
                    pos_weights_flat = weights[pos_indices] # [Total Positives in Batch]

                    # Apply weights
                    weighted_log_probs = log_prob_pos * pos_weights_flat

                    # Calculate weighted loss per image
                    # Need to group by image index
                    image_indices = torch.where(pos_mask)[0] # Row indices
                    loss_per_image_weighted = torch.zeros(batch_size, device=device).scatter_add_(
                        0, image_indices, -weighted_log_probs # Accumulate negative weighted log probs
                    )
                    # Calculate sum of weights per image for normalization
                    sum_pos_weights_per_image = torch.zeros(batch_size, device=device).scatter_add_(
                        0, image_indices, pos_weights_flat
                    )
                    # Normalize loss
                    loss_per_image = loss_per_image_weighted / (sum_pos_weights_per_image + self.eps)

                else: # Fallback if weights mismatch
                    print(f"Warn: Dim mismatch for weights attr {attr}")
                    loss_per_image = torch.zeros(batch_size, device=device).scatter_add_(
                        0, torch.where(pos_mask)[0], -log_prob_pos
                    )
                    loss_per_image = loss_per_image / (num_pos_per_image + self.eps) # Average
            else: # No weighting
                # Calculate average loss per image (average over positives)
                 loss_per_image = torch.zeros(batch_size, device=device).scatter_add_(
                     0, torch.where(pos_mask)[0], -log_prob_pos
                 )
                 loss_per_image = loss_per_image / (num_pos_per_image + self.eps) # Average

            # --- 7. 計算最終批次損失 ---
            # Only consider images that had positive labels
            valid_images_mask = num_pos_per_image > 0
            if valid_images_mask.sum() > 0:
                attr_loss = loss_per_image[valid_images_mask].mean()
                total_loss += attr_loss
                num_valid_attrs += 1

        if num_valid_attrs == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / num_valid_attrs


# --- 提示詞分離損失 (保持不變) ---
class PromptSeparationLoss(nn.Module):
    """
    確保同一屬性的提示詞彼此盡可能分離（相似度越低越好）。
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, prompts_dict, attributes):
        all_attr_losses = []
        device = None

        for attr in attributes:
            if attr not in prompts_dict:
                continue
            P_attr = prompts_dict[attr]
            if P_attr is None or P_attr.size(0) <= 1:
                continue

            if device is None:
                device = P_attr.device

            norm_P_attr = F.normalize(P_attr, p=2, dim=1)
            sim_matrix = torch.matmul(norm_P_attr, norm_P_attr.t())
            num_classes = sim_matrix.size(0)

            if num_classes <= 1:
                continue

            # 只取上三角（排除對角線）
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            upper_triangle_sim = sim_matrix[mask]

            if upper_triangle_sim.numel() > 0:
                mean_sim = upper_triangle_sim.mean()
                all_attr_losses.append(mean_sim)

        if not all_attr_losses:
            return torch.tensor(0.0, device=device if device else 'cpu')

        total_loss = torch.stack(all_attr_losses).mean()
        return total_loss


# --- 工廠函數：分類損失 ---
def get_loss_function(loss_type, multi_hot_labels_dict=None, **kwargs):
    """
    獲取分類損失函數實例。
    """
    loss_type = loss_type.lower()
    print(f"初始化分類損失函數: {loss_type}")

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)

    elif loss_type == "asl":
        gamma_neg = kwargs.get('gamma_neg', 4)
        gamma_pos = kwargs.get('gamma_pos', 0)
        clip = kwargs.get('clip', 0.05)
        eps = kwargs.get('eps', 1e-8)
        print(f"  ASL params: gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, clip={clip}")
        return AsymmetricLossOptimized(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, eps=eps)

    elif loss_type == "dbfocal":
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', 0.25)
        reduction = kwargs.get('reduction', 'mean')
        print(f"  DBFocal params: gamma={gamma}, alpha={alpha}, reduction={reduction}")
        if multi_hot_labels_dict is None:
            print("  警告: DBFocalLoss 需要 multi_hot_labels_dict 計算權重但未提供。")
            return DBFocalLoss(gamma=gamma, alpha=alpha, reduction=reduction, weight=None)
        else:
            focal_loss_dict = {}
            for attr, label_mat in multi_hot_labels_dict.items():
                if (label_mat is None or label_mat.numel() == 0 or
                    label_mat.size(0) == 0 or label_mat.size(1) == 0):
                    print(f"  警告: 屬性 '{attr}' 標籤無效，跳過 DBFocalLoss 權重計算。")
                    focal_loss_dict[attr] = DBFocalLoss(gamma=gamma, alpha=alpha, reduction=reduction, weight=None)
                    continue

                num_classes = label_mat.size(1)
                counts = label_mat.sum(dim=0).float()
                beta = kwargs.get('db_beta', 0.999)

                effective_num = 1.0 - torch.pow(beta, counts)
                weight = (1.0 - beta) / (effective_num + 1e-8)
                weight = weight / torch.sum(weight) * num_classes
                weight = torch.clamp(weight, min=0.1, max=10.0)

                focal_loss_dict[attr] = DBFocalLoss(gamma=gamma, alpha=alpha, reduction=reduction, weight=weight)

            return focal_loss_dict

    elif loss_type == "mls":
        return nn.MultiLabelSoftMarginLoss(**kwargs)

    else:
        print(f"未知的分類損失類型 '{loss_type}'，返回預設 BCEWithLogitsLoss。")
        return nn.BCEWithLogitsLoss()


# --- 進階：Prototype-based 對比損失加記憶庫 ---
class PrototypeContrastiveLoss(nn.Module):
    """
    Prototype-based 對比學習損失，加入記憶庫:
    - 每個類別維護一個原型向量 (prototype)
    - 記憶庫存儲歷史特徵，擴大有效批次大小
    - 結合多標籤的正樣本權重調整
    - 使用原型來收集和整合類別級別的知識
    """
    def __init__(self,
                 memory_size=4096,
                 temperature=0.1,
                 momentum=0.9,
                 use_hard_negatives=True,
                 positive_weighting_beta=0.999):
        super().__init__()
        self.memory_size = memory_size
        self.temperature = temperature
        self.momentum = momentum
        self.use_hard_negatives = use_hard_negatives
        self.positive_weighting_beta = positive_weighting_beta

        # 初始化空的原型和記憶庫
        self.prototypes = {}  # {attr: {class_idx: prototype_vector}}
        self.class_weights = {}  # {attr: {class_idx: weight}}
        self.memories = {}  # {attr: tensor of features}
        self.memory_labels = {}  # {attr: tensor of labels}
        self.memory_ptr = {}  # {attr: current position in memory bank}
        self.eps = 1e-8

    def _update_prototype(self, attr, class_idx, features, mask):
        """更新單個類別的原型向量"""
        # 獲取帶有此類別的樣本
        valid_features = features[mask]
        if valid_features.size(0) == 0:
            return

        # 計算平均特徵
        mean_feature = torch.mean(valid_features, dim=0)
        mean_feature = F.normalize(mean_feature, p=2, dim=0)

        if attr not in self.prototypes:
            self.prototypes[attr] = {}

        # 如果是第一次更新，直接設定；否則使用動量更新
        if class_idx not in self.prototypes[attr]:
            # 使用 clone().detach() 以避免原地修改
            self.prototypes[attr][class_idx] = mean_feature.clone().detach()
        else:
            # 計算新的原型向量
            proto = self.prototypes[attr][class_idx]
            # 使用 clone().detach() 以避免原地修改
            new_proto = (self.momentum * proto + (1 - self.momentum) * mean_feature).clone().detach()
            self.prototypes[attr][class_idx] = F.normalize(new_proto, p=2, dim=0)

    def _update_memory(self, attr, features, labels):
        """更新記憶庫"""
        batch_size = features.size(0)
        feature_dim = features.size(1)
        num_classes = labels.size(1)

        # 初始化記憶庫（如果尚未初始化）
        if attr not in self.memories:
            self.memories[attr] = torch.zeros(self.memory_size, feature_dim, device=features.device)
            self.memory_labels[attr] = torch.zeros(self.memory_size, num_classes, dtype=labels.dtype, device=labels.device)
            self.memory_ptr[attr] = 0

        # 計算要添加的樣本數量
        ptr = self.memory_ptr[attr]
        # 創建新的記憶庫張量而不是直接修改現有的
        new_memories = self.memories[attr].clone().detach()
        new_memory_labels = self.memory_labels[attr].clone().detach()

        if ptr + batch_size > self.memory_size:
            # 如果超出記憶庫大小，先添加能添加的部分
            num_to_add = self.memory_size - ptr
            new_memories[ptr:ptr+num_to_add] = features[:num_to_add].clone().detach()
            new_memory_labels[ptr:ptr+num_to_add] = labels[:num_to_add].clone().detach()
            # 然後從頭開始覆蓋舊樣本
            remaining = batch_size - num_to_add
            if remaining > 0:
                new_memories[:remaining] = features[num_to_add:].clone().detach()
                new_memory_labels[:remaining] = labels[num_to_add:].clone().detach()
            new_ptr = remaining
        else:
            # 否則直接添加
            new_memories[ptr:ptr+batch_size] = features.clone().detach()
            new_memory_labels[ptr:ptr+batch_size] = labels.clone().detach()
            new_ptr = (ptr + batch_size) % self.memory_size

        # 使用新張量替換舊張量，避免就地修改
        self.memories[attr] = new_memories
        self.memory_labels[attr] = new_memory_labels
        self.memory_ptr[attr] = new_ptr

    def _calculate_class_weights(self, labels):
        """根據標籤頻率計算類別權重"""
        if self.positive_weighting_beta is None or self.positive_weighting_beta <= 0:
            return None

        counts = labels.sum(dim=0).float()  # [num_classes]
        effective_num = 1.0 - torch.pow(torch.tensor(self.positive_weighting_beta), counts)
        weights = (1.0 - self.positive_weighting_beta) / (effective_num + self.eps)

        # 正規化權重
        weights = weights / torch.sum(weights) * labels.size(1)
        return weights

    def forward(self, image_features, prompts_dict, labels_dict, attributes):
        total_loss = 0.0
        num_valid_attrs = 0
        device = image_features.device
        batch_size = image_features.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # 圖像特徵正規化
        z_img = F.normalize(image_features, p=2, dim=1)

        for attr in attributes:
            if attr not in prompts_dict or attr not in labels_dict:
                continue

            labels_attr = labels_dict[attr].float()  # [B, num_classes]
            P_attr = prompts_dict[attr]  # [num_classes, D]

            if P_attr is None or P_attr.numel() == 0 or labels_attr is None or labels_attr.numel() == 0:
                continue

            num_classes_attr = P_attr.size(0)
            if labels_attr.size(1) != num_classes_attr:
                continue

            # 計算或更新此屬性的類別權重
            if attr not in self.class_weights or self.class_weights[attr] is None:
                self.class_weights[attr] = self._calculate_class_weights(labels_attr)

            # 更新原型 - 使用 with torch.no_grad() 以避免干擾梯度計算
            with torch.no_grad():
                for class_idx in range(num_classes_attr):
                    class_mask = labels_attr[:, class_idx] > 0
                    if class_mask.sum() > 0:
                        self._update_prototype(attr, class_idx, z_img.detach(), class_mask)

            # 更新記憶庫
            # 使用 detach() 斷開梯度連接，避免原地修改
            with torch.no_grad():
                self._update_memory(attr, z_img.detach(), labels_attr.detach())

            # 獲取增強的特徵集（當前批次 + 記憶庫）
            enhanced_features = z_img
            enhanced_labels = labels_attr

            if attr in self.memories and self.memory_ptr[attr] > 0:
                # 使用實際填充的記憶庫大小
                memory_size = min(self.memory_size, self.memory_ptr[attr])
                if memory_size > 0:
                    memory_features = self.memories[attr][:memory_size]
                    memory_labels = self.memory_labels[attr][:memory_size]

                    # 合併當前批次和記憶庫
                    enhanced_features = torch.cat([z_img, memory_features], dim=0)
                    enhanced_labels = torch.cat([labels_attr, memory_labels], dim=0)

            # 計算增強特徵與提示詞的相似度
            z_prompts_attr = F.normalize(P_attr, p=2, dim=1)
            sim = enhanced_features @ z_prompts_attr.t()  # [B+M, num_classes]
            sim = sim / self.temperature

            # 計算正樣本的相似度和損失
            pos_mask = (enhanced_labels > 0)  # [B+M, num_classes]

            # 準備分母的 mask
            if self.use_hard_negatives:
                # 只使用困難負樣本（相似度較高的）
                K = min(num_classes_attr // 4, 10)  # 預設使用 25% 的類別作為困難負樣本
                sim_for_neg = sim.clone()
                sim_for_neg[pos_mask] = -float('inf')  # 遮蔽正樣本

                # 確保 K 不超過可用負樣本數量
                valid_neg_per_sample = (~pos_mask).sum(dim=1).min().item()  # 每個樣本最少的負樣本數量

                # 如果沒有可用的負樣本，使用所有負樣本
                if valid_neg_per_sample == 0:
                    neg_mask = ~pos_mask
                    full_mask = torch.ones_like(sim, dtype=torch.bool)
                else:
                    # 計算實際可用的 K 值
                    actual_K = min(K, valid_neg_per_sample)

                    # 只有在有足夠負樣本時才進行 topk 選擇
                    if actual_K > 0:
                        # 選擇困難負樣本
                        hard_neg_vals, _ = torch.topk(sim_for_neg, k=actual_K, dim=1)  # [B+M, K]
                        min_hard_neg = hard_neg_vals[:, -1].unsqueeze(1)  # [B+M, 1]

                        # 創建用於 LogSumExp 的 mask
                        neg_mask = (~pos_mask) & (sim >= min_hard_neg)
                    else:
                        neg_mask = ~pos_mask

                    full_mask = pos_mask | neg_mask
            else:
                # 使用所有負樣本
                full_mask = torch.ones_like(sim, dtype=torch.bool)

            # 計算 LogSumExp
            sim_masked = sim.clone()
            sim_masked[~full_mask] = -float('inf')
            logsumexp_val = torch.logsumexp(sim_masked, dim=1)  # [B+M]

            # 計算正樣本的 log prob
            log_prob_pos = sim[pos_mask] - logsumexp_val.repeat_interleave(pos_mask.sum(dim=1))

            # 應用類別權重
            loss_per_sample = torch.zeros(enhanced_features.size(0), device=device)
            if self.class_weights[attr] is not None:
                # 權重化的損失
                weights = self.class_weights[attr].to(device)
                pos_indices = torch.where(pos_mask)[0]  # 樣本索引
                class_indices = torch.where(pos_mask)[1]  # 類別索引
                pos_weights = weights[class_indices]  # 權重

                # 累積權重化的 log_prob
                weighted_log_probs = -log_prob_pos * pos_weights
                loss_per_sample.scatter_add_(0, pos_indices, weighted_log_probs)

                # 計算每個樣本的權重總和（用於正規化）
                weight_sums = torch.zeros_like(loss_per_sample)
                weight_sums.scatter_add_(0, pos_indices, pos_weights)

                # 正規化每個樣本的損失
                valid_samples = weight_sums > 0
                loss_per_sample[valid_samples] /= weight_sums[valid_samples] + self.eps
            else:
                # 非權重化的損失
                pos_indices = torch.where(pos_mask)[0]
                loss_per_sample.scatter_add_(0, pos_indices, -log_prob_pos)

                # 正規化（每個樣本的正樣本數量）
                num_pos_per_sample = pos_mask.sum(dim=1)
                valid_samples = num_pos_per_sample > 0
                loss_per_sample[valid_samples] /= num_pos_per_sample[valid_samples] + self.eps

            # 只考慮有正樣本的樣本
            if valid_samples.sum() > 0:
                attr_loss = loss_per_sample[valid_samples].mean()
                total_loss += attr_loss
                num_valid_attrs += 1

        if num_valid_attrs == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / num_valid_attrs


# --- 工廠函數：對比損失 ---
def get_contrastive_loss(loss_type='supcon',
                         temperature=0.1,
                         base_temperature=None,
                         top_k_negatives=None,
                         class_weights_dict=None,
                         memory_size=4096,
                         momentum=0.9,
                         positive_weighting_beta=None):
    """
    獲取對比學習損失函數實例。

    Args:
        loss_type (str): 'mse', 'supcon', 'image_text', 'image_text_v2', 'prototype'
        temperature (float): 溫度參數
        base_temperature (float, optional): SupCon 中的 base_temperature
        top_k_negatives (int, optional): 給 ImageTextContrastiveLossV2 用的 Hard Negative Mining
        class_weights_dict (dict, optional): 給 ImageTextContrastiveLossV2 用的類別權重
        memory_size (int, optional): 用於 Prototype 對比損失的記憶庫大小
        momentum (float, optional): 用於 Prototype 對比損失的動量更新率
        positive_weighting_beta (float, optional): 用於 Prototype 對比損失的正樣本加權參數
    """
    loss_type = loss_type.lower()
    print(f"初始化對比學習損失函數: {loss_type}")

    if loss_type == 'mse':
        print(f"  MSE Contrastive params: temperature={temperature}")
        return ContrastiveLossMSE(temperature=temperature)

    elif loss_type == 'supcon':
        if base_temperature is None:
            base_temperature = temperature
        print(f"  SupCon Style Contrastive params: temperature={temperature}, base_temperature={base_temperature}")
        return SupConStyleLoss(temperature=temperature, base_temperature=base_temperature)

    # 'image_text' 類型已經移除，僅使用更先進的 image_text_v2

    elif loss_type == 'image_text_v2':
        print(f"  Image-Text V2 Contrastive params: temperature={temperature}, top_k={top_k_negatives}, "
              f"use_class_weights={class_weights_dict is not None}")
        return ImageTextContrastiveLossV2(
            temperature=temperature,
            top_k_negatives=top_k_negatives,
            class_weights_dict=class_weights_dict
        )

    elif loss_type == 'prototype':
        # 使用默認值 0.999 如果沒有提供
        pw_beta = positive_weighting_beta if positive_weighting_beta is not None else 0.999
        print(f"  Prototype Contrastive params: temperature={temperature}, memory_size={memory_size}, "
              f"momentum={momentum}, use_hard_negatives=True, "
              f"positive_weighting_beta={pw_beta}")
        return PrototypeContrastiveLoss(
            temperature=temperature,
            memory_size=memory_size,
            momentum=momentum,
            use_hard_negatives=True,
            positive_weighting_beta=pw_beta
        )
    else:
        raise ValueError(f"未知的對比損失類型: {loss_type}. 請選擇 'mse', 'supcon', 'image_text_v2' 或 'prototype'.")


# --- 新增：計算類別權重的輔助函數 ---
def calculate_class_weights(labels_dict, beta=0.999, smoothing=1e-8):
    """
    根據標籤頻率計算類別權重 (類似 Class-Balanced Loss)。
    Args:
        labels_dict (dict): {attr: Tensor([N, num_classes])} 多熱標籤。
        beta (float): 平滑因子 (建議 0.9~0.9999)，越大時權重差異越大。
        smoothing (float): 防止除以 0 的微小常數。
    Returns:
        dict: {attr: Tensor([num_classes])}
    """
    class_weights = {}
    print(f"計算類別權重 (beta={beta})...")

    for attr, labels in labels_dict.items():
        if labels is None or labels.numel() == 0 or labels.size(1) == 0:
            print(f"  屬性 '{attr}' 標籤無效，跳過權重計算。")
            class_weights[attr] = None
            continue

        num_classes = labels.size(1)
        counts = labels.sum(dim=0).float()  # [num_classes]

        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / (effective_num + smoothing)

        # 使權重總和約為 num_classes
        weights = weights / torch.sum(weights) * num_classes

        # 可額外對權重做 clamp 防止極端值
        # weights = torch.clamp(weights, min=0.1, max=10.0)

        class_weights[attr] = weights.detach()
        print(f"  屬性 '{attr}' 權重計算完成。Min: {weights.min():.4f}, Max: {weights.max():.4f}")

    return class_weights
