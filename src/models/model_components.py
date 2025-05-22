# model_components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##################################################
# 1) Visual-Semantic Interaction
##################################################
class VisualSemanticInteraction(nn.Module):
    """
    視覺語義互動模組 (VSI)
    visual_features 作為 Query, prompts 作為 Key/Value
    """
    def __init__(self, prompt_dim, num_heads=8, dropout_rate=0.2):
        super(VisualSemanticInteraction, self).__init__()
        self.num_heads = num_heads
        self.scale = prompt_dim ** -0.5
        # q_linear 作用於 visual_features
        self.q_linear = nn.Linear(prompt_dim, prompt_dim)
        # k_linear 和 v_linear 作用於 prompts
        self.k_linear = nn.Linear(prompt_dim, prompt_dim)
        self.v_linear = nn.Linear(prompt_dim, prompt_dim)
        self.out_linear = nn.Linear(prompt_dim, prompt_dim)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)

    def forward(self, visual_features, prompts):
        """
        visual_features: [B, v, d] (v 通常是 1，代表全局視覺特徵)
        prompts:         [B, c, d] (c 是類別/prompt 數量)

        返回: 精煉後的視覺特徵 [B, v, d]
        """
        B, v, d = visual_features.size()
        B, c, d = prompts.size()

        # Q, K, V 的來源
        Q = self.q_linear(visual_features) # Query 來自 visual_features
        K = self.k_linear(prompts)         # Key 來自 prompts
        V = self.v_linear(prompts)         # Value 來自 prompts

        # 重塑成 [B, num_heads, seq_len, dim_per_head]
        Q = Q.view(B, v, self.num_heads, d // self.num_heads).transpose(1, 2) # [B, h, v, d/h]
        K = K.view(B, c, self.num_heads, d // self.num_heads).transpose(1, 2) # [B, h, c, d/h]
        V = V.view(B, c, self.num_heads, d // self.num_heads).transpose(1, 2) # [B, h, c, d/h]

        # 計算注意力分數 (Query attends to Keys)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, h, v, c]

        attn_probs = F.softmax(attn_scores, dim=-1) # 在 Key (prompts) 維度 softmax
        attn_probs = self.attn_dropout(attn_probs)

        # 用注意力權重聚合 Values (來自 prompts)
        attn_output = torch.matmul(attn_probs, V) # [B, h, v, d/h]

        # 重塑回 [B, v, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, v, d)
        output = self.out_linear(attn_output)
        output = self.out_dropout(output)
        # 返回精煉後的視覺特徵
        return output

##################################################
# 2) Multi-Prompt Initializer
##################################################
class MultiPromptInitializer(nn.Module):
    """
    使用直接的 MLP (text_embed_dim -> hidden_dim -> prompt_dim)
    將文字嵌入轉成 prompt。
    """
    def __init__(self,
                 attributes,
                 text_embed_dim,
                 prompt_dim,
                 hidden_dim,
                 dropout_rate=0.2):
        super(MultiPromptInitializer, self).__init__()
        self.attribute_names = attributes

        print(f"  [MultiPromptInitializer] Hidden Dim: {hidden_dim}")

        self.initializers = nn.ModuleDict({
            attr: nn.Sequential(
                # 直接使用 hidden_dim 的 MLP 結構
                nn.Linear(text_embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, prompt_dim),
                nn.LayerNorm(prompt_dim)
            ) for attr in self.attribute_names
        })

    def forward(self, text_embeddings_dict):
        prompts_dict = {}
        for attr in self.attribute_names:
            if attr in text_embeddings_dict:
                prompts_dict[attr] = self.initializers[attr](text_embeddings_dict[attr])
        return prompts_dict

##################################################
# 3) FiLM (Feature-wise Linear Modulation) Fusion
##################################################
class FiLMFusion(nn.Module):
    def __init__(self, prompt_dim, hidden_dim=None):
        super(FiLMFusion, self).__init__()
        if hidden_dim is None:
            hidden_dim = prompt_dim // 2
        self.gamma_generator = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, prompt_dim)
        )
        self.beta_generator = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, prompt_dim)
        )
    def forward(self, random_prompts, clip_prompts):
        gamma = self.gamma_generator(random_prompts)
        beta = self.beta_generator(random_prompts)
        return gamma * clip_prompts + beta

##################################################
# 4) ContrastiveProjector
##################################################
class ContrastiveProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout_rate=0.2):
        super(ContrastiveProjector, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim)
        )
    def forward(self, x):
        return self.projector(x)

##################################################
# 5) 主模型 MultiCPRFL
##################################################
class MultiCPRFL(nn.Module):
    def __init__(self,
                 attributes,
                 text_embed_dim,
                 prompt_dim,
                 hidden_dim,
                 num_classes_dict,
                 image_input_dim,
                 use_vsi=True,
                 fusion_mode="concat",
                 use_contrastive=True,
                 dropout_rate=0.2,
                 vsi_num_heads=8
                 ):
        super(MultiCPRFL, self).__init__()
        self.use_vsi = use_vsi
        self.fusion_mode = fusion_mode
        self.use_contrastive = use_contrastive

        # 視覺投影層直接使用 hidden_dim
        visual_hidden_dim = hidden_dim
        print(f"[MultiCPRFL] Visual Hidden Dim: {visual_hidden_dim}")

        # 建立視覺投影層
        self.visual_proj = nn.Sequential(
            nn.Linear(image_input_dim, visual_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(visual_hidden_dim, prompt_dim),
            nn.LayerNorm(prompt_dim)
            )

        # Prompt 初始化器 (使用 hidden_dim)
        if fusion_mode in ["clip", "concat", "weighted", "film"]:
            self.prompt_initializer = MultiPromptInitializer(
                attributes,
                text_embed_dim,
                prompt_dim,
                hidden_dim,
                dropout_rate=dropout_rate
            )

        if fusion_mode in ["random", "concat", "film"]:
            self.random_prompts = nn.ParameterDict({
                attr: nn.Parameter(torch.randn(num_classes_dict[attr], prompt_dim))
                for attr in attributes
            })
        if fusion_mode == "concat":
            self.fusion_concat = nn.ModuleDict({
                attr: nn.Sequential(
                    nn.Linear(2 * prompt_dim, prompt_dim),
                    nn.Dropout(dropout_rate)
                ) for attr in attributes
            })

        if fusion_mode == "film":
            self.fusion_film = nn.ModuleDict({
                attr: FiLMFusion(prompt_dim, hidden_dim=hidden_dim)
                for attr in attributes
            })

        # VSI
        self.vsi = VisualSemanticInteraction(prompt_dim, num_heads=vsi_num_heads, dropout_rate=dropout_rate) if use_vsi else None

        # 對比學習投影器
        if use_contrastive:
            contrastive_hidden_dim = prompt_dim // 2
            self.contrastive_projector = ContrastiveProjector(prompt_dim, prompt_dim, contrastive_hidden_dim, dropout_rate)

        self.attributes = attributes
        self.num_classes_dict = num_classes_dict
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, visual_features, text_embeddings_dict):
        B = visual_features.size(0)
        # 1. 視覺特徵投影
        proj_visual = self.visual_proj(visual_features)  # [B, prompt_dim]

        # 2. 生成對比學習特徵
        contrastive_features = self.contrastive_projector(proj_visual) if self.use_contrastive else None

        # 3. 獲取/融合 Prompts
        if self.fusion_mode == "clip":
            # prompt_initializer 使用 hidden_dim
            prompts_dict = self.prompt_initializer(text_embeddings_dict)
        elif self.fusion_mode == "random":
            prompts_dict = {attr: self.random_prompts[attr] for attr in self.attributes}
        else:
            # clip_prompts 使用 hidden_dim
            clip_prompts = self.prompt_initializer(text_embeddings_dict)
            random_prompts = {attr: self.random_prompts[attr] for attr in self.attributes}
            fused_prompts = {}
            for attr in self.attributes:
                R = random_prompts[attr]
                C = clip_prompts[attr]
                if self.fusion_mode == "concat":
                    fused = self.fusion_concat[attr](torch.cat([R, C], dim=-1))
                elif self.fusion_mode == "film":
                    fused = self.fusion_film[attr](R, C)
                else:
                    raise ValueError(f"不支援的 fusion_mode: {self.fusion_mode}")
                fused_prompts[attr] = fused
            prompts_dict = fused_prompts

        # 4. 計算 Logits
        logits_dict = {}
        final_prompts_dict = {}

        for attr, prompt_mat in prompts_dict.items():
            prompts_expanded = prompt_mat.unsqueeze(0).expand(B, -1, -1)
            final_prompts_dict[attr] = prompt_mat

            if self.use_vsi and self.vsi is not None:
                proj_visual_seq = proj_visual.unsqueeze(1)
                refined_visual = self.vsi(proj_visual_seq, prompts_expanded)
                refined_visual = refined_visual.squeeze(1)
                refined_visual_norm = F.normalize(refined_visual, p=2, dim=-1)
                prompt_mat_norm = F.normalize(prompt_mat, p=2, dim=-1)
                logits = torch.matmul(refined_visual_norm, prompt_mat_norm.t()) * self.logit_scale.exp()
            else:
                proj_visual_norm = F.normalize(proj_visual, p=2, dim=-1)
                prompt_mat_norm = F.normalize(prompt_mat, p=2, dim=-1)
                logits = torch.matmul(proj_visual_norm, prompt_mat_norm.t()) * self.logit_scale.exp()

            logits_dict[attr] = logits

        # 返回結果
        if self.use_contrastive:
            return logits_dict, final_prompts_dict, contrastive_features
        else:
            return logits_dict, final_prompts_dict