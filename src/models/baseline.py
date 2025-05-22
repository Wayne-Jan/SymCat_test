# src/models/baseline.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineClassifier(nn.Module):
    """
    純圖像分類器作為baseline。
    這個模型只使用圖像嵌入，完全不使用文本嵌入或提示詞，也沒有對比學習。
    """
    def __init__(self, 
                 attributes, 
                 num_classes_dict, 
                 image_input_dim, 
                 hidden_dim=768, 
                 dropout_rate=0.2):
        super(BaselineClassifier, self).__init__()
        self.attributes = attributes
        self.num_classes_dict = num_classes_dict
        
        # 一個共享的視覺特徵提取器
        self.visual_feature_extractor = nn.Sequential(
            nn.Linear(image_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 為每個屬性創建獨立的分類頭
        self.classifiers = nn.ModuleDict({
            attr: nn.Linear(hidden_dim, num_classes_dict[attr])
            for attr in self.attributes
        })
    
    def forward(self, image_embeddings, *args, **kwargs):
        """
        前向傳播函數
        Args:
            image_embeddings: 圖像嵌入 [batch_size, image_input_dim]
            *args, **kwargs: 忽略所有其他輸入，如文本嵌入
            
        Returns:
            logits_dict: 每個屬性的logits {attr: tensor[batch_size, num_classes]}
        """
        # 提取視覺特徵
        visual_features = self.visual_feature_extractor(image_embeddings)
        
        # 為每個屬性計算logits
        logits_dict = {}
        for attr in self.attributes:
            logits_dict[attr] = self.classifiers[attr](visual_features)
            
        # 為了與原始模型保持接口一致，返回一個空的字典作為第二個返回值
        empty_dict = {}
            
        return logits_dict, empty_dict