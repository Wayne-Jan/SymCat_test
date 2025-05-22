# data_utils.py
import json
import torch
import numpy as np
from collections import defaultdict
from rich.console import Console

console = Console()

# 更新後的屬性列表
attributes = ['crop', 'part', 'symptomCategories', 'symptomTags']
index_keys = ['crops', 'parts', 'symptomCategories', 'symptomCategories']  # symptomTags 需要特殊處理

def load_index(index_json_path='SymCat.json'):
    """
    讀取 SymCat.json，並回傳:
      - label_to_idx_dict: {attr: {id字串: index}, ...}
      - idx_to_label_dict: {attr: {index: id字串}, ...}
      - index_data: 解析後的 index json 資料
    適應新的 symptomCategories 和 symptomTags 結構
    """
    with open(index_json_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    label_to_idx_dict = {}
    idx_to_label_dict = {}
    
    # 處理 crops 和 parts
    for attr, key in zip(['crop', 'part'], ['crops', 'parts']):
        label_to_idx_dict[attr] = {item["id"]: i for i, item in enumerate(index_data[key])}
        idx_to_label_dict[attr] = {i: item["id"] for i, item in enumerate(index_data[key])}
    
    # 處理 symptomCategories
    label_to_idx_dict['symptomCategories'] = {category["id"]: i for i, category in enumerate(index_data['symptomCategories'])}
    idx_to_label_dict['symptomCategories'] = {i: category["id"] for i, category in enumerate(index_data['symptomCategories'])}
    
    # 處理 symptomTags (從所有類別中收集)
    symptom_tags = []
    for category in index_data['symptomCategories']:
        for tag in category['tags']:
            symptom_tags.append(tag)
    
    label_to_idx_dict['symptomTags'] = {tag["id"]: i for i, tag in enumerate(symptom_tags)}
    idx_to_label_dict['symptomTags'] = {i: tag["id"] for i, tag in enumerate(symptom_tags)}

    return label_to_idx_dict, idx_to_label_dict, index_data

def convert_to_multihot(labels, label_to_idx):
    """
    將標籤轉換成 multi-hot 向量。
    labels 可能是單一字串或是字串列表。
    label_to_idx: 例如 {'SYM_CAT_01': 0, 'SYM_CAT_02': 1, ...}
    """
    num_classes = len(label_to_idx)
    vec = torch.zeros(num_classes, dtype=torch.long)

    def process_label(lbl):
        if lbl not in label_to_idx:
            console.print(f"[yellow]Warning: label '{lbl}' not found in mapping[/yellow]")
            return None
        return label_to_idx[lbl]

    if isinstance(labels, list):
        for lbl in labels:
            idx = process_label(lbl)
            if idx is not None:
                vec[idx] = 1
    else:
        idx = process_label(labels)
        if idx is not None:
            vec[idx] = 1

    return vec

def compute_head_medium_tail(nchu_sorted_data_json='NCHU_sorted_data_fixed.json'):
    """
    從 NCHU_sorted_data_fixed.json 中統計各屬性的標籤頻率，將其分為 tail / medium / head。
    處理新的多類屬性情況。
    回傳結構類似:
    {
      'crop': {
         'tail': { labelA: 次數, labelB: 次數, ... },
         'medium': {...},
         'head': {...},
         '33_percentile': X,
         '67_percentile': Y
      },
      'part': {...},
      'symptomCategories': {...},
      'symptomTags': {...}
    }
    """
    # Try to use the fixed file first, if not available fall back to the original
    try:
        with open(nchu_sorted_data_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[yellow]Warning: {nchu_sorted_data_json} not found, trying NCHU_sorted_data.json[/yellow]")
        try:
            with open("NCHU_sorted_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: NCHU_sorted_data.json not found either. H/M/T analysis will not be available.[/red]")
            return {}

    freq = {attr: defaultdict(int) for attr in attributes}
    for _, label_str in data.items():
        try:
            label_info = json.loads(label_str)
        except Exception as e:
            console.print(f"Error parsing label_str: {e}")
            continue
        
        # 處理 crop 和 part
        for attr in ['crop', 'part']:
            if attr not in label_info:
                continue
            value = label_info[attr]
            freq[attr][value] += 1
        
        # 處理 symptomCategories (多選)
        if 'symptomCategories' in label_info and isinstance(label_info['symptomCategories'], list):
            for category in label_info['symptomCategories']:
                freq['symptomCategories'][category] += 1
        
        # 處理 symptomTags (多選)
        if 'symptomTags' in label_info and isinstance(label_info['symptomTags'], list):
            for tag in label_info['symptomTags']:
                freq['symptomTags'][tag] += 1

    head_tail = {}
    for attr in attributes:
        counts = np.array(list(freq[attr].values()))
        if counts.size == 0:
            low_threshold = high_threshold = 0
        elif counts.size == 1:
            # 若只有一類，那它同時屬於 head & tail
            low_threshold = high_threshold = counts[0]
        else:
            low_threshold, high_threshold = np.percentile(counts, [33, 67])

        tail = {label: c for label, c in freq[attr].items() if c < low_threshold}
        medium = {label: c for label, c in freq[attr].items() if low_threshold <= c < high_threshold}
        head = {label: c for label, c in freq[attr].items() if c >= high_threshold}

        head_tail[attr] = {
            "33_percentile": low_threshold,
            "67_percentile": high_threshold,
            "tail": tail,
            "medium": medium,
            "head": head
        }

    return head_tail

def analyze_label_overlap(train_labels_dict, attributes):
    """
    分析訓練集中多標籤數據的標籤重疊情況，
    用於對比學習分析
    """
    overlap_stats = {}
    
    for attr in attributes:
        if attr not in train_labels_dict:
            continue
            
        labels = train_labels_dict[attr]  # [N, num_classes]
        N = labels.size(0)
        
        # 計算每個樣本的標籤數量
        label_counts = torch.sum(labels, dim=1)  # [N]
        avg_labels_per_sample = torch.mean(label_counts.float()).item()
        
        # 計算樣本對的重疊率統計
        overlap_matrix = torch.matmul(labels, labels.t())  # [N, N]
        
        # 移除對角線
        mask = torch.ones_like(overlap_matrix) - torch.eye(N)
        masked_overlap = overlap_matrix * mask
        
        # 計算重疊率分佈
        overlap_vals = masked_overlap.flatten().tolist()
        overlap_vals = [v for v in overlap_vals if v > 0]  # 僅考慮非零重疊
        
        if len(overlap_vals) > 0:
            avg_overlap = np.mean(overlap_vals)
            max_overlap = np.max(overlap_vals)
            
            # 計算重疊分佈
            overlap_dist = {}
            for v in overlap_vals:
                if v not in overlap_dist:
                    overlap_dist[v] = 0
                overlap_dist[v] += 1
                
            overlap_stats[attr] = {
                'avg_labels_per_sample': avg_labels_per_sample,
                'avg_overlap': avg_overlap,
                'max_overlap': max_overlap,
                'overlap_distribution': dict(sorted(overlap_dist.items()))
            }
    
    return overlap_stats