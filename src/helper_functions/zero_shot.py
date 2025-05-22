# src/helper_functions/zero_shot.py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, average_precision_score, f1_score
from rich.console import Console
from rich.table import Table
from utils.util import create_progress_bar

console = Console()

def initialize_clip_text_embeddings(attributes, categories_dict, clip_model, device, index_data):
    """
    根據每個屬性的所有類別，生成對應的 CLIP 文字嵌入。
    與 train_eval.py 中的 initialize_multi_category_embeddings 類似，但直接使用 CLIP 模型。
    """
    # 給不同屬性的提示詞模板
    prompt_templates = {
        'crop': "一張{}植物的照片",
        'part': "植物{}的特寫，一個特定的植物部位",
        'symptomCategories': "帶有{}類型症狀的植物疾病：",
        'symptomTags': "具有{}特定症狀的植物："
    }

    # 建立一個 dictionary 儲存各 id 對應的英文描述
    descriptions = {}

    # 1. 載入作物描述
    for item in index_data['crops']:
        descriptions[item['id']] = item['description']

    # 2. 載入部位描述
    for item in index_data['parts']:
        descriptions[item['id']] = item['description']

    # 3. 載入症狀類別與標籤描述
    for category in index_data['symptomCategories']:
        cat_desc = category['description']
        # 假設英文描述在括號內
        if '(' in cat_desc and ')' in cat_desc:
            eng_desc = cat_desc.split('(')[1].rstrip(')')
        else:
            eng_desc = cat_desc
        descriptions[category['id']] = eng_desc

        # 對應此症狀類別下各 tag
        for tag in category['tags']:
            tag_desc = tag['description']
            if '(' in tag_desc and ')' in tag_desc:
                eng_tag_desc = tag_desc.split('(')[1].rstrip(')')
            else:
                eng_tag_desc = tag_desc
            descriptions[tag['id']] = eng_tag_desc

    # 依照屬性生成 prompt 文本
    embeddings_dict = {}
    for attr in attributes:
        cat_ids = categories_dict[attr]
        text_prompts = []
        for cid in cat_ids:
            if cid in descriptions:
                if attr == 'crop':
                    text = f"a photo of {descriptions[cid]} plant, showing plant features and characteristics"
                elif attr == 'part':
                    text = f"closeup of plant {descriptions[cid].lower()}, a specific part of the plant"
                elif attr == 'symptomCategories':
                    text = f"plant disease showing {descriptions[cid]} symptoms, a category of plant disease signs"
                elif attr == 'symptomTags':
                    text = f"plant disease with {descriptions[cid]}, a specific symptom on plant"
                else:
                    text = prompt_templates[attr].format(descriptions[cid])
            else:
                # 若找不到對應描述，直接使用 ID 產生 prompt
                text = prompt_templates[attr].format(cid)

            text_prompts.append(text)

        with torch.no_grad():
            # clip_model.encode() 回傳 numpy array 或 tensor
            clip_emb = clip_model.encode(text_prompts)  # shape = [num_classes, embed_dim]
        clip_emb = torch.tensor(clip_emb)
        # L2 normalize
        clip_emb = F.normalize(clip_emb, dim=-1)
        # 存到字典
        embeddings_dict[attr] = clip_emb.to(device).float()

    return embeddings_dict


def evaluate_clip_zero_shot(
    attributes,
    test_data,
    test_labels_dict,
    label_to_idx_dict,
    idx_to_label_dict,
    index_data,
    clip_model,
    device,
    batch_size=32
):
    """
    零訓練的 CLIP 評估函數 - 直接使用 CLIP 嵌入和餘弦相似度進行預測
    """
    console.print("\n[bold cyan]=== 開始評估 CLIP 零訓練模型 (Zero-Shot Baseline) ===[/bold cyan]")

    # 嘗試從 data_utils 匯入 compute_head_medium_tail
    try:
        from src.data_loader.data_utils import compute_head_medium_tail
        hmt_info = compute_head_medium_tail()
        can_analyze_hmt = True
    except ImportError:
        console.print("[yellow]警告: 無法導入 compute_head_medium_tail，不進行 H/M/T 分析。[/yellow]")
        hmt_info = {}
        can_analyze_hmt = False

    # 準備文本嵌入
    categories_dict = {}
    for attr in attributes:
        if attr == 'crop':
            categories_dict[attr] = [item["id"] for item in index_data["crops"]]
        elif attr == 'part':
            categories_dict[attr] = [item["id"] for item in index_data["parts"]]
        elif attr == 'symptomCategories':
            categories_dict[attr] = [item["id"] for item in index_data["symptomCategories"]]
        elif attr == 'symptomTags':
            all_tags = []
            for cat in index_data["symptomCategories"]:
                for tag in cat["tags"]:
                    all_tags.append(tag["id"])
            categories_dict[attr] = all_tags

    # 獲取文字嵌入
    text_embeddings_dict = initialize_clip_text_embeddings(attributes, categories_dict, clip_model, device, index_data)

    # 創建 DataLoader
    dataset = TensorDataset(
        test_data['image_embeddings'],
        *[test_labels_dict[attr] for attr in attributes]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 評估
    all_probs = {attr: [] for attr in attributes}
    all_labels = {attr: [] for attr in attributes}

    with torch.no_grad():
        for batch in dataloader:
            img_batch = batch[0].to(device).float()
            
            # 對影像嵌入進行 L2 正規化
            img_batch_norm = F.normalize(img_batch, p=2, dim=1)
            
            # 計算每個屬性的相似度分數
            for i, attr in enumerate(attributes):
                if attr not in text_embeddings_dict:
                    continue
                    
                # 獲取對應屬性的文字嵌入
                text_emb = text_embeddings_dict[attr]  # [num_classes, embed_dim]
                
                # 計算餘弦相似度 (dot product of normalized vectors)
                similarity = torch.matmul(img_batch_norm, text_emb.t())  # [batch_size, num_classes]
                
                # 保存相似度作為"概率"(雖然不是真正的概率，但可用於排序和評估)
                # 將範圍從 [-1, 1] 調整到 [0, 1] 以便與標籤比較
                similarity = (similarity + 1) / 2
                all_probs[attr].append(similarity.cpu().numpy())
                all_labels[attr].append(batch[i+1].cpu().numpy())

    # 計算評估指標
    results = {}
    for attr in attributes:
        if attr not in all_probs or not all_probs[attr]:
            console.print(f"[yellow]警告: 屬性 '{attr}' 沒有有效預測，跳過評估。[/yellow]")
            continue
            
        p = np.concatenate(all_probs[attr], axis=0)
        l = np.concatenate(all_labels[attr], axis=0)
        num_classes = p.shape[1]

        ap_list = []
        for c in range(num_classes):
            if np.sum(l[:, c]) == 0:
                continue
            ap_list.append(average_precision_score(l[:, c], p[:, c]))
        mAP_all = np.mean(ap_list)*100 if len(ap_list) > 0 else 0

        # 使用 0.5 閾值轉換為二進制預測
        preds = (p > 0.5).astype(np.int32)
        acc = np.mean(preds.flatten() == l.flatten())*100

        f1_list = []
        for c in range(num_classes):
            if np.sum(l[:, c]) == 0:
                continue
            f1_list.append(f1_score(l[:, c], preds[:, c], zero_division=0))
        overall_f1 = np.mean(f1_list)*100 if len(f1_list) > 0 else 0

        overall_report = classification_report(l.flatten(), preds.flatten(), output_dict=True, zero_division=0)

        # 若能做頭中尾 (HMT) 分析
        type_metrics = {}
        if can_analyze_hmt and attr in hmt_info:
            for typ in ["tail", "medium", "head"]:
                if typ not in hmt_info[attr]:
                    type_metrics[typ] = {"mAP": None, "acc": None, "f1": None}
                    continue
                label_ids = list(hmt_info[attr][typ].keys())
                type_indices = []
                for label_id in label_ids:
                    if label_id in label_to_idx_dict[attr]:
                        idx = label_to_idx_dict[attr][label_id]
                        type_indices.append(idx)
                if len(type_indices) == 0:
                    type_metrics[typ] = {"mAP": None, "acc": None, "f1": None}
                    continue
                p_sub = p[:, type_indices]
                l_sub = l[:, type_indices]

                ap_sub = []
                for j in range(len(type_indices)):
                    if np.sum(l_sub[:, j]) == 0:
                        continue
                    ap_sub.append(average_precision_score(l_sub[:, j], p_sub[:, j]))

                sub_mAP = np.mean(ap_sub)*100 if len(ap_sub) > 0 else None
                preds_sub = (p_sub > 0.5).astype(np.int32)
                acc_sub = np.mean(preds_sub.flatten() == l_sub.flatten())*100

                f1_sub_list = []
                for j in range(len(type_indices)):
                    if np.sum(l_sub[:, j]) == 0:
                        continue
                    f1_sub_list.append(f1_score(l_sub[:, j], preds_sub[:, j], zero_division=0))
                sub_f1 = np.mean(f1_sub_list)*100 if len(f1_sub_list) > 0 else None

                type_metrics[typ] = {
                    "mAP": sub_mAP,
                    "acc": acc_sub,
                    "f1": sub_f1
                }
        else:
            type_metrics = {
                "tail": {"mAP": None, "acc": None, "f1": None},
                "medium": {"mAP": None, "acc": None, "f1": None},
                "head": {"mAP": None, "acc": None, "f1": None},
            }

        results[attr] = {
            "overall_mAP": mAP_all,
            "overall_acc": acc,
            "overall_f1": overall_f1,
            "report": overall_report,
            "type_metrics": type_metrics
        }

        # 顯示表格
        table = Table(title=f"{attr} 零訓練 CLIP 評估")
        table.add_column("Overall mAP")
        table.add_column("Overall F1")
        table.add_column("Head mAP")
        table.add_column("Medium mAP")
        table.add_column("Tail mAP")
        table.add_column("Head F1")
        table.add_column("Medium F1")
        table.add_column("Tail F1")
        table.add_column("Overall Acc")

        head_mAP_val = type_metrics["head"]["mAP"]
        medium_mAP_val = type_metrics["medium"]["mAP"]
        tail_mAP_val = type_metrics["tail"]["mAP"]
        head_f1_val = type_metrics["head"]["f1"]
        medium_f1_val = type_metrics["medium"]["f1"]
        tail_f1_val = type_metrics["tail"]["f1"]

        table.add_row(
            f"{mAP_all:.2f}%",
            f"{overall_f1:.2f}%",
            f"{head_mAP_val:.2f}%" if head_mAP_val else "N/A",
            f"{medium_mAP_val:.2f}%" if medium_mAP_val else "N/A",
            f"{tail_mAP_val:.2f}%" if tail_mAP_val else "N/A",
            f"{head_f1_val:.2f}%" if head_f1_val else "N/A",
            f"{medium_f1_val:.2f}%" if medium_f1_val else "N/A",
            f"{tail_f1_val:.2f}%" if tail_f1_val else "N/A",
            f"{acc:.2f}%"
        )
        console.print(table)

    # 頭中尾分析
    console.print("\n[bold cyan]=== 頭中尾 (H/M/T) 分析 ===[/bold cyan]")
    for attr in attributes:
        if attr not in results:
            continue
        type_metrics = results[attr]["type_metrics"]
        console.print(f"[bold]{attr}[/bold] - Head/Medium/Tail 分析:")
        for t_type in ["head", "medium", "tail"]:
            tm = type_metrics[t_type]
            if tm["mAP"] is not None:
                console.print(
                    f"  {t_type.capitalize()} mAP: {tm['mAP']:.2f}%, F1: {tm['f1']:.2f}%"
                )
            else:
                console.print(f"  {t_type.capitalize()} - N/A")
                
    return results