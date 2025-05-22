# train_eval.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, average_precision_score, f1_score
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table

# === 從 losses.py 導入所需函數、類別 ===
try:
    from src.loss_functions.loss import (
        get_loss_function,
        get_contrastive_loss,
        PromptSeparationLoss,
        calculate_class_weights
    )
except ImportError:
    print("錯誤：無法從 src.loss_functions.loss 導入。請確保文件存在並且路徑正確。")
    # 可以視需求提供備用方案，這裡僅示範簡單回退:
    def get_loss_function(loss_type, **kwargs):
        print("警告: src.loss_functions.loss 導入失敗，使用預設 BCEWithLogitsLoss。")
        return nn.BCEWithLogitsLoss()

    def get_contrastive_loss(**kwargs):
        print("警告: src.loss_functions.loss 導入失敗，對比損失將輸出 0。")
        return lambda *args, **kwargs: torch.tensor(0.0)

    def PromptSeparationLoss(**kwargs):
        print("警告: src.loss_functions.loss 導入失敗，提示詞分離損失將輸出 0。")
        return lambda *args, **kwargs: torch.tensor(0.0)

    def calculate_class_weights(**kwargs):
        print("警告: src.loss_functions.loss 導入失敗，無法計算類別權重，將返回空字典。")
        return {}

console = Console()

# 若有資料 key 與屬性名稱的對應需要，可在此定義
data_key_map = {
    "crop": "crop_labels",
    "part": "part_labels",
    "symptomCategories": "symptom_category_labels",
    "symptomTags": "symptom_tag_labels"
}

def initialize_multi_category_embeddings(attributes, categories_dict, clip_model, device, index_data):
    """
    根據每個屬性的所有類別，生成對應的 CLIP 文字嵌入。
    categories_dict: {attr: [category_id1, category_id2, ...], ...}
    index_data: SymCat.json 載入後的資料，包含各屬性的詳細描述。
    clip_model: 例如 sentence-transformer/CLIP 類的模型，其 encode() 可輸入文字列表回傳嵌入。
    """
    # 給不同屬性的提示詞模板 (可依需求自訂)
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
            # clip_model.encode() 回傳 numpy array 或 tensor，需視實際 CLIP 函數
            clip_emb = clip_model.encode(text_prompts)  # shape = [num_classes, embed_dim]
        clip_emb = torch.tensor(clip_emb)
        # L2 normalize
        clip_emb = F.normalize(clip_emb, dim=-1)
        # 存到字典
        embeddings_dict[attr] = clip_emb.to(device).float()

    return embeddings_dict


def train_multi_cprfl(
    model,
    attributes,
    train_data,
    val_data,
    train_labels_dict,
    val_labels_dict,
    label_to_idx_dict,
    idx_to_label_dict,
    index_data,
    clip_model,
    device,
    test_data=None,
    test_labels_dict=None,
    epochs=5,
    batch_size=32,
    loss_type="bce",
    contrastive_loss_type="supcon",  # 也可設定為 'image_text_v2' 或 'prototype' 等
    lr=5e-5,
    weight_decay=1e-4,
    lr_scheduler_type="cosine",
    warmup_epochs=1,
    lambda_contrastive_start=0.2,
    lambda_contrastive_end=0.8,
    lambda_schedule="linear",
    contrastive_temperature=0.1,
    beta_prompt_separation=0.0,      # 提示詞分離損失的權重
    contrastive_top_k_negatives=None,# image_text_v2 的 Hard Negative Mining
    positive_weighting_beta=None,    # image_text_v2 和 prototype 的正樣本加權 beta
    memory_size=4096,                # prototype 的記憶庫大小
    momentum=0.9,                    # prototype 的原型更新動量
    early_stopping_patience=10,
    map_weight=0.5,                  # mAP 權重，默認為 0.5，表示 mAP 和 F1 各佔 50%
    save_best_combined=True          # 是否保存綜合分數最佳的模型，默認為 True
):
    """
    訓練函數:
      - model: 模型實例 (需支持多屬性輸出，且若有對比功能則 model.use_contrastive=True)
      - attributes: 要處理的屬性列表，如 ["crop", "part", "symptomCategories", "symptomTags"]
      - train_data/val_data: dict，至少需包含 'image_embeddings'，shape=(N, embed_dim)
      - train_labels_dict/val_labels_dict: {attr: Tensor([N, num_classes])} 的多熱標籤
      - label_to_idx_dict/idx_to_label_dict: {attr: {label_id: idx}} or 反向，用於後處理 (可選)
      - index_data: SymCat.json 等資訊，用於生成文字 prompt
      - clip_model: 用於產生文字嵌入的模型 (encode 函數)
      - device: cuda or cpu
      - test_data/test_labels_dict: 若提供則可在每個 epoch 顯示 Test 集結果
      - epochs/batch_size: 訓練參數
      - loss_type: "bce", "asl", "dbfocal", "mls" 等
      - contrastive_loss_type: "mse", "supcon", "image_text_v2", "prototype" ...
      - lr_scheduler_type: "cosine", "step", "plateau"
      - lambda_contrastive_start / end / schedule: 控制對比損失在訓練中權重的調度
      - beta_prompt_separation: 若模型會輸出 prompt embedding，可用於對 prompts 做分離損失 (預設 0.0 表示不啟用)
      - contrastive_top_k_negatives: 給 image_text_v2 作 Hard Negative Mining (None=全部)
      - positive_weighting_beta: 給 image_text_v2 和 prototype 作正樣本加權 (None=無)
      - memory_size: 給 prototype 用的記憶庫大小 (預設 4096)
      - momentum: 給 prototype 用的原型更新動量 (預設 0.9)
      - early_stopping_patience: 若驗證指標在 patience 階段內無提升，則提早停止訓練
      - map_weight: 綜合分數中 mAP 的權重 (1-map_weight 為 F1 的權重)
      - save_best_combined: 是否根據綜合分數保存最佳模型 (True) 還是只考慮 mAP (False)
    """

    console.print("\n[bold cyan]=== 開始訓練多重屬性模型 (含優化改進) ===[/bold cyan]")
    console.print(f"分類損失類型: {loss_type}")
    console.print(f"對比損失類型: {contrastive_loss_type}")
    console.print(f"初始學習率: {lr}, 權重衰減: {weight_decay}")
    console.print(f"學習率調度: {lr_scheduler_type}, 預熱期: {warmup_epochs} epochs")
    console.print(f"對比學習權重 (Lambda): {lambda_contrastive_start} → {lambda_contrastive_end}, 調度: {lambda_schedule}")
    if contrastive_loss_type != 'mse':
        console.print(f"對比學習溫度: {contrastive_temperature}")
    if contrastive_loss_type == 'image_text_v2':
        console.print(f"  Hard Negative K: {contrastive_top_k_negatives if contrastive_top_k_negatives else '所有'}")
        console.print(f"  正樣本加權 Beta: {positive_weighting_beta if positive_weighting_beta else '無'}")
    console.print(f"提示詞分離損失權重 (Beta): {beta_prompt_separation}")
    console.print(f"Early Stopping Patience: {early_stopping_patience} epochs")
    
    if save_best_combined:
        console.print(f"模型選擇策略: 綜合分數 (mAP 權重: {map_weight:.2f}, F1 權重: {1-map_weight:.2f})")
    else:
        console.print(f"模型選擇策略: 僅 mAP")
    console.print("")

    # 1. 準備每個屬性的所有 category id
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

    # 2. 生成文本嵌入 (CLIP prompts)
    text_embeddings_dict = initialize_multi_category_embeddings(attributes, categories_dict, clip_model, device, index_data)

    # 3. 構建 DataLoader
    train_dataset = TensorDataset(
        train_data['image_embeddings'],
        *[train_labels_dict[attr] for attr in attributes]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    val_dataset = TensorDataset(
        val_data['image_embeddings'],
        *[val_labels_dict[attr] for attr in attributes]
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    has_test_data = False
    test_dataloader = None
    if test_data is not None and test_labels_dict is not None:
        test_dataset = TensorDataset(
            test_data['image_embeddings'],
            *[test_labels_dict[attr] for attr in attributes]
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        has_test_data = True

    # 4. 計算正樣本類別權重 (若需要 image_text_v2 或 prototype + positive_weighting_beta)
    class_weights_for_contrastive = None
    if (contrastive_loss_type in ['image_text_v2', 'prototype']) and positive_weighting_beta is not None and positive_weighting_beta > 0:
        console.print(f"根據訓練集標籤計算正樣本類別權重 (beta={positive_weighting_beta})...")
        class_weights_for_contrastive = calculate_class_weights(
            train_labels_dict,
            beta=positive_weighting_beta
        )

    # 5. 建立損失函數
    #   (1) 分類損失 (ASL / BCE / DBFocal / MLS ... )
    #   (2) 對比損失 (MSE / SupCon / ImageText / ImageTextV2)
    #   (3) 提示詞分離損失 (選用)
    loss_obj = get_loss_function(
        loss_type,
        multi_hot_labels_dict=(train_labels_dict if loss_type == "dbfocal" else None)
    )
    contrastive_loss_args = {
        "loss_type": contrastive_loss_type,
        "temperature": contrastive_temperature,
        "top_k_negatives": contrastive_top_k_negatives,
        "class_weights_dict": class_weights_for_contrastive
    }

    # 為 prototype 模式添加額外參數
    if contrastive_loss_type == 'prototype':
        contrastive_loss_args["memory_size"] = memory_size
        contrastive_loss_args["momentum"] = momentum
        contrastive_loss_args["positive_weighting_beta"] = positive_weighting_beta

    contrastive_loss_fn = get_contrastive_loss(**contrastive_loss_args)
    prompt_separation_loss_fn = PromptSeparationLoss()

    # 6. 建立優化器與學習率調度器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    warmup_scheduler = None

    if lr_scheduler_type == "cosine":
        # 餘弦退火，T_max = 總 epoch - 預熱期
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=lr/10
        )
    elif lr_scheduler_type == "step":
        # 每 (大約訓練 1/5 總 epoch) 就衰減
        step_size = max(1, epochs // 5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.7)
    elif lr_scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7,
            patience=max(1, early_stopping_patience // 4),
            verbose=True
        )

    # warmup
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs * len(train_dataloader)
        )

    # 對比損失權重的調度函數
    def get_contrastive_lambda(cur_epoch, total_epochs, schedule_type):
        # 注意 cur_epoch 從 0 開始計算更直觀
        if schedule_type == "linear":
            return lambda_contrastive_start + (lambda_contrastive_end - lambda_contrastive_start) * (cur_epoch / total_epochs)
        elif schedule_type == "cosine":
            return lambda_contrastive_start + (lambda_contrastive_end - lambda_contrastive_start) \
                   * 0.5 * (1 - np.cos(np.pi * cur_epoch / total_epochs))
        elif schedule_type == "exp":
            return lambda_contrastive_start + (lambda_contrastive_end - lambda_contrastive_start) \
                   * (1 - np.exp(-5 * cur_epoch / total_epochs))
        else:
            return lambda_contrastive_start

    # Early Stopping 相關
    best_val_map_for_stopping = 0.0
    best_val_f1_for_stopping = 0.0
    best_combined_score = 0.0
    early_stopping_counter = 0
    best_epoch = 0
    best_state = None
    final_epoch_ran = 0

    # === 主要訓練迴圈 ===
    for epoch in range(1, epochs + 1):
        final_epoch_ran = epoch
        current_lambda = get_contrastive_lambda(epoch - 1, epochs, lambda_schedule)
        model.train()

        total_loss = 0.0
        total_class_loss = 0.0
        total_contr_loss = 0.0
        total_prompt_sep_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            img_batch = batch[0].to(device).float()
            label_batches = batch[1:]
            current_labels_dict = {
                attr: label_batches[i].to(device).float()
                for i, attr in enumerate(attributes)
            }

            # forward
            contrastive_features = None
            final_prompts_dict = None

            # 模型需看是否返回 (logits_dict, prompts_dict, contrastive_features)
            if hasattr(model, 'use_contrastive') and model.use_contrastive:
                logits_dict, final_prompts_dict, contrastive_features = model(img_batch, text_embeddings_dict)
            else:
                logits_dict, final_prompts_dict = model(img_batch, text_embeddings_dict)

            # 1) 分類損失
            classification_loss = 0.0
            for i, attr in enumerate(attributes):
                logits = logits_dict[attr]
                gt = label_batches[i].to(device).float()

                if isinstance(loss_obj, dict):
                    # dbfocal 的情況: loss_obj 為 {attr: DBFocalLoss實例}
                    if attr in loss_obj:
                        classification_loss += loss_obj[attr](logits, gt)
                    else:
                        # 如果 attr 不在 dbfocal 字典中，就用預設 BCE
                        default_loss = nn.BCEWithLogitsLoss()
                        classification_loss += default_loss(logits, gt)
                else:
                    # 其他損失函數 (ASL、BCE、MLS ...)
                    classification_loss += loss_obj(logits, gt)

            # 2) 對比損失
            contrastive_loss = torch.tensor(0.0, device=device)
            if contrastive_features is not None and final_prompts_dict is not None:
                if contrastive_loss_type in ['image_text_v2', 'prototype']:
                    # 需要 prompts_dict & labels_dict
                    contrastive_loss = contrastive_loss_fn(
                        contrastive_features,
                        final_prompts_dict,
                        current_labels_dict,
                        attributes
                    )
                else:
                    # supcon / mse
                    contrastive_loss = contrastive_loss_fn(
                        contrastive_features,
                        current_labels_dict,
                        attributes
                    )

            # 3) 提示詞分離損失
            prompt_separation_loss = torch.tensor(0.0, device=device)
            if final_prompts_dict is not None and beta_prompt_separation > 0:
                prompt_separation_loss = prompt_separation_loss_fn(final_prompts_dict, attributes)

            # 總損失
            loss = classification_loss + current_lambda * contrastive_loss + beta_prompt_separation * prompt_separation_loss

            # 反向傳播、優化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # warmup
            if epoch <= warmup_epochs and warmup_scheduler is not None:
                warmup_scheduler.step()

            # 累加
            total_loss += loss.item()
            total_class_loss += classification_loss.item()
            total_contr_loss += contrastive_loss.item()
            total_prompt_sep_loss += prompt_separation_loss.item()

        # 這個 epoch 的平均損失
        avg_loss = total_loss / len(train_dataloader)
        avg_class_loss = total_class_loss / len(train_dataloader)
        avg_contr_loss = total_contr_loss / len(train_dataloader)
        avg_prompt_sep_loss = total_prompt_sep_loss / len(train_dataloader)
        current_lr = optimizer.param_groups[0]['lr']

        # === 驗證 ===
        model.eval()
        val_probs = {attr: [] for attr in attributes}
        val_labels_storage = {attr: [] for attr in attributes}

        with torch.no_grad():
            for batch in val_dataloader:
                val_img_batch = batch[0].to(device).float()
                if hasattr(model, 'use_contrastive') and model.use_contrastive:
                    logits_dict, _, _ = model(val_img_batch, text_embeddings_dict)
                else:
                    logits_dict, _ = model(val_img_batch, text_embeddings_dict)

                for i, attr in enumerate(attributes):
                    val_probs[attr].append(torch.sigmoid(logits_dict[attr]).cpu().numpy())
                    val_labels_storage[attr].append(batch[i+1].cpu().numpy())

        val_map_scores, val_f1_scores = [], []
        for attr in attributes:
            p = np.concatenate(val_probs[attr], axis=0)
            l = np.concatenate(val_labels_storage[attr], axis=0)
            ap_list = [
                average_precision_score(l[:, c], p[:, c])
                for c in range(p.shape[1])
                if np.sum(l[:, c]) > 0
            ]
            val_mAP_attr = np.mean(ap_list) * 100 if ap_list else 0
            val_map_scores.append(val_mAP_attr)

            preds = (p > 0.5).astype(np.int32)
            f1_list = [
                f1_score(l[:, c], preds[:, c], zero_division=0)
                for c in range(p.shape[1])
                if np.sum(l[:, c]) > 0
            ]
            val_f1_attr = np.mean(f1_list) * 100 if f1_list else 0
            val_f1_scores.append(val_f1_attr)

        val_overall_map = np.mean(val_map_scores)
        val_overall_f1 = np.mean(val_f1_scores)

        # scheduler 更新
        if epoch > warmup_epochs:
            if lr_scheduler_type == "plateau" and scheduler is not None:
                scheduler.step(val_overall_map)
            elif scheduler is not None:
                scheduler.step()

        # === 測試 (若有) ===
        test_overall_map_str, test_overall_f1_str = "", ""
        if has_test_data:
            test_probs = {attr: [] for attr in attributes}
            test_labels_storage = {attr: [] for attr in attributes}
            with torch.no_grad():
                for batch in test_dataloader:
                    test_img_batch = batch[0].to(device).float()
                    if hasattr(model, 'use_contrastive') and model.use_contrastive:
                        logits_dict, _, _ = model(test_img_batch, text_embeddings_dict)
                    else:
                        logits_dict, _ = model(test_img_batch, text_embeddings_dict)

                    for i, attr in enumerate(attributes):
                        test_probs[attr].append(torch.sigmoid(logits_dict[attr]).cpu().numpy())
                        test_labels_storage[attr].append(batch[i+1].cpu().numpy())

            test_map_scores, test_f1_scores = [], []
            for attr in attributes:
                p = np.concatenate(test_probs[attr], axis=0)
                l = np.concatenate(test_labels_storage[attr], axis=0)
                ap_list = [
                    average_precision_score(l[:, c], p[:, c])
                    for c in range(p.shape[1])
                    if np.sum(l[:, c]) > 0
                ]
                test_mAP_attr = np.mean(ap_list) * 100 if ap_list else 0
                test_map_scores.append(test_mAP_attr)

                preds = (p > 0.5).astype(np.int32)
                f1_list = [
                    f1_score(l[:, c], preds[:, c], zero_division=0)
                    for c in range(p.shape[1])
                    if np.sum(l[:, c]) > 0
                ]
                test_f1_attr = np.mean(f1_list) * 100 if f1_list else 0
                test_f1_scores.append(test_f1_attr)

            test_overall_map = np.mean(test_map_scores)
            test_overall_f1 = np.mean(test_f1_scores)
            test_overall_map_str = f"Test mAP: {test_overall_map:.2f}%, "
            test_overall_f1_str = f"Test F1: {test_overall_f1:.2f}%"

        # 打印日誌
        console.print(
            f"Epoch [{epoch}/{epochs}] | "
            f"LR: {current_lr:.6f} | "
            f"λ: {current_lambda:.4f} | "
            f"β: {beta_prompt_separation:.4f} | "
            f"Loss: {avg_loss:.4f} (Cls: {avg_class_loss:.4f}, "
            f"Ctr: {avg_contr_loss:.4f}, PrmSep: {avg_prompt_sep_loss:.4f}), "
            f"Val mAP: {val_overall_map:.2f}%, Val F1: {val_overall_f1:.2f}%, "
            f"{test_overall_map_str}{test_overall_f1_str}"
        )

        # === Early Stopping ===
        if save_best_combined:
            # 計算綜合分數 (mAP 和 F1 的加權平均)
            combined_score = map_weight * val_overall_map + (1 - map_weight) * val_overall_f1
            
            if combined_score > best_combined_score:
                console.print(
                    f"[green]綜合分數 (mAP * {map_weight:.2f} + F1 * {1-map_weight:.2f}) 有所改善 "
                    f"({best_combined_score:.2f} -> {combined_score:.2f})。"
                    f"[/green]"
                )
                console.print(
                    f"[green]詳細指標: mAP = {val_overall_map:.2f}% (之前最佳: {best_val_map_for_stopping:.2f}%), "
                    f"F1 = {val_overall_f1:.2f}% (之前最佳: {best_val_f1_for_stopping:.2f}%)[/green]"
                )
                
                best_combined_score = combined_score
                best_val_map_for_stopping = val_overall_map
                best_val_f1_for_stopping = val_overall_f1
                best_epoch = epoch
                best_state = model.state_dict()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                console.print(
                    f"綜合分數未改善，目前: {combined_score:.2f}，最佳: {best_combined_score:.2f}，"
                    f"已經 {early_stopping_counter} 個 epoch 無進展。"
                )
                console.print(
                    f"詳細指標: mAP = {val_overall_map:.2f}% (最佳: {best_val_map_for_stopping:.2f}%), "
                    f"F1 = {val_overall_f1:.2f}% (最佳: {best_val_f1_for_stopping:.2f}%)"
                )
                
                if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
                    console.print(
                        f"[bold red]Early stopping 已觸發，連續 {early_stopping_patience} 個 epoch "
                        f"綜合分數未見改善。[/bold red]"
                    )
                    break
        else:
            # 原始的只考慮 mAP 的邏輯
            if val_overall_map > best_val_map_for_stopping:
                console.print(
                    f"[green]Validation mAP improved "
                    f"({best_val_map_for_stopping:.2f}% -> {val_overall_map:.2f}%). "
                    f"Saving model.[/green]"
                )
                best_val_map_for_stopping = val_overall_map
                best_epoch = epoch
                best_state = model.state_dict()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                console.print(f"Validation mAP did not improve for {early_stopping_counter} epoch(s).")
                if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
                    console.print(
                        f"[bold red]Early stopping triggered after {early_stopping_patience} epochs "
                        f"without improvement on validation mAP.[/bold red]"
                    )
                    break

    # === 訓練結束 ===
    console.print(f"\nTraining finished after {final_epoch_ran} epochs.")
    if best_state is not None:
        if save_best_combined:
            console.print(
                f"[green]Loading best model from Epoch {best_epoch} "
                f"(Combined Score: {best_combined_score:.2f}, "
                f"Val mAP: {best_val_map_for_stopping:.2f}%, "
                f"Val F1: {best_val_f1_for_stopping:.2f}%) [/green]"
            )
        else:
            console.print(
                f"[green]Loading best model from Epoch {best_epoch} "
                f"(Val mAP: {best_val_map_for_stopping:.2f}%) [/green]"
            )
        model.load_state_dict(best_state)
    else:
        console.print("[yellow]Warning: No best model state saved, returning last epoch model.[/yellow]")

    return model


def evaluate_multi_cprfl(
    model,
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
    簡易評估函數，輸出各屬性的 mAP、F1、Accuracy 等等。
    """
    console.print("\n[bold cyan]=== 開始評估 ===[/bold cyan]")

    # 嘗試從 data_utils 匯入 compute_head_medium_tail
    # 若失敗則無法進行分佈 (Head/Medium/Tail) 分析
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

    text_embeddings_dict = initialize_multi_category_embeddings(attributes, categories_dict, clip_model, device, index_data)

    dataset = TensorDataset(
        test_data['image_embeddings'],
        *[test_labels_dict[attr] for attr in attributes]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    all_probs = {attr: [] for attr in attributes}
    all_labels = {attr: [] for attr in attributes}

    with torch.no_grad():
        for batch in dataloader:
            img_batch = batch[0].to(device).float()
            if hasattr(model, 'use_contrastive') and model.use_contrastive:
                logits_dict, _, _ = model(img_batch, text_embeddings_dict)
            else:
                logits_dict, _ = model(img_batch, text_embeddings_dict)

            for i, attr in enumerate(attributes):
                probs = torch.sigmoid(logits_dict[attr]).cpu().numpy()
                labels = batch[i+1].cpu().numpy()
                all_probs[attr].append(probs)
                all_labels[attr].append(labels)

    results = {}
    for attr in attributes:
        p = np.concatenate(all_probs[attr], axis=0)
        l = np.concatenate(all_labels[attr], axis=0)
        num_classes = p.shape[1]

        ap_list = []
        for c in range(num_classes):
            if np.sum(l[:, c]) == 0:
                continue
            ap_list.append(average_precision_score(l[:, c], p[:, c]))
        mAP_all = np.mean(ap_list)*100 if len(ap_list) > 0 else 0

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
            # 無法分析 HMT，或該屬性不在 hmt_info 中
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
        table = Table(title=f"{attr} 分類評估")
        table.add_column("Overall mAP")
        table.add_column("Overall F1")
        table.add_column("Head mAP")
        table.add_column("Medium mAP")
        table.add_column("Tail mAP")
        table.add_column("Overall Acc")

        head_mAP_val = type_metrics["head"]["mAP"]
        medium_mAP_val = type_metrics["medium"]["mAP"]
        tail_mAP_val = type_metrics["tail"]["mAP"]

        table.add_row(
            f"{mAP_all:.2f}%",
            f"{overall_f1:.2f}%",
            f"{head_mAP_val:.2f}%" if head_mAP_val else "N/A",
            f"{medium_mAP_val:.2f}%" if medium_mAP_val else "N/A",
            f"{tail_mAP_val:.2f}%" if tail_mAP_val else "N/A",
            f"{acc:.2f}%"
        )
        console.print(table)

    # 額外可做對比學習分析
    console.print("\n[bold cyan]=== 對比學習效果分析 (若適用) ===[/bold cyan]")
    for attr in attributes:
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
