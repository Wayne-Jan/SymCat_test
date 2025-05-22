# train_baseline_with_asl.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, average_precision_score, f1_score
from rich.console import Console
from rich.table import Table

# 匯入ASL損失函數
from src.loss_functions.loss import AsymmetricLossOptimized

console = Console()

def train_baseline_classifier(
    model,
    attributes,
    train_data,
    val_data,
    train_labels_dict,
    val_labels_dict,
    label_to_idx_dict,
    idx_to_label_dict,
    device,
    test_data=None,
    test_labels_dict=None,
    epochs=5,
    batch_size=32,
    lr=5e-5,
    weight_decay=1e-4,
    loss_type="asl",  # 新增參數，預設為asl
    gamma_neg=4,      # ASL參數
    gamma_pos=0,      # ASL參數
    clip_value=0.05,  # ASL參數
    early_stopping_patience=10,
    map_weight=0.5,
    save_best_combined=True
):
    """
    訓練基礎分類器的函數，支援ASL損失函數
    """
    console.print("\n[bold cyan]=== 開始訓練基礎分類器 (Baseline) ===[/bold cyan]")
    console.print(f"損失函數: {loss_type}")
    if loss_type.lower() == "asl":
        console.print(f"ASL參數: gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, clip={clip_value}")
    console.print(f"初始學習率: {lr}, 權重衰減: {weight_decay}")
    console.print(f"Early Stopping Patience: {early_stopping_patience} epochs")
    
    if save_best_combined:
        console.print(f"模型選擇策略: 綜合分數 (mAP 權重: {map_weight:.2f}, F1 權重: {1-map_weight:.2f})")
    else:
        console.print(f"模型選擇策略: 僅 mAP")
    console.print("")

    # 構建 DataLoader
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

    # 建立損失函數 (ASL或BCE)
    if loss_type.lower() == "asl":
        loss_fn = AsymmetricLossOptimized(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip_value
        )
    else:
        # 默認使用BCE
        loss_fn = nn.BCEWithLogitsLoss()

    # 建立優化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

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
        model.train()

        total_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            img_batch = batch[0].to(device).float()
            label_batches = batch[1:]
            
            # 前向傳播
            logits_dict, _ = model(img_batch)

            # 計算損失
            batch_loss = 0.0
            for i, attr in enumerate(attributes):
                logits = logits_dict[attr]
                gt = label_batches[i].to(device).float()
                batch_loss += loss_fn(logits, gt)

            # 反向傳播、優化
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 累加損失
            total_loss += batch_loss.item()

        # 這個 epoch 的平均損失
        avg_loss = total_loss / len(train_dataloader)
        current_lr = optimizer.param_groups[0]['lr']

        # === 驗證 ===
        model.eval()
        val_probs = {attr: [] for attr in attributes}
        val_labels_storage = {attr: [] for attr in attributes}

        with torch.no_grad():
            for batch in val_dataloader:
                val_img_batch = batch[0].to(device).float()
                logits_dict, _ = model(val_img_batch)

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
        scheduler.step()

        # === 測試 (若有) ===
        test_overall_map_str, test_overall_f1_str = "", ""
        if has_test_data:
            test_probs = {attr: [] for attr in attributes}
            test_labels_storage = {attr: [] for attr in attributes}
            with torch.no_grad():
                for batch in test_dataloader:
                    test_img_batch = batch[0].to(device).float()
                    logits_dict, _ = model(test_img_batch)

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
            f"Loss: {avg_loss:.4f} | "
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

# 評估函數可以保持不變

def evaluate_baseline_classifier(
    model,
    attributes,
    test_data,
    test_labels_dict,
    label_to_idx_dict,
    idx_to_label_dict,
    device,
    batch_size=32
):
    """
    評估基礎分類器的函數，類似於原始的evaluate_multi_cprfl
    """
    console.print("\n[bold cyan]=== 開始評估基礎分類器 (Baseline) ===[/bold cyan]")

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
            logits_dict, _ = model(img_batch)

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

    # 顯示頭中尾分析
    console.print("\n[bold cyan]=== 頭中尾 (H/M/T) 分析 ===[/bold cyan]")
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