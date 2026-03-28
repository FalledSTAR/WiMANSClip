import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import linear_sum_assignment

def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@torch.no_grad()
def evaluate_retrieval(model, val_loader, device):
    """
    使用全局特征 [Batch, 512] 进行跨模态检索评估
    """
    model.eval()
    
    all_video_features = []
    all_wifi_features = []

    with torch.no_grad():
        for batch in val_loader:
            video_inputs = batch["video"].to(device)
            wifi_inputs = batch["wifi"].to(device)

            # 获取特征，当前模型默认返回的是 video_features, wifi_features, logit_scale
            video_features, wifi_features, _ = model(video_inputs, wifi_inputs)

            # 【修复点】：如果在测试阶段 video 仍为 3D 序列特征，将其全局池化为 [Batch, 512]
            if video_features.dim() == 3:
                video_features = video_features.mean(dim=1)
                
            # 确保 WiFi 也是 [Batch, 512] (在 wifi_that.py 中我们在测试模式已默认做过了，这里加一层保险)
            if wifi_features.dim() == 3:
                wifi_features = wifi_features.mean(dim=1)

            all_video_features.append(video_features.cpu())
            all_wifi_features.append(wifi_features.cpu())

    # 拼接所有批次的特征
    V = torch.cat(all_video_features, dim=0)  # [N, 512]
    W = torch.cat(all_wifi_features, dim=0)   # [N, 512]

    # L2 归一化
    V = F.normalize(V, p=2, dim=-1)
    W = F.normalize(W, p=2, dim=-1)

    # 【核心修复】：直接使用二维矩阵乘法计算全局相似度矩阵
    sim_matrix = V @ W.t()  # 形状为 [N, N]

    metrics = {}
    N = sim_matrix.shape[0]

    # ========== Video 检索 WiFi (V2W) ==========
    v2w_ranks = torch.argsort(sim_matrix, dim=1, descending=True)
    # R@1
    v2w_r1 = (v2w_ranks[:, 0] == torch.arange(N)).float().mean().item()
    # R@5
    v2w_r5_correct = 0
    for i in range(N):
        if i in v2w_ranks[i, :5]:
            v2w_r5_correct += 1
    v2w_r5 = v2w_r5_correct / N

    # ========== WiFi 检索 Video (W2V) ==========
    w2v_sim = sim_matrix.t()
    w2v_ranks = torch.argsort(w2v_sim, dim=1, descending=True)
    # R@1
    w2v_r1 = (w2v_ranks[:, 0] == torch.arange(N)).float().mean().item()
    # R@5
    w2v_r5_correct = 0
    for i in range(N):
        if i in w2v_ranks[i, :5]:
            w2v_r5_correct += 1
    w2v_r5 = w2v_r5_correct / N

    metrics['V2W_R1'] = v2w_r1
    metrics['V2W_R5'] = v2w_r5
    metrics['W2V_R1'] = w2v_r1
    metrics['W2V_R5'] = w2v_r5

    return metrics

def save_slot_predictions(sample_ids, logits, labels, epoch, save_dir):
    """
    将基于位置 (a-f) 的 10 分类预测结果与真值对比保存到 txt 文件中
    """
    idx_to_class = {
        0: 'nan', 1: 'nothing', 2: 'walk', 3: 'rotation',
        4: 'jump', 5: 'wave', 6: 'lie_down', 7: 'pick_up',
        8: 'sit_down', 9: 'stand_up'
    }
    loc_names = ['a', 'b', 'c', 'd', 'e', 'f']

    preds = torch.argmax(logits, dim=-1)
    gts = torch.argmax(labels, dim=-1)

    pred_dir = os.path.join(save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    log_file = os.path.join(pred_dir, f"epoch_{epoch}_val_predictions.txt")
    
    with open(log_file, "w", encoding="utf-8") as f:
        for b in range(len(sample_ids)):
            sample_id = sample_ids[b]
            f.write(f"Sample: {sample_id}\n")
            
            for i in range(6):
                gt_idx = gts[b, i].item()
                pred_idx = preds[b, i].item()
                
                gt_cls = idx_to_class.get(gt_idx, "UNKNOWN")
                pred_cls = idx_to_class.get(pred_idx, "UNKNOWN")
                
                mark = "✔" if gt_cls == pred_cls else "❌"
                f.write(f"  Loc {loc_names[i]} | GT: {gt_cls:<8} | Pred: {pred_cls:<8} {mark}\n")
            f.write("-" * 40 + "\n")

def evaluate_classification(model, val_loader, device, epoch, save_dir):
    """
    遍历验证集，计算 6 槽位分类的准确率，并调用保存日志函数
    """
    model.eval()
    correct = 0
    total = 0
    
    all_sample_ids = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            sample_ids = batch["sample_id"]
            video_inputs = batch["video"].to(device)
            wifi_inputs = batch["wifi"].to(device)
            labels = batch["label"].to(device) # [B, 6, 10]
            
            # 前向传播，获取 6 槽位的 logits
            _, _, _, wifi_logits = model(video_inputs, wifi_inputs, return_logits=True)
            
            # 计算准确率 (6 个位置均参与计算)
            preds = torch.argmax(wifi_logits, dim=-1)
            gts = torch.argmax(labels, dim=-1)
            correct += (preds == gts).sum().item()
            total += gts.numel()

            all_sample_ids.extend(sample_ids)
            all_logits.append(wifi_logits.cpu())
            all_labels.append(labels.cpu())

    # 拼接所有批次的数据并保存日志
    cat_logits = torch.cat(all_logits, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)
    save_slot_predictions(all_sample_ids, cat_logits, cat_labels, epoch, save_dir)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def evaluate_linear_probe(model, train_loader, test_loader, device, num_classes=10, epochs=80):
    """
    全新升级：适配 [Batch, 6, 10] 位置锚定架构的线性探测评估
    彻底废除匈牙利匹配，完美保留所有详细终端辅助信息与日志保存
    """
    print(f"\n[{_now_str()}] [阶段 1/3] 准备极速线性探测 (Fast Linear Probing) 环境...")
    
    # 冻结骨干网络
    for param in model.wifi_encoder.parameters():
        param.requires_grad = False
        
    # 【架构简化】：由于 wifi_encoder 已经输出了完美解耦的 6 个槽位特征
    # 我们不需要复杂的 SharedHead 或 Attention，只需要一个简单的线性探测器
    classifier = nn.Linear(512, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    # 使用交叉熵损失 (CrossEntropyLoss 自动处理 10 分类)
    criterion = nn.CrossEntropyLoss()

    export_dir = "../result/clip/predictions"
    os.makedirs(export_dir, exist_ok=True)

    top_k = 5
    topk_records = [] 

    def _update_topk_prediction_file(score, epoch_idx, file_content):
        """根据综合评分保留前10名结果，文件名仅保留轮次信息"""
        should_save = len(topk_records) < top_k or score > topk_records[-1][0]
        if not should_save:
            return
        file_name = f"epoch_{epoch_idx+1:02d}.txt"
        file_path = os.path.join(export_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        topk_records.append((score, file_path))
        topk_records.sort(key=lambda x: x[0], reverse=True)
        if len(topk_records) > top_k:
            _, path_to_remove = topk_records.pop()
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

    final_eval_stats = None

    print(f"\n[{_now_str()}] [阶段 2/3] 预提取并缓存骨干网络特征 (只需执行 1 次)...")
    model.wifi_encoder.eval()
    
    def extract_features(loader):
        feat_list, label_list, ids_list = [], [], []
        with torch.no_grad():
            for batch in loader:
                wifi_inputs = batch["wifi"].to(device)
                labels = batch["label"].to(device) # [B, 6, 10]
                ids_list.extend(batch.get("sample_id", []))
                
                # 提取槽位特征 [B, 6, 512]
                outputs = model.wifi_encoder(wifi_inputs, return_logits=True)
                raw_feat = outputs[0] if isinstance(outputs, tuple) else outputs
                
                feat_list.append(raw_feat.cpu())
                label_list.append(labels.cpu())
        return torch.cat(feat_list, dim=0), torch.cat(label_list, dim=0), ids_list

    train_feats, train_labels, _ = extract_features(train_loader)
    test_feats, test_labels, all_sample_ids = extract_features(test_loader)
            
    cached_train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=train_loader.batch_size, shuffle=True)
    cached_test_loader = DataLoader(TensorDataset(test_feats, test_labels), batch_size=test_loader.batch_size, shuffle=False)
    
    print(f"\n[{_now_str()}] [阶段 3/3] 启动全速微调 (总 Epochs: {epochs})...")
    log_interval = 20

    for epoch in range(epochs):
        classifier.train()
        total_train_loss = 0.0
        
        for i, (cached_feat, labels) in enumerate(cached_train_loader):
            cached_feat = cached_feat.to(device) # [B, 6, 512]
            labels = labels.to(device)           # [B, 6, 10]
            
            # 【核心修改】：平铺特征去训练唯一的线性层
            feats_flat = cached_feat.view(-1, 512)
            # 获取真实的 10 分类标签索引 (0~9)
            lbls_idx = torch.argmax(labels, dim=-1).view(-1)
            
            optimizer.zero_grad()
            logits_flat = classifier(feats_flat) # [B*6, 10]
            
            # 直接计算 CE Loss，不再需要匈牙利重排
            loss = criterion(logits_flat, lbls_idx)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (i + 1) % log_interval == 0 or (i + 1) == len(cached_train_loader):
                print(f"[{_now_str()}]   -> Epoch [{epoch+1:02d}/{epochs}] | Step [{i+1:02d}/{len(cached_train_loader)}] | 当前 Batch Loss: {loss.item():.4f}")
                
        avg_loss = total_train_loss / len(cached_train_loader)
        
        # =========================================================
        # 核心评估：保留你所有的细粒度统计指标
        # =========================================================
        classifier.eval()
        all_preds, all_gts = [], []
        
        with torch.no_grad():
            for cached_feat, labels in cached_test_loader:
                cached_feat = cached_feat.to(device)
                logits = classifier(cached_feat) # [B, 6, 10]
                
                preds_idx = torch.argmax(logits, dim=-1) # [B, 6]
                gts_idx = torch.argmax(labels.to(device), dim=-1) # [B, 6]
                
                all_preds.append(preds_idx.cpu())
                all_gts.append(gts_idx.cpu())
                
        full_preds = torch.cat(all_preds, dim=0) # [N, 6]
        full_gts = torch.cat(all_gts, dim=0)     # [N, 6]

        # 还原回 10 维 one-hot 格式，以兼容你的 txt 文件写入逻辑
        pred_onehot_int = F.one_hot(full_preds, num_classes=10).int()
        full_labels_int = F.one_hot(full_gts, num_classes=10).int()
        
        if epoch == 0:
            gt_path = os.path.join(export_dir, "ground_truth.txt")
            with open(gt_path, "w", encoding="utf-8") as f:
                f.write("真实标签 (Ground Truth)\n")
                f.write("-" * 50 + "\n")
                for idx, s_id in enumerate(all_sample_ids):
                    f.write(f"Sample: {s_id}\n")
                    for u in range(6):
                        f.write(f"  User_{u+1}: {full_labels_int[idx, u].tolist()}\n")
                    f.write("\n")

        # 【指标掩码计算】: 索引 0是nan, 1是nothing。 大于 1 (2~9) 才是真实的 action
        correct_matrix = (full_preds == full_gts) # [N, 6]
        active_mask = full_gts > 1                # 真实存在动作的槽位
        empty_mask = full_gts <= 1                # 静默或无人槽位 (nan/nothing)
        
        acc_active = correct_matrix[active_mask].float().mean().item() if active_mask.sum().item() > 0 else 0.0
        acc_empty = correct_matrix[empty_mask].float().mean().item() if empty_mask.sum().item() > 0 else 0.0
        acc_total = correct_matrix.float().mean().item()

        # 房间级命中率 (6个槽位全部预测正确，且该房间必须有真实动作)
        room_correct = correct_matrix.all(dim=1)
        room_active_mask = active_mask.any(dim=1)
        room_recall = room_correct[room_active_mask].float().mean().item() if room_active_mask.sum().item() > 0 else 0.0

        comprehensive_score = (acc_total + acc_active + acc_empty) / 3.0

        # ========== 构建你熟悉的文本日志 ==========
        prediction_lines = []
        prediction_lines.append(f"Epoch {epoch+1:02d} 预测结果 (10分类位置锚定)")
        prediction_lines.append(f"当前综合得分 (Score): {comprehensive_score:.4f}")
        prediction_lines.append(f"  - 全局总准确率: {acc_total:.4f}")
        prediction_lines.append(f"  - 真实动作准确率: {acc_active:.4f}")
        prediction_lines.append(f"  - 静默抑制准确率: {acc_empty:.4f}")
        prediction_lines.append("-" * 70)
        
        for idx, s_id in enumerate(all_sample_ids):
            sample_correct = bool(correct_matrix[idx].all().item())
            sample_status = "✔" if sample_correct else "❌"
            prediction_lines.append(f"Sample: {s_id} | SampleCorrect: {sample_status}")
            
            for u in range(6):
                user_correct = bool(correct_matrix[idx, u].item())
                user_status = "✔" if user_correct else "❌"
                gt_active = 1 if full_gts[idx, u].item() > 1 else 0
                pred_active = 1 if full_preds[idx, u].item() > 1 else 0
                
                prediction_lines.append(
                    f"  Loc_{u+1} | Correct: {user_status} | GT_Active: {gt_active} | Pred_Active: {pred_active}"
                )
                prediction_lines.append(f"    GT  : {full_labels_int[idx, u].tolist()}")
                prediction_lines.append(f"    Pred: {pred_onehot_int[idx, u].tolist()}")
            prediction_lines.append("")

        prediction_content = "\n".join(prediction_lines)
        _update_topk_prediction_file(comprehensive_score, epoch, prediction_content)

        # ========== 恢复详细统计数据台账 ==========
        total_test_samples = full_gts.size(0)
        total_slots = total_test_samples * 6
        active_slots = int(active_mask.sum().item())
        empty_slots = int(empty_mask.sum().item())
        
        active_samples = int(room_active_mask.sum().item())
        zero_user_samples = int((full_gts == 0).all(dim=1).sum().item())
        empty_slots_from_zero_user_samples = zero_user_samples * 6
        empty_slots_from_partial_absent_users = max(empty_slots - empty_slots_from_zero_user_samples, 0)
        
        final_eval_stats = {
            "total_test_samples": total_test_samples,
            "total_slots": total_slots,
            "active_slots": active_slots,
            "empty_slots": empty_slots,
            "active_samples": active_samples,
            "zero_user_samples": zero_user_samples,
            "empty_slots_from_zero_user_samples": empty_slots_from_zero_user_samples,
            "empty_slots_from_partial_absent_users": empty_slots_from_partial_absent_users
        }

        print(f"[{_now_str()}] ==> Epoch [{epoch+1:02d}/{epochs}] 总结 | 平均 Loss: {avg_loss:.4f} | "
              f"【综合得分】: {comprehensive_score * 100:.2f} | "
              f"全局总准确率: {acc_total * 100:.2f}% | "
              f"真实动作准确率: {acc_active * 100:.2f}% | "
              f"静默抑制准确率: {acc_empty * 100:.2f}% | "
              f"房间级命中率: {room_recall * 100:.2f}%\n")

    # 训练结束后打印最终台账
    if final_eval_stats is not None:
        print(f"[{_now_str()}] [评估样本统计] 总测试样本数: {final_eval_stats['total_test_samples']}")
        print(f"[{_now_str()}] [评估样本统计] 槽位总数(测试样本数*6): {final_eval_stats['total_slots']}")
        print(f"[{_now_str()}] [评估样本统计] 真实有动作槽位数(>1): {final_eval_stats['active_slots']}")
        print(f"[{_now_str()}] [评估样本统计] 空置/静默槽位数(<=1): {final_eval_stats['empty_slots']}")
        print(f"[{_now_str()}] [评估样本统计] 含动作样本数(至少1个用户有动作): {final_eval_stats['active_samples']}")
        print(f"[{_now_str()}] [评估样本统计] 0用户空房间样本数: {final_eval_stats['zero_user_samples']}")
        print(f"[{_now_str()}] [评估样本统计] 来自空房间的纯静默槽位数: {final_eval_stats['empty_slots_from_zero_user_samples']}")
        print(f"[{_now_str()}] [评估样本统计] 来自非空房间的无动作槽位数: {final_eval_stats['empty_slots_from_partial_absent_users']}")
        print(f"[{_now_str()}] [Top-{top_k} 保存] 预测结果已保存轮次: {len(topk_records)}")
    
    return comprehensive_score