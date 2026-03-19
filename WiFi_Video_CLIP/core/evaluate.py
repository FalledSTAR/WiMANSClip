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
def evaluate_retrieval(model, dataloader, device):
    """跨模态检索评估 (细粒度序列对齐版)"""
    model.eval()
    all_video_features, all_wifi_features = [], []
    
    for batch in dataloader:
        video_inputs = batch["video"].to(device)
        wifi_inputs = batch["wifi"].to(device)
        
        video_features, wifi_features, _ = model(video_inputs, wifi_inputs)
        all_video_features.append(video_features)
        all_wifi_features.append(wifi_features)
        
    V = torch.cat(all_video_features, dim=0)
    W = torch.cat(all_wifi_features, dim=0)
    N_samples = V.shape[0]

    sim = torch.einsum('imd,jnd->ijmn', V, W)
    
    sim_v2w, _ = sim.max(dim=3)     
    sim_v2w = sim_v2w.mean(dim=2)   
    
    sim_w2v, _ = sim.max(dim=2)     
    sim_w2v = sim_w2v.mean(dim=2)   
    
    similarity = (sim_v2w + sim_w2v) / 2.0 

    def calculate_recall(sim_matrix, k=1):
        _, topk_indices = sim_matrix.topk(k, dim=1)
        ground_truth = torch.arange(N_samples, device=device).view(-1, 1)
        correct = (topk_indices == ground_truth).sum().item()
        return correct / N_samples

    return {
        "V2W_R1": calculate_recall(similarity, k=1),
        "V2W_R5": calculate_recall(similarity, k=5),
        "W2V_R1": calculate_recall(similarity.T, k=1),
        "W2V_R5": calculate_recall(similarity.T, k=5)
    }

def evaluate_linear_probe(model, train_loader, test_loader, device, num_classes=54, epochs=80):
    """
    WiFi 单模态端到端微调 
    (引入预提取特征缓存技术与监督级匈牙利匹配)
    """
    print(f"\n[{_now_str()}] [阶段 1/3] 准备极速线性探测 (Fast Linear Probing) 环境...")
    
    for param in model.wifi_encoder.parameters():
        param.requires_grad = False
        
    class SlotAttentionClassifier(nn.Module):
        def __init__(self, in_features, num_users=6, actions_per_user=9):
            super().__init__()
            self.num_users = num_users
            
            # 6 个吸盘 (身份对称)
            self.query_embed = nn.Parameter(torch.randn(1, num_users, in_features))
            self.cross_attn = nn.MultiheadAttention(embed_dim=in_features, num_heads=8, batch_first=True, dropout=0.1)
            
            self.norm1 = nn.LayerNorm(in_features)
            self.norm2 = nn.LayerNorm(in_features)
            self.ffn = nn.Sequential(
                nn.Linear(in_features, in_features * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features * 2, in_features)
            )
            
            # ==========================================================
            # 【神级修复】：废除 6 个独立的脑袋，使用唯一共享的分类头！
            # 保证所有槽位的分类标准绝对统一，彻底解决过拟合与幻觉！
            # ==========================================================
            self.shared_head = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5), # 调高 Dropout，进一步榨干 1500 个样本的泛化潜力
                nn.Linear(256, actions_per_user)
            )
            
        def forward(self, x):
            B = x.size(0)
            queries = self.query_embed.expand(B, -1, -1)
            
            # 交叉注意力提取 6 个实例
            attn_out, _ = self.cross_attn(query=queries, key=x, value=x)
            out = self.norm1(queries + attn_out)
            out = self.norm2(out + self.ffn(out)) 
            
            # [Batch, 6, 512] -> 平铺 -> [Batch * 6, 512]
            out_flat = out.view(-1, out.size(-1))
            
            # 用唯一的大脑进行分类 -> [Batch * 6, 9]
            logits_flat = self.shared_head(out_flat)
            
            # 恢复形状 -> [Batch, 6, 9]
            return logits_flat.view(B, self.num_users, -1)

    def match_and_reorder_logits(logits_tensor, labels_tensor):
        """
        利用真实标签作为锚点，动态寻找无序槽位的最优排列组合
        使用 L1 距离 (绝对值误差) 作为代价矩阵，完美解决空槽位匹配问题
        保留梯度图，确保 BCE Loss 可以正常回传
        """
        B, N, C = logits_tensor.shape
        
        # 1. 在无梯度环境下，仅仅计算并找出最优对齐索引
        with torch.no_grad():
            probs = torch.sigmoid(logits_tensor)
            batch_indices = []
            
            for i in range(B):
                # 计算 6x6 代价矩阵 (L1 距离)
                cost_matrix = torch.sum(torch.abs(probs[i].unsqueeze(1) - labels_tensor[i].unsqueeze(0)), dim=-1)
                cost_matrix_np = cost_matrix.cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
                
                # col_ind 是标签的顺序 (0~5)。我们需要根据 col_ind 对 row_ind (预测) 进行排序
                # 这样就能让预测结果严格对齐 0~5 的标签坑位
                sort_idx = np.argsort(col_ind)
                aligned_row_ind = row_ind[sort_idx]
                
                batch_indices.append(torch.tensor(aligned_row_ind, device=logits_tensor.device))
                
            # 堆叠成 [B, 6] 的索引矩阵
            batch_indices = torch.stack(batch_indices)
            
        # 2. 在有梯度的计算图内，使用 gather 算子根据索引重排预测结果
        # 这样 classifier 就能正常接收反向传播的梯度！
        indices_expanded = batch_indices.unsqueeze(-1).expand(-1, -1, C)
        matched_logits = torch.gather(logits_tensor, dim=1, index=indices_expanded)
            
        return matched_logits

    classifier = SlotAttentionClassifier(model.wifi_encoder.projection.out_features).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    pos_weight = torch.ones([num_classes]).to(device) * 5.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    export_dir = "../result/clip/predictions"
    os.makedirs(export_dir, exist_ok=True)

    top_k = 5
    topk_records = [] 

    def _update_topk_prediction_file(score, epoch_idx, file_content):
        """根据综合评分保留前10名结果，文件名仅保留轮次信息"""
        should_save = len(topk_records) < top_k or score > topk_records[-1][0]

        if not should_save:
            return

        # 文件名精简：仅使用轮次
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
    
    train_feat_list, train_label_list = [], []
    with torch.no_grad():
        for batch in train_loader:
            wifi_inputs = batch["wifi"].to(device)
            labels = batch["label"].view(-1, num_classes).to(device)
            raw_feat = model.wifi_encoder(wifi_inputs)
            train_feat_list.append(raw_feat.cpu())
            train_label_list.append(labels.cpu())
            
    train_dataset_cached = TensorDataset(torch.cat(train_feat_list, dim=0), torch.cat(train_label_list, dim=0))
    cached_train_loader = DataLoader(train_dataset_cached, batch_size=train_loader.batch_size, shuffle=True)

    test_feat_list, test_label_list = [], []
    all_sample_ids = []
    with torch.no_grad():
        for batch in test_loader:
            wifi_inputs = batch["wifi"].to(device)
            labels = batch["label"].to(device) 
            raw_feat = model.wifi_encoder(wifi_inputs)
            test_feat_list.append(raw_feat.cpu())
            test_label_list.append(labels.cpu())
            all_sample_ids.extend(batch["sample_id"])
            
    test_dataset_cached = TensorDataset(torch.cat(test_feat_list, dim=0), torch.cat(test_label_list, dim=0))
    cached_test_loader = DataLoader(test_dataset_cached, batch_size=test_loader.batch_size, shuffle=False)
    
    print(f"\n[{_now_str()}] [阶段 3/3] 启动全速微调 (总 Epochs: {epochs})...")
    log_interval = 20

    for epoch in range(epochs):
        classifier.train()
        total_train_loss = 0.0
        
        for i, (cached_feat, labels) in enumerate(cached_train_loader):
            cached_feat = cached_feat.to(device)
            labels = labels.view(-1, 6, 9).to(device)
            
            optimizer.zero_grad()
            logits = classifier(cached_feat)
            logits = logits.view(-1, 6, 9)
            
            matched_logits = match_and_reorder_logits(logits, labels)
            
            loss = criterion(matched_logits.view(-1, 54), labels.view(-1, 54))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (i + 1) % log_interval == 0 or (i + 1) == len(cached_train_loader):
                print(f"[{_now_str()}]   -> Epoch [{epoch+1:02d}/{epochs}] | Step [{i+1:02d}/{len(cached_train_loader)}] | 当前 Batch Loss: {loss.item():.4f}")
                
        avg_loss = total_train_loss / len(cached_train_loader)
        
        classifier.eval()
        all_preds_logits, all_labels = [], []
        
        with torch.no_grad():
            for cached_feat, labels in cached_test_loader:
                cached_feat = cached_feat.to(device)
                labels = labels.view(-1, 6, 9).to(device)
                
                logits = classifier(cached_feat)      
                logits = logits.view(-1, 6, 9)     
                
                matched_logits = match_and_reorder_logits(logits, labels)
                
                all_preds_logits.append(matched_logits.cpu())
                all_labels.append(labels.cpu())
                
        full_logits = torch.cat(all_preds_logits, dim=0) 
        full_labels = torch.cat(all_labels, dim=0)       
        
        probs = torch.sigmoid(full_logits) 
        max_probs, pred_indices = torch.max(probs, dim=2) 
        
        pred_argmax_onehot = torch.zeros_like(full_logits)
        pred_argmax_onehot.scatter_(2, pred_indices.unsqueeze(-1), 1)
        valid_mask = max_probs > 0.5 
        pred_onehot = pred_argmax_onehot * valid_mask.unsqueeze(-1).float()
        
        if epoch == 0:
            gt_path = os.path.join(export_dir, "ground_truth.txt")
            with open(gt_path, "w", encoding="utf-8") as f:
                f.write("真实标签 (Ground Truth)\n")
                f.write("-" * 50 + "\n")
                for idx, s_id in enumerate(all_sample_ids):
                    f.write(f"Sample: {s_id}\n")
                    for u in range(6):
                        f.write(f"  User_{u+1}: {full_labels[idx, u].int().tolist()}\n")
                    f.write("\n")

        pred_flat = pred_onehot.view(-1, 9)
        labels_flat = full_labels.view(-1, 9)
        
        correct_all = (pred_flat == labels_flat).all(dim=1)
        active_mask = labels_flat.sum(dim=1) > 0   
        empty_mask = ~active_mask                  
        
        acc_active = correct_all[active_mask].sum().item() / active_mask.sum().item() if active_mask.sum().item() > 0 else 0.0
        acc_empty = correct_all[empty_mask].sum().item() / empty_mask.sum().item() if empty_mask.sum().item() > 0 else 0.0
        acc_total = correct_all.sum().item() / labels_flat.size(0)

        room_labels = full_labels.sum(dim=1) > 0 
        room_preds = pred_onehot.sum(dim=1) > 0 
        room_active_mask = room_labels.sum(dim=1) > 0 
        room_recall = ((room_preds == room_labels).all(dim=1)[room_active_mask].sum().item() / room_active_mask.sum().item()) if room_active_mask.sum().item() > 0 else 0.0

        # === 核心修改：三项指标的均值计算 ===
        comprehensive_score = (acc_total + acc_active + acc_empty) / 3.0

        pred_onehot_int = pred_onehot.int()
        full_labels_int = full_labels.int()

        prediction_lines = []
        prediction_lines.append(f"Epoch {epoch+1:02d} 预测结果 (Threshold-Gated Argmax)")
        prediction_lines.append(f"当前综合得分 (Score): {comprehensive_score:.4f}")
        prediction_lines.append(f"  - 全局总准确率: {acc_total:.4f}")
        prediction_lines.append(f"  - 真实动作准确率: {acc_active:.4f}")
        prediction_lines.append(f"  - 静默抑制准确率: {acc_empty:.4f}")
        prediction_lines.append("-" * 70)
        
        for idx, s_id in enumerate(all_sample_ids):
            sample_correct = bool((pred_onehot_int[idx] == full_labels_int[idx]).all().item())
            sample_status = "✔" if sample_correct else "❌"
            prediction_lines.append(f"Sample: {s_id} | SampleCorrect: {sample_status}")
            
            for u in range(6):
                user_correct = bool((pred_onehot_int[idx, u] == full_labels_int[idx, u]).all().item())
                user_status = "✔" if user_correct else "❌"
                gt_active = int(full_labels_int[idx, u].sum().item() > 0)
                pred_active = int(pred_onehot_int[idx, u].sum().item() > 0)
                prediction_lines.append(
                    f"  User_{u+1} | Correct: {user_status} | GT_Active: {gt_active} | Pred_Active: {pred_active}"
                )
                prediction_lines.append(f"    GT  : {full_labels_int[idx, u].tolist()}")
                prediction_lines.append(f"    Pred: {pred_onehot_int[idx, u].tolist()}")
            prediction_lines.append("")

        prediction_content = "\n".join(prediction_lines)
        
        _update_topk_prediction_file(comprehensive_score, epoch, prediction_content)

        total_test_samples = full_labels.size(0)
        total_slots = labels_flat.size(0)
        active_slots = int(active_mask.sum().item())
        empty_slots = int(empty_mask.sum().item())
        sample_has_action_mask = full_labels.view(total_test_samples, -1).sum(dim=1) > 0
        active_samples = int(sample_has_action_mask.sum().item())
        zero_user_samples = int((~sample_has_action_mask).sum().item())
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

    if final_eval_stats is not None:
        print(f"[{_now_str()}] [评估样本统计] 总测试样本数: {final_eval_stats['total_test_samples']}")
        print(f"[{_now_str()}] [评估样本统计] 槽位总数(测试样本数*6): {final_eval_stats['total_slots']}")
        print(f"[{_now_str()}] [评估样本统计] 真实有动作槽位数(active slots): {final_eval_stats['active_slots']}")
        print(f"[{_now_str()}] [评估样本统计] 全0槽位数(empty slots): {final_eval_stats['empty_slots']}")
        print(f"[{_now_str()}] [评估样本统计] 含动作样本数(至少1个用户有动作): {final_eval_stats['active_samples']}")
        print(f"[{_now_str()}] [评估样本统计] 0用户样本数(全用户全动作为0): {final_eval_stats['zero_user_samples']}")
        print(f"[{_now_str()}] [评估样本统计] 来自0用户样本的全0槽位数: {final_eval_stats['empty_slots_from_zero_user_samples']}")
        print(f"[{_now_str()}] [评估样本统计] 来自非0用户样本的全0槽位数: {final_eval_stats['empty_slots_from_partial_absent_users']}")

        print(f"[{_now_str()}] [Top-{top_k} 保存] 预测结果已保存轮次: {len(topk_records)}")
    
    return comprehensive_score