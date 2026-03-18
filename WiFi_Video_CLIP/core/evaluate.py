import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader


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
        
        # 此时输出的特征维度为 [Batch, Time, 512]
        video_features, wifi_features, _ = model(video_inputs, wifi_inputs)
        all_video_features.append(video_features)
        all_wifi_features.append(wifi_features)
        
    V = torch.cat(all_video_features, dim=0) # [N, Time_v, 512]
    W = torch.cat(all_wifi_features, dim=0)  # [N, Time_w, 512]
    N_samples = V.shape[0]

    # 1. 计算细粒度交叉相似度矩阵
    sim = torch.einsum('imd,jnd->ijmn', V, W)
    
    # 2. V2W 局部对齐 (视频找WiFi)
    sim_v2w, _ = sim.max(dim=3)     
    sim_v2w = sim_v2w.mean(dim=2)   
    
    # 3. W2V 局部对齐 (WiFi找视频)
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
    (引入预提取特征缓存技术，实现百倍极速加速)
    """
    print(f"\n[{_now_str()}] [阶段 1/3] 准备极速线性探测 (Fast Linear Probing) 环境...")
    
    # 彻底冻结 WiFi 编码器，将其变为纯粹的特征提取器
    for param in model.wifi_encoder.parameters():
        param.requires_grad = False
        
    class SlotAttentionClassifier(nn.Module):
        def __init__(self, in_features, num_users=6, actions_per_user=9):
            super().__init__()
            self.num_users = num_users
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
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, actions_per_user)
                ) for _ in range(num_users)
            ])
            
        def forward(self, x):
            B = x.size(0)
            queries = self.query_embed.expand(B, -1, -1)
            attn_out, _ = self.cross_attn(query=queries, key=x, value=x)
            out = self.norm1(queries + attn_out)
            out = self.norm2(out + self.ffn(out)) 
            logits = []
            for i in range(self.num_users):
                slot_feat = out[:, i, :] 
                logits.append(self.heads[i](slot_feat)) 
            return torch.cat(logits, dim=-1) 

    classifier = SlotAttentionClassifier(model.wifi_encoder.projection.out_features).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    pos_weight = torch.ones([num_classes]).to(device) * 5.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    export_dir = "../result/clip/predictions"
    os.makedirs(export_dir, exist_ok=True)

    # ==========================================================
    # 【核心加速引擎】：预提取特征缓存 (Feature Caching)
    # ==========================================================
    print(f"\n[{_now_str()}] [阶段 2/3] 预提取并缓存骨干网络特征 (只需执行 1 次)...")
    model.wifi_encoder.eval()
    
    train_feat_list, train_label_list = [], []
    with torch.no_grad():
        for batch in train_loader:
            wifi_inputs = batch["wifi"].to(device)
            labels = batch["label"].view(-1, num_classes).to(device)
            raw_feat = model.wifi_encoder(wifi_inputs)
            # 存入 CPU 内存，彻底释放显存并切断硬盘 IO
            train_feat_list.append(raw_feat.cpu())
            train_label_list.append(labels.cpu())
            
    # 重新构建极速 DataLoader
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

    # ==========================================================
    # 极速训练循环 (直接读取缓存特征，不再经过 Encoder)
    # ==========================================================
    for epoch in range(epochs):
        classifier.train()
        total_train_loss = 0.0
        
        for i, (cached_feat, labels) in enumerate(cached_train_loader):
            cached_feat = cached_feat.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = classifier(cached_feat)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (i + 1) % log_interval == 0 or (i + 1) == len(cached_train_loader):
                print(f"[{_now_str()}]   -> Epoch [{epoch+1:02d}/{epochs}] | Step [{i+1:02d}/{len(cached_train_loader)}] | 当前 Batch Loss: {loss.item():.4f}")
                
        avg_loss = total_train_loss / len(cached_train_loader)
        
        # --- 验证阶段 ---
        classifier.eval()
        all_preds_logits, all_labels = [], []
        
        with torch.no_grad():
            for cached_feat, labels in cached_test_loader:
                cached_feat = cached_feat.to(device)
                
                logits = classifier(cached_feat)      
                logits = logits.view(-1, 6, 9)     
                
                all_preds_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                
        full_logits = torch.cat(all_preds_logits, dim=0) 
        full_labels = torch.cat(all_labels, dim=0)       
        
        # 阈值门控的 Argmax (Threshold-Gated Argmax)
        probs = torch.sigmoid(full_logits) 
        max_probs, pred_indices = torch.max(probs, dim=2) 
        
        pred_argmax_onehot = torch.zeros_like(full_logits)
        pred_argmax_onehot.scatter_(2, pred_indices.unsqueeze(-1), 1)
        valid_mask = max_probs > 0.5 
        pred_onehot = pred_argmax_onehot * valid_mask.unsqueeze(-1).float()
        
        # 导出文件逻辑
        if epoch == 0:
            gt_path = os.path.join(export_dir, "ground_truth.txt")
            with open(gt_path, "w") as f:
                f.write("真实标签 (Ground Truth)\n")
                f.write("-" * 50 + "\n")
                for idx, s_id in enumerate(all_sample_ids):
                    f.write(f"Sample: {s_id}\n")
                    for u in range(6):
                        f.write(f"  User_{u+1}: {full_labels[idx, u].int().tolist()}\n")
                    f.write("\n")

        if epoch < 5:
            pred_path = os.path.join(export_dir, f"prediction_epoch_{epoch+1:02d}.txt")
            with open(pred_path, "w") as f:
                f.write(f"Epoch {epoch+1:02d} 预测结果 (Threshold-Gated Argmax)\n")
                f.write("-" * 50 + "\n")
                for idx, s_id in enumerate(all_sample_ids):
                    f.write(f"Sample: {s_id}\n")
                    for u in range(6):
                        f.write(f"  User_{u+1}: {pred_onehot[idx, u].int().tolist()}\n")
                    f.write("\n")

        # 指标计算：全维度透明评估
        pred_flat = pred_onehot.view(-1, 9)
        labels_flat = full_labels.view(-1, 9)
        
        correct_all = (pred_flat == labels_flat).all(dim=1)
        active_mask = labels_flat.sum(dim=1) > 0   
        empty_mask = ~active_mask                  
        
        if active_mask.sum().item() > 0:
            acc_active = correct_all[active_mask].sum().item() / active_mask.sum().item()
        else:
            acc_active = 0.0
            
        if empty_mask.sum().item() > 0:
            acc_empty = correct_all[empty_mask].sum().item() / empty_mask.sum().item()
        else:
            acc_empty = 0.0
            
        acc_total = correct_all.sum().item() / labels_flat.size(0)

        # ==========================================
        # 【新增】：计算房间级/无序动作召回率 (Room-Level Recall)
        # ==========================================
        # 1. 房间级别的真实动作集合 (只要房间里有人做某动作，对应动作位就为 True)
        # full_labels 形状: [N, 6, 9] -> room_labels 形状: [N, 9]
        room_labels = full_labels.sum(dim=1) > 0 
        
        # 2. 房间级别的预测动作集合
        # pred_onehot 形状: [N, 6, 9] -> room_preds 形状: [N, 9]
        room_preds = pred_onehot.sum(dim=1) > 0 
        
        # 3. 计算有动作发生的房间的召回率
        room_active_mask = room_labels.sum(dim=1) > 0 # 筛选出确实有动作的房间 [N]
        
        if room_active_mask.sum().item() > 0:
            # 在有动作的房间中，预测出的动作集合完全覆盖/等同于真实动作集合的比例
            correct_rooms = (room_preds == room_labels).all(dim=1)
            room_recall = correct_rooms[room_active_mask].sum().item() / room_active_mask.sum().item()
        else:
            room_recall = 0.0

        print(f"[{_now_str()}] ==> Epoch [{epoch+1:02d}/{epochs}] 总结 | 平均 Loss: {avg_loss:.4f} | "
              f"全局总准确率: {acc_total * 100:.2f}% | "
              f"真实动作准确率(严苛): {acc_active * 100:.2f}% | "
              f"静默抑制准确率: {acc_empty * 100:.2f}% | "
              f"【核心验证】房间级无序动作命中率: {room_recall * 100:.2f}%\n")
    
    return acc_active