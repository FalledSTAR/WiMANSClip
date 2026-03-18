import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

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
        
    V = torch.cat(all_video_features, dim=0) # 形状: [N, Time_v, 512]
    W = torch.cat(all_wifi_features, dim=0)  # 形状: [N, Time_w, 512]
    N_samples = V.shape[0]

    # 1. 计算细粒度交叉相似度矩阵
    # 结果矩阵 sim 形状: [N, N, Time_v, Time_w]
    # 注意: N=377 时，这步会瞬时占用约 1.5GB 显存，一般显卡可轻松承受
    sim = torch.einsum('imd,jnd->ijmn', V, W)
    
    # 2. V2W 局部对齐 (视频找WiFi)
    sim_v2w, _ = sim.max(dim=3)     # 取 WiFi 时间步上的最大值: [N, N, Time_v]
    sim_v2w = sim_v2w.mean(dim=2)   # 对视频时间步取平均: [N, N]
    
    # 3. W2V 局部对齐 (WiFi找视频)
    sim_w2v, _ = sim.max(dim=2)     # 取视频时间步上的最大值: [N, N, Time_w]
    sim_w2v = sim_w2v.mean(dim=2)   # 对 WiFi 时间步取平均: [N, N]
    
    # 4. 融合得到全局相似度矩阵
    similarity = (sim_v2w + sim_w2v) / 2.0 # 形状: [N, N]

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
    (修复：真实样本ID溯源 + 树状文件导出 + Argmax 最高概率评估)
    """
    print("\n[阶段 1/2] 准备极速线性探测 (Fast Linear Probing) 环境...")
    
    # 彻底冻结 WiFi 编码器，将其变为纯粹的特征提取器
    for param in model.wifi_encoder.parameters():
        param.requires_grad = False
        
    # 2. 【核心重构】：引入 DETR 风格的 Slot Attention (槽位交叉注意力机制)
    class SlotAttentionClassifier(nn.Module):
        def __init__(self, in_features, num_users=6, actions_per_user=9):
            super().__init__()
            self.num_users = num_users
            
            # 【神级组件 1】：6 个可学习的查询向量 (Learnable Queries)，代表 6 个物理坑位
            self.query_embed = nn.Parameter(torch.randn(1, num_users, in_features))
            
            # 【神级组件 2】：交叉注意力机制 (Cross-Attention)
            # 让 6 个 Query 去 150 帧的特征序列中，主动提取属于自己的动作信息
            self.cross_attn = nn.MultiheadAttention(embed_dim=in_features, num_heads=8, batch_first=True, dropout=0.1)
            
            # 归一化与前馈网络 (标准的 Transformer Decoder 结构)
            self.norm1 = nn.LayerNorm(in_features)
            self.norm2 = nn.LayerNorm(in_features)
            self.ffn = nn.Sequential(
                nn.Linear(in_features, in_features * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features * 2, in_features)
            )
            
            # 6 个物理分支，独立解码用户槽位
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
            # x 的输入形状: [Batch, 150, 512] (预训练提取的 WiFi 序列特征)
            B = x.size(0)
            
            # 1. 将 6 个 Query 扩展到当前 Batch -> [Batch, 6, 512]
            queries = self.query_embed.expand(B, -1, -1)
            
            # 2. Cross-Attention 分离特征
            # Query=吸盘(6个), Key/Value=特征序列(150帧)
            # 输出 attn_out 形状: [Batch, 6, 512]
            attn_out, _ = self.cross_attn(query=queries, key=x, value=x)
            
            # 3. 残差连接与 FFN
            out = self.norm1(queries + attn_out)
            out = self.norm2(out + self.ffn(out)) # [Batch, 6, 512]
            
            # 4. 解耦后的特征送入独立的分类头
            logits = []
            for i in range(self.num_users):
                slot_feat = out[:, i, :] # 精确提取第 i 个人的独立特征: [Batch, 512]
                logits.append(self.heads[i](slot_feat)) # [Batch, 9]
                
            return torch.cat(logits, dim=-1) # [Batch, 54]

    classifier = SlotAttentionClassifier(model.wifi_encoder.projection.out_features).to(device)
    
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    pos_weight = torch.ones([num_classes]).to(device) * 5.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    export_dir = "../result/clip/predictions"
    os.makedirs(export_dir, exist_ok=True)

    print(f"\n[阶段 2/2] 启动端到端微调 (总 Epochs: {epochs})...")
    log_interval = 20

    for epoch in range(epochs):
        model.wifi_encoder.train()
        classifier.train()
        total_train_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            wifi_inputs = batch["wifi"].to(device)
            # 训练时依然拍平为 54 维计算整体 BCE 损失
            labels = batch["label"].view(-1, num_classes).to(device)
            
            optimizer.zero_grad()
            raw_feat = model.wifi_encoder(wifi_inputs)
            logits = classifier(raw_feat)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (i + 1) % log_interval == 0 or (i + 1) == len(train_loader):
                print(f"  -> Epoch [{epoch+1:02d}/{epochs}] | Step [{i+1:02d}/{len(train_loader)}] | 当前 Batch Loss: {loss.item():.4f}")
                
        avg_loss = total_train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.wifi_encoder.eval()
        classifier.eval()
        all_preds_logits, all_labels = [], []
        all_sample_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                sample_ids = batch["sample_id"]
                wifi_inputs = batch["wifi"].to(device)
                labels = batch["label"].to(device) # 形状: [Batch, 6, 9]
                
                raw_feat = model.wifi_encoder(wifi_inputs)
                logits = classifier(raw_feat)      # 形状: [Batch, 54]
                logits = logits.view(-1, 6, 9)     # 还原为: [Batch, 6, 9]
                
                all_sample_ids.extend(sample_ids)
                all_preds_logits.append(logits)
                all_labels.append(labels)
                
        full_logits = torch.cat(all_preds_logits, dim=0) # [N, 6, 9]
        full_labels = torch.cat(all_labels, dim=0)       # [N, 6, 9]
        
        # ==========================================
        # 【终极修正】：阈值门控的 Argmax (Threshold-Gated Argmax)
        # ==========================================
        probs = torch.sigmoid(full_logits) # 将 Logits 转为 0~1 的概率
        
        # 1. 找到每个用户槽位的最高概率及其对应的动作索引
        max_probs, pred_indices = torch.max(probs, dim=2) # [N, 6]
        
        # 2. 生成基础的 Argmax One-Hot 矩阵
        pred_argmax_onehot = torch.zeros_like(full_logits)
        pred_argmax_onehot.scatter_(2, pred_indices.unsqueeze(-1), 1)
        
        # 3. 生成有效性掩码：只有最高概率突破 0.5 的槽位，才被认为“有人做动作”
        valid_mask = max_probs > 0.5 # [N, 6]
        
        # 4. 将掩码应用到 Argmax 矩阵上 (没突破阈值的槽位将被强制归零)
        pred_onehot = pred_argmax_onehot * valid_mask.unsqueeze(-1).float()
        
        # ==========================================
        # 导出真值与预测结果 (层级结构 + 真实 ID)
        # ==========================================
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

        # ==========================================
        # 指标计算：仅计算真实发生的动作的 Top-1 准确率
        # ==========================================
        # 将张量展平计算
        pred_flat = pred_onehot.view(-1, 9)
        labels_flat = full_labels.view(-1, 9)
        
        correct_all = (pred_flat == labels_flat).all(dim=1)
        active_mask = labels_flat.sum(dim=1) > 0 
        
        if active_mask.sum().item() > 0:
            acc_active = correct_all[active_mask].sum().item() / active_mask.sum().item()
        else:
            acc_active = 0.0
            
        print(f"==> Epoch [{epoch+1:02d}/{epochs}] 总结 | 平均 Loss: {avg_loss:.4f} | 真实动作准确率: {acc_active * 100:.2f}%\n")
    
    return acc_active