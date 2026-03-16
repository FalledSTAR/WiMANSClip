import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def evaluate_retrieval(model, dataloader, device):
    """跨模态检索评估 (V2W & W2V)"""
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
    N = V.shape[0]

    similarity = V @ W.T 

    def calculate_recall(sim_matrix, k=1):
        _, topk_indices = sim_matrix.topk(k, dim=1)
        ground_truth = torch.arange(N, device=device).view(-1, 1)
        correct = (topk_indices == ground_truth).sum().item()
        return correct / N

    return {
        "V2W_R1": calculate_recall(similarity, k=1),
        "V2W_R5": calculate_recall(similarity, k=5),
        "W2V_R1": calculate_recall(similarity.T, k=1),
        "W2V_R5": calculate_recall(similarity.T, k=5)
    }

def evaluate_linear_probe(model, train_loader, test_loader, device, num_classes=54, epochs=50):
    """
    WiFi 单模态端到端微调 (终极修正版)
    """
    print("\n[阶段 1/2] 准备全面微调 (Fine-tuning) 环境...")
    
    # 1. 解除 WiFi 编码器的参数冻结
    for param in model.wifi_encoder.parameters():
        param.requires_grad = True
        
    # 2. 【核心重构】：引入多头解码器 (Multi-Head Decoder)
    class MultiHeadClassifier(nn.Module):
        def __init__(self, in_features, num_users=6, actions_per_user=9):
            super().__init__()
            # 创建 6 个独立的物理分支，每个分支专注解码 1 个用户的 9 种动作
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
            # 将 6 个头的输出 (每个是 [Batch, 9]) 在最后一个维度拼接成 [Batch, 54]
            return torch.cat([head(x) for head in self.heads], dim=-1)

    classifier = MultiHeadClassifier(model.wifi_encoder.projection.out_features).to(device)
    
    # 3. 优化器配置
    optimizer = optim.Adam([
        {'params': model.wifi_encoder.parameters(), 'lr': 1e-4}, 
        {'params': classifier.parameters(), 'lr': 1e-3}
    ])
    
    # 【降低惩罚】：既然有了多头机制增强表达力，将过于极端的 20 倍惩罚降回 5 倍，防止模型乱开火
    pos_weight = torch.ones([num_classes]).to(device) * 5.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\n[阶段 2/2] 启动端到端微调 (总 Epochs: {epochs})...")
    
    for epoch in range(epochs):
        model.wifi_encoder.train()
        classifier.train()
        total_train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}] Training", leave=False)
        for batch in loop:
            wifi_inputs = batch["wifi"].to(device)
            labels = batch["label"].view(-1, num_classes).to(device)
            
            optimizer.zero_grad()
            
            # 【修复3】直接使用原始特征 raw_feat，彻底去除 F.normalize 束缚
            raw_feat = model.wifi_encoder(wifi_inputs)
            logits = classifier(raw_feat)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_loss = total_train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.wifi_encoder.eval()
        classifier.eval()
        
        all_preds, all_labels = [], []
        max_probs = [] # 【新增】记录模型给出的最大概率
        
        with torch.no_grad():
            for batch in test_loader:
                wifi_inputs = batch["wifi"].to(device)
                labels = batch["label"].view(-1, num_classes).to(device)
                
                raw_feat = model.wifi_encoder(wifi_inputs)
                logits = classifier(raw_feat)
                probs = torch.sigmoid(logits)
                
                max_probs.append(probs.max().item()) # 记录这一批次的最大预测概率
                preds = (probs > 0.5).float()
                
                all_preds.append(preds)
                all_labels.append(labels)
                
        full_preds = torch.cat(all_preds, dim=0).view(-1, 9)
        full_labels = torch.cat(all_labels, dim=0).view(-1, 9)
        
        correct_all = (full_preds == full_labels).all(dim=1)
        acc_all = correct_all.sum().item() / full_labels.size(0)
        
        active_mask = full_labels.sum(dim=1) > 0
        if active_mask.sum().item() > 0:
            acc_active = correct_all[active_mask].sum().item() / active_mask.sum().item()
        else:
            acc_active = 0.0
            
        current_max_prob = max(max_probs) if max_probs else 0.0
        print(f"  -> Finetune Epoch [{epoch+1:02d}/{epochs}] | Loss: {avg_loss:.4f} | 总准确率: {acc_all * 100:.2f}% | 真实动作准确率: {acc_active * 100:.2f}% | 模型输出最高概率: {current_max_prob:.2f}")
    
    return acc_all