import torch
import torch.nn as nn
import torch.optim as optim
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

def evaluate_linear_probe(model, train_loader, test_loader, device, num_classes=54, epochs=10):
    """
    WiFi 单模态线性探测 (带有详细输出日志)
    """
    model.eval()
    # 定义线性分类器
    classifier = nn.Linear(model.wifi_encoder.projection.out_features, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    # 带有进度条的特征提取辅助函数
    @torch.no_grad()
    def extract_features(dataloader, desc_text):
        features, labels = [], []
        loop = tqdm(dataloader, desc=desc_text)
        for batch in loop:
            _, wifi_feat, _ = model(video_inputs=batch["video"].to(device), wifi_inputs=batch["wifi"].to(device))
            features.append(wifi_feat)
            labels.append(batch["label"].view(-1, num_classes).to(device))
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    print("\n[阶段 1/3] 正在通过冻结的 WiFi 编码器提取特征...")
    train_features, train_labels = extract_features(train_loader, "提取训练集特征")
    test_features, test_labels = extract_features(test_loader, "提取测试集/验证集特征")
    
    print(f"\n[阶段 2/3] 训练外挂线性分类器 (总 Epochs: {epochs})...")
    classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播与计算 Loss
        logits = classifier(train_features)
        loss = criterion(logits, train_labels)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        # 实时输出当前 Epoch 的分类 Loss
        print(f"  -> Linear Probe Epoch [{epoch+1}/{epochs}] | BCE Loss: {loss.item():.4f}")
        
    print("\n[阶段 3/3] 在测试集上评估最终准确率...")
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_features)
        predictions = (torch.sigmoid(test_logits) > 0.5).float()
        
        # 计算 54 维多标签全部匹配正确的严格准确率
        correct = (predictions == test_labels).all(dim=1).sum().item()
        accuracy = correct / test_labels.size(0)
        
    return accuracy