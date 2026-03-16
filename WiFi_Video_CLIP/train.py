import os
import torch
from tqdm import tqdm
from core.evaluate import evaluate_retrieval

def train_loop(model, train_loader, val_loader, criterion, optimizer, cfg, device):
    epochs = cfg['train']['epochs']
    save_dir = cfg['train']['save_dir']
    top_k = cfg.get('train', {}).get('save_top_k', 3)
    
    # 【新增】梯度累加步数，默认设为 4。相当于有效 Batch Size = BS * steps，适合显存受限的情况。
    accumulation_steps = cfg.get('train', {}).get('accumulation_steps', 1) 
    
    os.makedirs(save_dir, exist_ok=True)
    saved_models = [] 

    # =====================================================================
    # 定义余弦退火学习率调度器
    # T_max 设置为总 epochs，eta_min 为学习率衰减的下限
    # =====================================================================
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs, 
        eta_min=5e-5
    )
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        # 设定每隔多少步打印一次信息 (比如每 20 步)
        log_interval = 20
        
        for i, batch in enumerate(train_loader):
            video_inputs = batch["video"].to(device)
            wifi_inputs = batch["wifi"].to(device)
            
            video_features, wifi_features, logit_scale = model(video_inputs, wifi_inputs)
            
            # 计算损失并除以累加步数
            loss = criterion(video_features, wifi_features, logit_scale)
            loss = loss / accumulation_steps
            loss.backward()
            
            # 梯度累加与参数更新
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            # 记录真实的 batch loss
            real_loss = loss.item() * accumulation_steps
            total_loss += real_loss
            
            # 实时动态打印日志
            if (i + 1) % log_interval == 0 or (i + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{i+1}/{len(train_loader)}] | Loss: {real_loss:.4f} | Temp: {logit_scale.item():.2f}")
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n---> Epoch {epoch+1} 结束 | 平均训练 Loss: {total_loss / len(train_loader):.4f} | 当前学习率: {current_lr:.6f}")    


        # --- 验证阶段 ---
        retrieval_metrics = evaluate_retrieval(model, val_loader, device)
        current_metric = retrieval_metrics['V2W_R1']
        print(f"Retrieval - V2W R@1: {current_metric:.4f}, W2V R@1: {retrieval_metrics['W2V_R1']:.4f}")

        # --- Top-K 保存逻辑 ---
        if len(saved_models) < top_k or current_metric > saved_models[-1][0]:
            filename = f"clip_epoch_{epoch+1}_v2w_{current_metric:.4f}.pth"
            save_path = os.path.join(save_dir, filename)
            
            torch.save(model.state_dict(), save_path)
            saved_models.append((current_metric, save_path))
            
            saved_models.sort(key=lambda x: x[0], reverse=True) 
            
            if len(saved_models) > top_k:
                _, path_to_remove = saved_models.pop()
                if os.path.exists(path_to_remove):
                    os.remove(path_to_remove)
            print(f"-> 模型已保存入 Top-{top_k} 队列: {filename}")