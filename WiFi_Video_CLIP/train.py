import os
import torch
from datetime import datetime  
from core.evaluate import evaluate_retrieval

def train_loop(model, train_loader, val_loader, criterion, optimizer, cfg, device):
    epochs = cfg['train']['epochs']
    save_dir = cfg['train']['save_dir']
    top_k = cfg.get('train', {}).get('save_top_k', 3)
    
    accumulation_steps = cfg.get('train', {}).get('accumulation_steps', 1) 
    
    os.makedirs(save_dir, exist_ok=True)
    saved_models = [] 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs, 
        eta_min=5e-5
    )
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        log_interval = 20
        
        for i, batch in enumerate(train_loader):
            video_inputs = batch["video"].to(device)
            wifi_inputs = batch["wifi"].to(device)
            
            # 1. 提取原始标签矩阵，形状: [Batch, 6, 9]
            action_labels = batch["label"].to(device) 
            
            # 2. 【核心逻辑】：聚合为场景级多标签 (Scene-level Multi-hot)
            # 只要房间内有任何一个用户在做该动作，该动作位置的值就设为 1
            # 形状变为: [Batch, 9]
            scene_labels = torch.clamp(torch.sum(action_labels, dim=1), max=1.0)
            
            # 3. 前向传播并获取 4 个返回值
            video_features, wifi_features, logit_scale, wifi_logits = model(
                video_inputs, wifi_inputs, return_logits=True
            )
            
            # 4. 传入压缩后的 scene_labels 计算联合 Loss
            loss, loss_nce, loss_ce = criterion(
                video_features, wifi_features, logit_scale, wifi_logits, scene_labels
            )
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            real_loss = loss.item() * accumulation_steps
            total_loss += real_loss
            
            if (i + 1) % log_interval == 0 or (i + 1) == len(train_loader):
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Epoch [{epoch+1}/{epochs}] | Step [{i+1}/{len(train_loader)}] "
                      f"| Total Loss: {real_loss:.4f} (NCE:{loss_nce.item():.4f}, CE:{loss_ce.item():.4f}) "
                      f"| Temp: {logit_scale.item():.2f}")
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{current_time}] ---> Epoch {epoch+1} 结束 | 平均训练 Loss: {total_loss / len(train_loader):.4f} | 当前学习率: {current_lr:.6f}")   

        # --- 验证阶段 --- (无需修改 evaluate 脚本，它默认调用的 return_logits=False)
        retrieval_metrics = evaluate_retrieval(model, val_loader, device)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Retrieval - V2W R@1: {retrieval_metrics['V2W_R1']:.4f}, W2V R@1: {retrieval_metrics['W2V_R1']:.4f}")
        print(f"[{current_time}] Retrieval - V2W R@5: {retrieval_metrics['V2W_R5']:.4f}, W2V R@5: {retrieval_metrics['W2V_R5']:.4f}")

        # --- Top-K 保存逻辑 ---
        current_v2w = retrieval_metrics['V2W_R1']
        current_w2v = retrieval_metrics['W2V_R1']

        should_save = False
        if len(saved_models) < top_k:
            should_save = True
        else:
            worst_v2w, worst_w2v, _ = saved_models[-1]
            should_save = current_v2w > worst_v2w and current_w2v > worst_w2v
            if not should_save:
                print(f"[{current_time}] -> 未进入 Top-{top_k}: 需同时超过队尾模型 | "
                      f"当前 V2W={current_v2w:.4f}, W2V={current_w2v:.4f} | "
                      f"队尾 V2W={worst_v2w:.4f}, W2V={worst_w2v:.4f}"
                )

        if should_save:
            filename = f"clip_epoch_{epoch+1}_v2w_{current_v2w:.4f}_w2v_{current_w2v:.4f}.pth"
            save_path = os.path.join(save_dir, filename)
            
            torch.save(model.state_dict(), save_path)
            saved_models.append((current_v2w, current_w2v, save_path))
            
            saved_models.sort(key=lambda x: (x[0], x[1]), reverse=True)
            
            if len(saved_models) > top_k:
                _, _, path_to_remove = saved_models.pop()
                if os.path.exists(path_to_remove):
                    os.remove(path_to_remove)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] -> 已替换并移除旧模型: {os.path.basename(path_to_remove)}")
            
            print(f"[{current_time}] -> 模型已保存入 Top-{top_k} 队列: {filename}")