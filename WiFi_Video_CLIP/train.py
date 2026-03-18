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
 
        print(f"Retrieval - V2W R@1: {retrieval_metrics['V2W_R1']:.4f}, W2V R@1: {retrieval_metrics['W2V_R1']:.4f}")
        print(f"Retrieval - V2W R@5: {retrieval_metrics['V2W_R5']:.4f}, W2V R@5: {retrieval_metrics['W2V_R5']:.4f}")

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
                print(
                    f"-> 未进入 Top-{top_k}: 需同时超过队尾模型 | "
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
                print(f"-> 已替换并移除旧模型: {os.path.basename(path_to_remove)}")
            print(f"-> 模型已保存入 Top-{top_k} 队列: {filename}")