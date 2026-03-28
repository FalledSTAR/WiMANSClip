import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, soft_tau=0.5, max_temp=40.0, ce_weight=1.0):
        super(CLIPLoss, self).__init__()
        
        # 【核心修改 1：重构权重体系】
        # 必须大幅提升 nan (索引0) 的权重，让模型“害怕”在空位置乱报动作。
        # 现在的权重体系更加平衡：不再允许模型通过“全部填满”来作弊。
        self.class_weights = torch.tensor([
            1.0,  # nan (绝对无人) - 权重提升到 1.0，禁止随意虚警
            1.2,  # nothing (有人静止)
            2.0,  # walk
            2.0,  # rotation
            2.0,  # jump
            2.0,  # wave
            2.0,  # lie_down
            2.0,  # pick_up
            2.0,  # sit_down
            2.0   # stand_up
        ])
        
        # 【核心修改 2：加入 label_smoothing】
        # Label Smoothing (0.1) 可以软化 One-hot 标签，防止网络产生过度自信的极端 Logits
        # 这对缓解 Query 坍缩（所有查询输出一样的极端结果）有奇效。
        self.slot_ce_criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, 
            label_smoothing=0.1 
        )
        
        # 对比学习的 Loss 保持平滑
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.max_temp = max_temp
        self.ce_weight = ce_weight

    def forward(self, video_global, wifi_global, logit_scale, wifi_logits=None, true_labels=None):
        B = video_global.size(0)
        device = video_global.device

        # ==========================================================
        # 1. 局部固定解码的 CE 损失
        # ==========================================================
        loss_ce = torch.tensor(0.0).to(device)
        if wifi_logits is not None and true_labels is not None:
            # true_labels: [B, 6, 10], wifi_logits: [B, 6, 10]
            true_class_idx = torch.argmax(true_labels, dim=-1) # [B, 6]
            
            # 严格计算多分类交叉熵
            loss_ce = self.slot_ce_criterion(
                wifi_logits.reshape(-1, 10), 
                true_class_idx.reshape(-1)
            )
            
            # 【提取实际有人位置的特征做对比对齐】
            active_mask = (true_class_idx > 0)
            
            aggregated_wifi_feats = []
            for b in range(B):
                active_idx = torch.where(active_mask[b])[0]
                if len(active_idx) > 0:
                    agg_feat = wifi_global[b, active_idx].mean(dim=0)
                else:
                    agg_feat = wifi_global[b].mean(dim=0)
                aggregated_wifi_feats.append(agg_feat)
                
            wifi_global_aligned = torch.stack(aggregated_wifi_feats, dim=0) 
        else:
            wifi_global_aligned = wifi_global.mean(dim=1) if wifi_global.dim() == 3 else wifi_global

        # ==========================================================
        # 2. 稳定的全局 InfoNCE 对比损失
        # ==========================================================
        if video_global.dim() == 3:
            video_global = video_global.mean(dim=1)

        video_norm = F.normalize(video_global, p=2, dim=-1)
        wifi_norm = F.normalize(wifi_global_aligned, p=2, dim=-1)
        
        logit_scale_clamped = torch.clamp(logit_scale, max=self.max_temp)
        logits_per_video = logit_scale_clamped * (video_norm @ wifi_norm.t())
        logits_per_wifi = logits_per_video.t()

        labels = torch.arange(B, dtype=torch.long, device=device)
        loss_video = self.criterion(logits_per_video, labels)
        loss_wifi = self.criterion(logits_per_wifi, labels)
        loss_contrastive = (loss_video + loss_wifi) / 2.0

        total_loss = loss_contrastive + self.ce_weight * loss_ce
        return total_loss, loss_contrastive, loss_ce