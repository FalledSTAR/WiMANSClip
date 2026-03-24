import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, soft_tau=0.5, max_temp=40.0, ce_weight=1.0):
        super(CLIPLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # 【修改点】：使用 BCEWithLogitsLoss 处理场景级的多动作标签
        self.ce_criterion = nn.BCEWithLogitsLoss() 
        self.soft_tau = soft_tau
        self.max_temp = max_temp
        self.ce_weight = ce_weight

    def forward(self, video_features, wifi_features, logit_scale, wifi_logits=None, true_labels=None):
        device = video_features.device
        batch_size = video_features.shape[0]

        # --- 对比学习对齐 (不变) ---
        sim = torch.einsum('imd,jnd->ijmn', video_features, wifi_features)
        sim_detach = sim.detach()

        attn_v2w = F.softmax(sim_detach / self.soft_tau, dim=3)
        soft_sim_v2w = torch.sum(sim * attn_v2w, dim=3)

        attn_w2v = F.softmax(sim_detach / self.soft_tau, dim=2)
        soft_sim_w2v = torch.sum(sim * attn_w2v, dim=2)

        frame_weight_v = F.softmax(soft_sim_v2w.detach() / self.soft_tau, dim=2)
        sim_v2w_global = torch.sum(soft_sim_v2w * frame_weight_v, dim=2)

        frame_weight_w = F.softmax(soft_sim_w2v.detach() / self.soft_tau, dim=2)
        sim_w2v_global = torch.sum(soft_sim_w2v * frame_weight_w, dim=2)

        logits_video = (sim_v2w_global + sim_w2v_global) / 2.0
        logits_wifi = logits_video.T

        logit_scale_clamped = torch.clamp(logit_scale, max=self.max_temp)
        logits_video = logits_video * logit_scale_clamped
        logits_wifi = logits_wifi * logit_scale_clamped

        labels = torch.arange(batch_size, dtype=torch.long, device=device)

        loss_video = self.criterion(logits_video, labels)
        loss_wifi = self.criterion(logits_wifi, labels)
        
        loss_contrastive = (loss_video + loss_wifi) / 2.0

        # --- 辅助分类 Loss 计算 ---
        if wifi_logits is not None and true_labels is not None:
            # wifi_logits: [Batch, 9], true_labels: [Batch, 9]
            loss_ce = self.ce_criterion(wifi_logits, true_labels)
            total_loss = loss_contrastive + self.ce_weight * loss_ce
            return total_loss, loss_contrastive, loss_ce
            
        return loss_contrastive, loss_contrastive, torch.tensor(0.0).to(device)