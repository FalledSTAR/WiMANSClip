import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, soft_tau=0.5, max_temp=40.0):
        super(CLIPLoss, self).__init__()
        # 标签平滑防止小 batch 过拟合
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.soft_tau = soft_tau
        self.max_temp = max_temp

    def forward(self, video_features, wifi_features, logit_scale):
        device = video_features.device
        batch_size = video_features.shape[0]

        # ==========================================================
        # 1. 批次内细粒度帧级对齐 (Dense Soft-Alignment)
        # ==========================================================
        sim = torch.einsum('imd,jnd->ijmn', video_features, wifi_features)
        sim_detach = sim.detach()

        # V2W: 视频帧找 WiFi 帧
        attn_v2w = F.softmax(sim_detach / self.soft_tau, dim=3)
        soft_sim_v2w = torch.sum(sim * attn_v2w, dim=3)

        # W2V: WiFi 帧找视频帧
        attn_w2v = F.softmax(sim_detach / self.soft_tau, dim=2)
        soft_sim_w2v = torch.sum(sim * attn_w2v, dim=2)

        # 帧级动态加权 (Frame-weighting)
        frame_weight_v = F.softmax(soft_sim_v2w.detach() / self.soft_tau, dim=2)
        sim_v2w_global = torch.sum(soft_sim_v2w * frame_weight_v, dim=2)

        frame_weight_w = F.softmax(soft_sim_w2v.detach() / self.soft_tau, dim=2)
        sim_w2v_global = torch.sum(soft_sim_w2v * frame_weight_w, dim=2)

        # 双向融合得到最终的 Batch 对齐矩阵 [Batch, Batch]
        logits_video = (sim_v2w_global + sim_w2v_global) / 2.0
        logits_wifi = logits_video.T

        # ==========================================================
        # 2. 温度硬钳制与 Loss 计算
        # ==========================================================
        logit_scale_clamped = torch.clamp(logit_scale, max=self.max_temp)
        logits_video = logits_video * logit_scale_clamped
        logits_wifi = logits_wifi * logit_scale_clamped

        # 正对角线标签
        labels = torch.arange(batch_size, dtype=torch.long, device=device)

        loss_video = self.criterion(logits_video, labels)
        loss_wifi = self.criterion(logits_wifi, labels)

        return (loss_video + loss_wifi) / 2.0