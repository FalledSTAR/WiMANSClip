import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, soft_tau=0.5, max_temp=40.0):
        super(CLIPLoss, self).__init__()
        # 1. 标签平滑：在 Batch Size=16 的小样本下，防止模型对少量负样本过度拟合
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # 2. 软对齐温度：0.5 是学术界验证的甜点值，既不过硬，也不过于平均
        self.soft_tau = soft_tau  
        # 3. 总体温度钳制：锁死在 40.0，拒绝通过无限放大 Temp 来作弊
        self.max_temp = max_temp  

    def forward(self, video_features, wifi_features, logit_scale):
        """
        终极版细粒度跨模态对比损失 (SOTA-Level Dense Alignment)
        
        参数:
        - video_features: 形状 [Batch_Size, Time_v, D]
        - wifi_features:  形状 [Batch_Size, Time_w, D]
        - logit_scale:    可学习的温度缩放参数
        """
        device = video_features.device
        batch_size = video_features.shape[0]

        # 1. 计算 Token-wise 交叉点积矩阵
        # sim 矩阵形状: [Batch_Size, Batch_Size, Time_v, Time_w]
        sim = torch.einsum('imd,jnd->ijmn', video_features, wifi_features)

        # ==========================================================
        # 【神级优化 1】：Gradient Detach (斩断梯度耦合)
        # 将用于计算注意力的 sim 从计算图中分离，使其仅作为 Routing 指导
        # ==========================================================
        sim_detach = sim.detach()

        # ==========================================================
        # 【神级优化 2】：Soft Alignment (基于分离后的 sim)
        # ==========================================================
        # V2W 软对齐：视频帧找 WiFi 帧
        attn_v2w = F.softmax(sim_detach / self.soft_tau, dim=3) 
        soft_sim_v2w = torch.sum(sim * attn_v2w, dim=3) # [Batch, Batch, Time_v]

        # W2V 软对齐：WiFi 帧找视频帧
        attn_w2v = F.softmax(sim_detach / self.soft_tau, dim=2)
        soft_sim_w2v = torch.sum(sim * attn_w2v, dim=2) # [Batch, Batch, Time_w]

        # ==========================================================
        # 【神级优化 3】：帧级动态加权 (破除 mean() 的背景稀释)
        # ==========================================================
        # 为视频的每个时间步计算重要性权重 (越像正样本的帧，权重越高)
        frame_weight_v = F.softmax(soft_sim_v2w.detach() / self.soft_tau, dim=2)
        sim_v2w_global = torch.sum(soft_sim_v2w * frame_weight_v, dim=2) # [Batch, Batch]

        # 为 WiFi 的每个时间步计算重要性权重
        frame_weight_w = F.softmax(soft_sim_w2v.detach() / self.soft_tau, dim=2)
        sim_w2v_global = torch.sum(soft_sim_w2v * frame_weight_w, dim=2) # [Batch, Batch]

        # 双向融合
        dense_similarity = (sim_v2w_global + sim_w2v_global) / 2.0

        # ==========================================================
        # 【神级优化 4】：温度硬钳制
        # ==========================================================
        logit_scale_clamped = torch.clamp(logit_scale, max=self.max_temp)
        logits_per_video = logit_scale_clamped * dense_similarity
        logits_per_wifi = logits_per_video.T

        labels = torch.arange(batch_size, dtype=torch.long, device=device)

        loss_video = self.criterion(logits_per_video, labels)
        loss_wifi = self.criterion(logits_per_wifi, labels)

        return (loss_video + loss_wifi) / 2.0