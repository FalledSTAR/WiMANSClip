import torch
import torch.nn as nn

class CLIPLoss(nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        # 交叉熵损失函数，用于计算对角线上的正样本概率
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, video_features, wifi_features, logit_scale):
        """
        计算对称的 InfoNCE 对比损失。
        
        参数:
        - video_features: 经过投影和L2归一化后的视频特征矩阵，形状 [Batch_Size, D]
        - wifi_features: 经过投影和L2归一化后的WiFi特征矩阵，形状 [Batch_Size, D]
        - logit_scale: 可学习的温度缩放参数 (标量)
        """
        device = video_features.device
        batch_size = video_features.shape[0]

        # 1. 计算相似度矩阵 (Logits)
        # 因为特征已归一化，点积即为余弦相似度
        # 结果矩阵形状: [Batch_Size, Batch_Size]
        logits_per_video = logit_scale * video_features @ wifi_features.T
        logits_per_wifi = logits_per_video.T

        # 2. 生成对齐的真实标签 (Ground Truth)
        # 理想情况下，对角线元素（同一物理时间段的Video和WiFi）的相似度应该最高
        labels = torch.arange(batch_size, dtype=torch.long, device=device)

        # 3. 计算双向交叉熵损失
        # Video 检索 WiFi 的 Loss
        loss_video = self.criterion(logits_per_video, labels)
        # WiFi 检索 Video 的 Loss
        loss_wifi = self.criterion(logits_per_wifi, labels)

        # 4. 取平均得到最终的对称对比损失
        total_loss = (loss_video + loss_wifi) / 2.0

        return total_loss