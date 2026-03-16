import torch.nn as nn
from torchvision.models.video.s3d import s3d, S3D_Weights

class S3D_Encoder(nn.Module):
    def __init__(self, projection_dim=512):
        super(S3D_Encoder, self).__init__()
        # 加载官方 Kinetics-400 预训练权重
        self.backbone = s3d(weights=S3D_Weights.KINETICS400_V1)
        # 冻结 S3D 主干参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        # S3D 原特征维度 400 -> 投影到联合空间
        # 仅将最后的投影层设为可训练状态
        self.projection = nn.Linear(400, projection_dim)

    def forward(self, x):
        features = self.backbone(x)
        projected = self.projection(features)
        return projected