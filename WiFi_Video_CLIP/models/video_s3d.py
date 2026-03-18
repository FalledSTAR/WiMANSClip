import torch.nn as nn
from torchvision.models.video.s3d import s3d, S3D_Weights

class S3D_Encoder(nn.Module):
    def __init__(self, projection_dim=512):
        super(S3D_Encoder, self).__init__()
        # 1. 加载官方 Kinetics-400 预训练权重
        self.backbone = s3d(weights=S3D_Weights.KINETICS400_V1)
        
        # 2. 依然保持彻底冻结，节省海量显存以维持大 Batch Size
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. 【核心修改】：S3D 截断分类头前的特征通道数为 1024
        self.projection = nn.Linear(1024, projection_dim)

    def forward(self, x):
        # x 的输入形状: [Batch, Channels, Time_in, Height, Width]
        
        # 1. 【核心修改】：绕过 backbone 默认的分类头，只调用 .features 提取时空特征矩阵
        # 输出特征形状: [Batch, 1024, Time_out, H_out, W_out]
        feat = self.backbone.features(x)
        
        # 2. 仅在空间维度 (Height=3, Width=4) 上进行平均池化，坚决保留 Time 维度
        # 输出形状变为: [Batch, 1024, Time_out]
        feat = feat.mean(dim=[3, 4])
        
        # 3. 维度置换以匹配 nn.Linear 的要求
        # 输出形状变为: [Batch, Time_out, 1024]
        feat = feat.permute(0, 2, 1)
        
        # 4. 对视频的每一帧特征独立进行投影
        # 最终输出形状: [Batch, Time_out, 512]
        projected = self.projection(feat)
        
        return projected