import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 从同级目录导入解耦后的模态编码器
from .video_s3d import S3D_Encoder
from .wifi_that import THAT_Encoder

class WiMANS_CLIP(nn.Module):
    def __init__(self, projection_dim=512, init_temperature=0.07):
        super(WiMANS_CLIP, self).__init__()
        
        # 实例化双流编码器
        self.video_encoder = S3D_Encoder(projection_dim=projection_dim)
        self.wifi_encoder = THAT_Encoder(projection_dim=projection_dim)
        
        # 可学习的温度参数，用于缩放 logits
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))

    def forward(self, video_inputs, wifi_inputs):
        # 1. 提取降维对齐后的模态特征
        video_features = self.video_encoder(video_inputs)
        wifi_features = self.wifi_encoder(wifi_inputs)
        
        # 2. L2 归一化 (计算余弦相似度的前提)
        video_features = F.normalize(video_features, p=2, dim=-1)
        wifi_features = F.normalize(wifi_features, p=2, dim=-1)
        
        # 3. 限制温度参数防止梯度爆炸
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        
        return video_features, wifi_features, logit_scale