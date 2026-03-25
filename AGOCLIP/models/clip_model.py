import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .video_s3d import S3D_Encoder
from .wifi_that import THAT_Encoder

class WiMANS_CLIP(nn.Module):
    # 【新增参数】：num_classes，请根据 WiMANS 的动作类别数传入 (如 150, 62 等)
    def __init__(self, projection_dim=512, init_temperature=0.07, num_classes=None):
        super(WiMANS_CLIP, self).__init__()
        self.num_classes = num_classes
        self.video_encoder = S3D_Encoder(projection_dim=projection_dim)
        self.wifi_encoder = THAT_Encoder(projection_dim=projection_dim, num_classes=num_classes)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))

    # 【新增参数】：return_logits 默认 False，保证 evaluate 脚本原逻辑不出错
    def forward(self, video_inputs, wifi_inputs, return_logits=False):
        video_features = self.video_encoder(video_inputs)
        
        if return_logits and self.num_classes is not None:
            wifi_features, wifi_logits = self.wifi_encoder(wifi_inputs, return_logits=True)
        else:
            wifi_features = self.wifi_encoder(wifi_inputs, return_logits=False)
            wifi_logits = None
            
        video_features = F.normalize(video_features, p=2, dim=-1)
        wifi_features = F.normalize(wifi_features, p=2, dim=-1)
        
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        
        if return_logits and wifi_logits is not None:
            return video_features, wifi_features, logit_scale, wifi_logits
            
        return video_features, wifi_features, logit_scale