import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .video_s3d import S3D_Encoder
from .wifi_that import THAT_Encoder

class WiMANS_CLIP(nn.Module):
    def __init__(self, projection_dim=512, init_temperature=0.07):
        super(WiMANS_CLIP, self).__init__()
        self.video_encoder = S3D_Encoder(projection_dim=projection_dim)
        self.wifi_encoder = THAT_Encoder(projection_dim=projection_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))

    def forward(self, video_inputs, wifi_inputs):
        video_features = self.video_encoder(video_inputs)
        wifi_features = self.wifi_encoder(wifi_inputs)
        
        video_features = F.normalize(video_features, p=2, dim=-1)
        wifi_features = F.normalize(wifi_features, p=2, dim=-1)
        
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        return video_features, wifi_features, logit_scale