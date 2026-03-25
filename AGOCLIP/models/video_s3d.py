import torch.nn as nn
from torchvision.models.video.s3d import s3d, S3D_Weights

class S3D_Encoder(nn.Module):
    def __init__(self, projection_dim=512):
        super(S3D_Encoder, self).__init__()
        self.backbone = s3d(weights=S3D_Weights.KINETICS400_V1)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.projection = nn.Linear(1024, projection_dim)

    def forward(self, x):
        feat = self.backbone.features(x)
        
        # 恢复空间平均池化，保留时间维度
        feat = feat.mean(dim=[3, 4])
        
        feat = feat.permute(0, 2, 1)
        projected = self.projection(feat)
        
        return projected 