import os
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.models.video import S3D_Weights

class WiMANS_CLIP_Dataset(Dataset):
    # 【修改】：将 config_path 改为直接接收解析好的 cfg 字典
    def __init__(self, cfg):
        self.cfg = cfg
            
        self.video_dir = self.cfg['data']['video_dir']
        self.wifi_dir = self.cfg['data']['wifi_dir']
        
        df = pd.read_csv(self.cfg['data']['annotation_file'], dtype=str)
        
        # 过滤规则 (根据传入的 cfg 动态决定场景)
        if 'environments' in self.cfg['data']:
            df = df[df["environment"].isin(self.cfg['data']['environments'])]
        if 'num_users' in self.cfg['data']:
            df = df[df["number_of_users"].isin(self.cfg['data']['num_users'])]
        if 'wifi_band' in self.cfg['data']:
            df = df[df["wifi_band"].isin(self.cfg['data']['wifi_band'])]
            
        # 提取全部 6 个用户坑位的动作
        user_cols = [f"user_{i}_activity" for i in range(1, 7)]
        self.samples = df[["label"] + user_cols].to_dict('records')
        
        # Multi-hot 编码
        self.activity_encoding = {
            "nan":      [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "nothing":  [1, 0, 0, 0, 0, 0, 0, 0, 0],
            "walk":     [0, 1, 0, 0, 0, 0, 0, 0, 0],
            "rotation": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "jump":     [0, 0, 0, 1, 0, 0, 0, 0, 0],
            "wave":     [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "lie_down": [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "pick_up":  [0, 0, 0, 0, 0, 0, 1, 0, 0],
            "sit_down": [0, 0, 0, 0, 0, 0, 0, 1, 0],
            "stand_up": [0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
        
        if self.cfg['model']['video_backbone'] == "S3D":
            self.video_transform = S3D_Weights.DEFAULT.transforms()
            
        self.wifi_length = self.cfg['model']['wifi_length']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sample_id = sample_info["label"]
        
        # 多用户标签矩阵编码
        user_activities = [str(sample_info[f"user_{i}_activity"]) for i in range(1, 7)]
        encoded_y = [self.activity_encoding.get(act, self.activity_encoding["nan"]) for act in user_activities]
        label_matrix = torch.tensor(encoded_y, dtype=torch.float32)
        
        # Video
        video_path = os.path.join(self.video_dir, f"{sample_id}.mp4")
        data_video_x, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format="TCHW")
        video_tensor = self.video_transform(data_video_x)
        
        # WiFi
        wifi_path = os.path.join(self.wifi_dir, f"{sample_id}.npy")
        data_csi = np.load(wifi_path)
        pad_length = self.wifi_length - data_csi.shape[0]
        if pad_length > 0:
            data_csi = np.pad(data_csi, ((pad_length, 0), (0, 0), (0, 0), (0, 0)))
        elif pad_length < 0:
            data_csi = data_csi[:self.wifi_length]
            
        wifi_tensor = torch.tensor(data_csi, dtype=torch.float32)

        return {
            "sample_id": sample_id,
            "video": video_tensor, 
            "wifi": wifi_tensor,
            "label": label_matrix 
        }