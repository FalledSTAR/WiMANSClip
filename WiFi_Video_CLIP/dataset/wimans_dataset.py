import os
import torch
import torchvision
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Dataset
from torchvision.models.video import S3D_Weights

class WiMANS_CLIP_Dataset(Dataset):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.video_dir = self.cfg['data']['video_dir']
        self.wifi_dir = self.cfg['data']['wifi_dir']
        
        df = pd.read_csv(self.cfg['data']['annotation_file'], dtype=str)
        
        # 过滤规则
        if 'environments' in self.cfg['data']:
            df = df[df["environment"].isin(self.cfg['data']['environments'])]
        if 'num_users' in self.cfg['data']:
            df = df[df["number_of_users"].isin(self.cfg['data']['num_users'])]
        if 'wifi_band' in self.cfg['data']:
            df = df[df["wifi_band"].isin(self.cfg['data']['wifi_band'])]
            
        # 【修正 1】不再只提取 user_1，而是提取全部 6 个用户坑位的动作
        user_cols = [f"user_{i}_activity" for i in range(1, 7)]
        self.samples = df[["label"] + user_cols].to_dict('records')
        
        # 【修正 2】完全复刻 WiMANS 的 preset.py 字典 (Multi-hot 编码)
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
        
        # ==========================================
        # 【修正 3】多用户标签矩阵编码 (完全等价于原库 encode_activity)
        # ==========================================
        # 提取 6 个用户的动作状态
        user_activities = [str(sample_info[f"user_{i}_activity"]) for i in range(1, 7)]
        
        # 映射为 6 x 9 的二维矩阵
        encoded_y = [self.activity_encoding.get(act, self.activity_encoding["nan"]) for act in user_activities]
        
        # 转换为 FloatTensor，后续评估多用户时可以直接使用 BCEWithLogitsLoss
        label_matrix = torch.tensor(encoded_y, dtype=torch.float32)
        
        # ==========================================
        # Video 和 WiFi 处理保持原生不变
        # ==========================================
        video_path = os.path.join(self.video_dir, f"{sample_id}.mp4")
        # 增加 pts_unit='sec' 消除警告
        data_video_x, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format="TCHW")
        video_tensor = self.video_transform(data_video_x)
        
        wifi_path = os.path.join(self.wifi_dir, f"{sample_id}.npy")
        data_csi = np.load(wifi_path)
        pad_length = self.wifi_length - data_csi.shape[0]
        if pad_length > 0:
            data_csi = np.pad(data_csi, ((pad_length, 0), (0, 0), (0, 0), (0, 0)))
        elif pad_length < 0:
            data_csi = data_csi[:self.wifi_length]
            
        wifi_tensor = torch.tensor(data_csi, dtype=torch.float32)

        return {
            "video": video_tensor, 
            "wifi": wifi_tensor,
            "label": label_matrix  # 输出形状为 [6, 9] 的真实标注矩阵
        }