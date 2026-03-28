import os
import torch
import torchvision
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Dataset
from torchvision.models.video import S3D_Weights

class WiMANS_CLIP_Dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
            
        self.video_dir = self.cfg['data']['video_dir']
        self.wifi_dir = self.cfg['data']['wifi_dir']
        
        df = pd.read_csv(self.cfg['data']['annotation_file'], dtype=str)
        
        if 'environments' in self.cfg['data']:
            df = df[df["environment"].isin(self.cfg['data']['environments'])]
        if 'num_users' in self.cfg['data']:
            df = df[df["number_of_users"].isin(self.cfg['data']['num_users'])]
        if 'wifi_band' in self.cfg['data']:
            df = df[df["wifi_band"].isin(self.cfg['data']['wifi_band'])]
            
        loc_cols = [f"user_{i}_location" for i in range(1, 7)]
        act_cols = [f"user_{i}_activity" for i in range(1, 7)]
        self.samples = df[["label"] + loc_cols + act_cols].to_dict('records')
        
        # 物理位置到槽位的映射
        self.loc_to_idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}
        
        # 【核心修正】：严格的 10 分类体系
        self.activity_encoding = {
            "nan":      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 索引0: 绝对无人
            "nothing":  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 索引1: 有人静止
            "walk":     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 索引2
            "rotation": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 索引3
            "jump":     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 索引4
            "wave":     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 索引5
            "lie_down": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 索引6
            "pick_up":  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 索引7
            "sit_down": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # 索引8
            "stand_up": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 索引9
        }
        
        if self.cfg['model']['video_backbone'] == "S3D":
            self.video_transform = S3D_Weights.DEFAULT.transforms()
            
        self.wifi_length = self.cfg['model']['wifi_length']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sample_id = sample_info["label"]
        
        # 初始化 6 个位置均为空 (nan)
        location_activities = ["nan"] * 6
        
        for i in range(1, 7):
            loc = str(sample_info[f"user_{i}_location"])
            act = str(sample_info[f"user_{i}_activity"])
            if loc in self.loc_to_idx:
                slot_idx = self.loc_to_idx[loc]
                location_activities[slot_idx] = act
                
        encoded_y = [self.activity_encoding.get(act, self.activity_encoding["nan"]) for act in location_activities]
        label_matrix = torch.tensor(encoded_y, dtype=torch.float32) # [6, 10]
        
        # Video 读取
        video_path = os.path.join(self.video_dir, f"{sample_id}.mp4")
        data_video_x, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format="TCHW")
        video_tensor = self.video_transform(data_video_x)
        
        # WiFi 读取
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