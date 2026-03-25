import os
import sys
import yaml
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

from dataset.wimans_dataset import WiMANS_CLIP_Dataset
from models.clip_model import WiMANS_CLIP
from core.loss import CLIPLoss
from train import train_loop

# 双通道日志拦截器
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    # =========================================================
    # 1. 生成时间戳，创建统一的实验输出文件夹
    # =========================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"../result/clip/ago_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # 立即重定向日志到该文件夹下
    sys.stdout = Logger(os.path.join(exp_dir, "training_log.txt"))
    
    print(f"========== 实验初始化 ==========")
    print(f"时间戳: {timestamp}")
    print(f"实验目录: {exp_dir}")

    # =========================================================
    # 2. 读取、打印并备份配置参数
    # =========================================================
    config_path = "./configs/wimans_clip_config.yaml"
    with open(config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # 强制将模型的保存目录重定向到当前的专属文件夹
    base_cfg['train']['save_dir'] = exp_dir

    print("\n--- 实验配置参数 (Configuration) ---")
    yaml.dump(base_cfg, sys.stdout, default_flow_style=False, allow_unicode=True)
    print("-----------------------------------")
    
    # 备份 config.yaml 文件到实验目录，确保随时可复现
    with open(os.path.join(exp_dir, "config_backup.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(base_cfg, f, default_flow_style=False, allow_unicode=True)

    # =========================================================
    # 3. 基础环境与随机种子
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    seed = base_cfg['train']['seed']
    torch.manual_seed(seed)

    # =========================================================
    # 4. 数据集场景隔离与划分记录
    # =========================================================
    train_cfg = copy.deepcopy(base_cfg)
    val_cfg = copy.deepcopy(base_cfg)

    # 【可以在这里动态修改场景配置】
    train_envs = ['classroom', 'meeting_room']
    val_envs = ['empty_room']
    train_cfg['data']['environments'] = train_envs
    val_cfg['data']['environments'] = val_envs

    train_dataset = WiMANS_CLIP_Dataset(train_cfg)
    val_dataset = WiMANS_CLIP_Dataset(val_cfg)

    print("\n" + "="*45)
    print("数据划分记录 (Data Split Record):")
    print(f"  - 训练集场景 (Train Envs): {train_envs}")
    print(f"  - 验证集场景 (Val Envs):   {val_envs}")
    print(f"  - 数据划分规模 | 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
    print("="*45 + "\n")

    # =========================================================
    # 5. 构建 DataLoader (切记 train_loader 要 drop_last=True)
    # =========================================================
    train_loader = DataLoader(train_dataset, batch_size=base_cfg['train']['batch_size'], 
                              shuffle=True, num_workers=base_cfg['train']['num_workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=base_cfg['train']['batch_size'], 
                            shuffle=False, num_workers=base_cfg['train']['num_workers'], drop_last=False)

    # =========================================================
    # 6. 初始化模型、损失函数与优化器
    # =========================================================
    model = WiMANS_CLIP(
        projection_dim=base_cfg['model']['projection_dim'],
        init_temperature=base_cfg['model']['init_temperature'],
        num_classes=9
    ).to(device)
    
    criterion = CLIPLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_cfg['train']['learning_rate'])

    # =========================================================
    # 7. 启动训练
    # =========================================================
    print("=== 开始训练 ===")
    train_loop(model, train_loader, val_loader, criterion, optimizer, base_cfg, device)

if __name__ == "__main__":
    main()