import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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

    base_cfg['train']['save_dir'] = exp_dir

    print("\n--- 实验配置参数 (Configuration) ---")
    yaml.dump(base_cfg, sys.stdout, default_flow_style=False, allow_unicode=True)
    print("-----------------------------------")
    
    with open(os.path.join(exp_dir, "config_backup.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(base_cfg, f, default_flow_style=False, allow_unicode=True)

    # =========================================================
    # 3. 基础环境与全局随机种子
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    seed = base_cfg['train']['seed']
    torch.manual_seed(seed)
    
    # 构建专门用于数据集划分的 Generator，确保绝对可复现
    generator = torch.Generator().manual_seed(seed)

    # =========================================================
    # 4. 数据集加载与随机划分 (由配置文件控制场景)
    # =========================================================
    full_dataset = WiMANS_CLIP_Dataset(base_cfg)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 恢复 80/20 随机打乱划分
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    loaded_envs = base_cfg['data'].get('environments', 'All')

    print("\n" + "="*45)
    print("数据加载与划分记录 (Data Split Record):")
    print(f"  - 当前加载的场景 (Envs from Config): {loaded_envs}")
    print(f"  - 总样本数 (Total Samples): {len(full_dataset)}")
    print(f"  - 数据划分比例 | 训练集: {train_size} (80%) | 验证集: {val_size} (20%)")
    print("="*45 + "\n")

    # =========================================================
    # 5. 构建 DataLoader (train_loader 保持 drop_last=True)
    # =========================================================
    train_loader = DataLoader(train_dataset, batch_size=base_cfg['train']['batch_size'], pin_memory=True,
                              shuffle=True, num_workers=base_cfg['train']['num_workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=base_cfg['train']['batch_size'], pin_memory=True,
                            shuffle=False, num_workers=base_cfg['train']['num_workers'], drop_last=False)

    # =========================================================
    # 6. 初始化模型、损失函数与优化器
    # =========================================================
    model = WiMANS_CLIP(
        projection_dim=base_cfg['model']['projection_dim'],
        init_temperature=base_cfg['model']['init_temperature'],
        num_classes=10
    ).to(device)
    
    criterion = CLIPLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_cfg['train']['learning_rate'])

    print("=== 开始训练 ===")
    train_loop(model, train_loader, val_loader, criterion, optimizer, base_cfg, device)

if __name__ == "__main__":
    main()