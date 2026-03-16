import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

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

# 在文件加载后立即重定向输出
os.makedirs("../result/clip", exist_ok=True)
sys.stdout = Logger("../result/clip/training_log.txt")

def main():
    config_path = "./configs/wimans_clip_config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 全局随机种子设置
    seed = cfg['train']['seed']
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # 2. 数据集加载与划分
    full_dataset = WiMANS_CLIP_Dataset(config_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 使用包含 seed 的 generator 划分，确保绝对可复现
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # ---------------------------------------------------------
    # 输出数据集规模与划分信息
    print("\n" + "="*40)
    print(f"数据集加载完成 | 总样本数: {len(full_dataset)}")
    print(f"数据划分比例   | 训练集: {train_size} (80%) | 验证集: {val_size} (20%)")
    print("="*40 + "\n")
    # ---------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train']['num_workers'], drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['train']['num_workers'], drop_last=False)

    # 3. 初始化模型与优化器
    model = WiMANS_CLIP(
        projection_dim=cfg['model']['projection_dim'],
        init_temperature=cfg['model']['init_temperature']
    ).to(device)
    
    criterion = CLIPLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])

    # 4. 启动训练循环
    print("=== 开始训练 ===")
    train_loop(model, train_loader, val_loader, criterion, optimizer, cfg, device)

if __name__ == "__main__":
    main()