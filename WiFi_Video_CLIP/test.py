import yaml
import torch
from torch.utils.data import DataLoader, random_split

from dataset.wimans_dataset import WiMANS_CLIP_Dataset
from models.clip_model import WiMANS_CLIP
from core.evaluate import evaluate_linear_probe

def main():
    config_path = "configs/wimans_clip_config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 保证与训练时完全一致的数据集划分
    generator = torch.Generator().manual_seed(cfg['train']['seed'])
    full_dataset = WiMANS_CLIP_Dataset(config_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['train']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['train']['num_workers'])

    # 2. 实例化模型并加载预训练权重
    model = WiMANS_CLIP(
        projection_dim=cfg['model']['projection_dim'],
        init_temperature=cfg['model']['init_temperature']
    ).to(device)
    
    weight_path = cfg['test']['weight_path']
    print(f"Loading weights from: {weight_path}")
    
    # 允许加载部分权重 (如果只关心 WiFi，其实 video 的权重即使不匹配也不影响分类测试)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    
    # 3. 纯粹的下游任务评估：WiFi 单模态多标签准确率测试
    print("\n=== WiFi 单模态多用户动作识别测试 (Linear Probing) ===")
    probe_acc = evaluate_linear_probe(
        model, 
        train_loader, 
        val_loader, 
        device, 
        num_classes=54, # 6 用户 x 9 动作
        epochs=cfg['test']['linear_probe_epochs']
    )
    print(f"最终评估 -> WiFi 单模态多用户准确率: {probe_acc * 100:.2f}%")

if __name__ == "__main__":
    main()