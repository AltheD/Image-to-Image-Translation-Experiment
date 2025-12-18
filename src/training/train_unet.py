"""
U-Net基线模型训练脚本（仅L1损失，无GAN）。

训练设置：
- 优化器：Adam, lr=2e-4
- 学习率调度：线性衰减（从epoch 100开始衰减到0）
- 训练轮数：200 epochs
- Batch size：1（论文设置）
- 输入尺寸：256×256
- 损失函数：仅L1损失
"""

import argparse
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image

from src.models.unet_baseline import UNetBaseline
from src.data.dataset import CityscapesDataset
from src.data.transforms import build_transform


def save_triplet(label, generated, ground_truth, save_path):
    """保存三联图：Label / Generated / Ground Truth"""
    # 将tensor转换为numpy图像
    def tensor_to_image(tensor):
        if tensor.dim() == 3:
            img = tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img = tensor[0].permute(1, 2, 0).cpu().numpy()
        # 如果是归一化到[-1, 1]的，转回[0, 1]
        if img.min() < 0:
            img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img
    
    label_img = tensor_to_image(label)
    gen_img = tensor_to_image(generated)
    gt_img = tensor_to_image(ground_truth)
    
    # 水平拼接
    triplet = np.hstack([label_img, gen_img, gt_img])
    Image.fromarray(triplet).save(save_path)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        label = batch["label"].to(device)
        photo = batch["photo"].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        generated = model(label)
        loss = criterion(generated, photo)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 记录到tensorboard
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Train/Loss", loss.item(), global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, output_dir, writer=None, num_samples=10):
    """验证并保存三联图"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    saved_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    for batch_idx, batch in enumerate(pbar):
        label = batch["label"].to(device)
        photo = batch["photo"].to(device)
        name = batch["name"]
        
        # 前向传播
        generated = model(label)
        loss = criterion(generated, photo)
        
        total_loss += loss.item()
        num_batches += 1
        
        # 保存前num_samples个样本的三联图
        if saved_samples < num_samples:
            save_path = output_dir / "images" / f"epoch_{epoch:03d}_sample_{saved_samples:02d}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_triplet(label, generated, photo, save_path)
            saved_samples += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # 记录到tensorboard
    if writer is not None:
        writer.add_scalar("Val/Loss", avg_loss, epoch)
    
    return avg_loss


def get_lr_scheduler(optimizer, num_epochs, start_decay_epoch=100):
    """线性学习率衰减（从start_decay_epoch开始衰减到0）"""
    def lr_lambda(epoch):
        if epoch < start_decay_epoch:
            return 1.0
        else:
            return 1.0 - (epoch - start_decay_epoch) / (num_epochs - start_decay_epoch)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train U-Net Baseline Model")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Data root directory")
    parser.add_argument("--split-index", type=Path, default=Path("data/splits/cityscapes_split_seed42.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--exp-name", type=str, default="unet_baseline", help="Experiment name")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--start-decay-epoch", type=int, default=100, help="Start decay epoch")
    
    # 数据增强配置
    parser.add_argument("--aug-mode", type=str, default="basic", 
                       choices=["none", "basic", "strong"],
                       help="Augmentation mode: none/basic/strong")
    
    # 其他参数
    parser.add_argument("--num-val-samples", type=int, default=10, help="Number of validation samples to save")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # 创建输出目录
    exp_dir = args.output_dir / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "images").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    # 保存配置
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 数据增强配置
    aug_configs = {
        "none": build_transform(
            image_size=256,
            jitter=False,
            horizontal_flip=False,
            color_jitter=None,
            scale_range=None,
            normalize_mode="tanh"
        ),
        "basic": build_transform(
            image_size=256,
            jitter=True,
            horizontal_flip=True,
            color_jitter=None,
            scale_range=None,
            normalize_mode="tanh"
        ),
        "strong": build_transform(
            image_size=256,
            jitter=True,
            horizontal_flip=True,
            color_jitter=(0.2, 0.2, 0.2, 0.05),
            scale_range=(0.8, 1.2),
            normalize_mode="tanh"
        ),
    }
    
    train_transform = aug_configs[args.aug_mode]
    val_transform = aug_configs["none"]  # 验证集不使用增强
    
    # 加载数据集
    print("Loading datasets...")
    train_dataset = CityscapesDataset(
        root=args.data_root,
        split="train",
        split_index=args.split_index,
        transform=train_transform
    )
    val_dataset = CityscapesDataset(
        root=args.data_root,
        split="val",
        split_index=args.split_index,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 创建模型
    print("Creating model...")
    model = UNetBaseline(in_channels=3, out_channels=3).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = get_lr_scheduler(optimizer, args.epochs, args.start_decay_epoch)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=exp_dir / "logs" / "tensorboard")
    
    # 训练循环
    print(f"\nStarting training (aug_mode={args.aug_mode})...")
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device, epoch, exp_dir, writer, args.num_val_samples)
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Train/LR", current_lr, epoch)
        
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # 保存checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, exp_dir / "checkpoints" / "best_model.pth")
        
        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, exp_dir / "checkpoints" / f"checkpoint_epoch_{epoch:03d}.pth")
    
    writer.close()
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()









