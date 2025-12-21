"""
CycleGAN训练脚本。

训练设置：
- 优化器：Adam, lr=2e-4（前100 epochs），然后线性衰减到0（后100 epochs）
- 训练轮数：200 epochs
- Batch size：1（论文设置）
- 输入尺寸：256×256
- 数据增强：随机jitter + 随机水平翻转

损失函数：
- 对抗损失（GAN Loss）
- 循环一致性损失（Cycle Consistency Loss）
- 身份损失（Identity Loss，可选）

四个网络：
- G: Label → Photo
- F: Photo → Label
- D_photo: 区分真实photo和生成的photo
- D_label: 区分真实label和生成的label
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

from src.models.cyclegan_generator import CycleGANGenerator
from src.models.cyclegan_discriminator import CycleGANDiscriminator
from src.data.dataset import CityscapesDataset
from src.data.transforms import build_transform
from src.eval.metrics import evaluate_batch


class GANLoss(nn.Module):
    """GAN损失函数（使用MSE Loss替代BCE Loss，更稳定）"""
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()
    
    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        
        target_tensor = target_tensor.expand_as(prediction)
        loss = self.loss(prediction, target_tensor)
        return loss


def save_triplet(label, generated, ground_truth, save_path, title="Label | Generated | Ground Truth"):
    """保存三联图：Label / Generated / Ground Truth"""
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


def save_cycle_triplet(label, fake_photo, recovered_label, photo, fake_label, recovered_photo, save_path):
    """保存CycleGAN的双向循环结果"""
    def tensor_to_image(tensor):
        if tensor.dim() == 3:
            img = tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img = tensor[0].permute(1, 2, 0).cpu().numpy()
        if img.min() < 0:
            img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img
    
    label_img = tensor_to_image(label)
    fake_photo_img = tensor_to_image(fake_photo)
    recovered_label_img = tensor_to_image(recovered_label)
    photo_img = tensor_to_image(photo)
    fake_label_img = tensor_to_image(fake_label)
    recovered_photo_img = tensor_to_image(recovered_photo)
    
    # 第一行：Label → Photo → Label
    row1 = np.hstack([label_img, fake_photo_img, recovered_label_img])
    # 第二行：Photo → Label → Photo
    row2 = np.hstack([photo_img, fake_label_img, recovered_photo_img])
    # 垂直拼接
    cycle_result = np.vstack([row1, row2])
    Image.fromarray(cycle_result).save(save_path)


def train_epoch(
    generator_G, generator_F,
    discriminator_photo, discriminator_label,
    dataloader, gan_loss, l1_loss,
    optimizer_G, optimizer_F,
    optimizer_D_photo, optimizer_D_label,
    device, epoch, lambda_cycle=10.0, lambda_identity=0.5,
    writer=None
):
    """训练一个epoch"""
    generator_G.train()
    generator_F.train()
    discriminator_photo.train()
    discriminator_label.train()
    
    total_loss_G = 0.0
    total_loss_F = 0.0
    total_loss_D_photo = 0.0
    total_loss_D_label = 0.0
    total_loss_cycle = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        label = batch["label"].to(device)
        photo = batch["photo"].to(device)
        
        batch_size = label.size(0)
        
        # ========== 训练判别器 D_photo ==========
        optimizer_D_photo.zero_grad()
        
        # 真实photo
        real_photo = photo
        pred_real_photo = discriminator_photo(real_photo)
        loss_D_photo_real = gan_loss(pred_real_photo, True)
        
        # 生成photo（detach，不更新生成器）
        fake_photo = generator_G(label.detach())
        pred_fake_photo = discriminator_photo(fake_photo.detach())
        loss_D_photo_fake = gan_loss(pred_fake_photo, False)
        
        # 判别器总损失
        loss_D_photo = (loss_D_photo_real + loss_D_photo_fake) * 0.5
        loss_D_photo.backward()
        optimizer_D_photo.step()
        
        # ========== 训练判别器 D_label ==========
        optimizer_D_label.zero_grad()
        
        # 真实label
        real_label = label
        pred_real_label = discriminator_label(real_label)
        loss_D_label_real = gan_loss(pred_real_label, True)
        
        # 生成label（detach，不更新生成器）
        fake_label = generator_F(photo.detach())
        pred_fake_label = discriminator_label(fake_label.detach())
        loss_D_label_fake = gan_loss(pred_fake_label, False)
        
        # 判别器总损失
        loss_D_label = (loss_D_label_real + loss_D_label_fake) * 0.5
        loss_D_label.backward()
        optimizer_D_label.step()
        
        # ========== 训练生成器 G 和 F ==========
        optimizer_G.zero_grad()
        optimizer_F.zero_grad()
        
        # 对抗损失
        fake_photo = generator_G(label)
        pred_fake_photo = discriminator_photo(fake_photo)
        loss_GAN_G = gan_loss(pred_fake_photo, True)
        
        fake_label = generator_F(photo)
        pred_fake_label = discriminator_label(fake_label)
        loss_GAN_F = gan_loss(pred_fake_label, True)
        
        # 循环一致性损失
        recovered_label = generator_F(fake_photo)
        loss_cycle_forward = l1_loss(recovered_label, label)
        
        recovered_photo = generator_G(fake_label)
        loss_cycle_backward = l1_loss(recovered_photo, photo)
        
        loss_cycle = (loss_cycle_forward + loss_cycle_backward) * lambda_cycle
        
        # 身份损失（可选）
        if lambda_identity > 0:
            # G(photo) ≈ photo, F(label) ≈ label
            identity_photo = generator_G(photo)
            loss_identity_G = l1_loss(identity_photo, photo)
            
            identity_label = generator_F(label)
            loss_identity_F = l1_loss(identity_label, label)
            
            loss_identity = (loss_identity_G + loss_identity_F) * lambda_identity
        else:
            loss_identity = torch.tensor(0.0, device=device)
        
        # 生成器总损失
        loss_G = loss_GAN_G + loss_cycle + loss_identity
        loss_F = loss_GAN_F + loss_cycle + loss_identity
        
        loss_G.backward()
        loss_F.backward()
        
        optimizer_G.step()
        optimizer_F.step()
        
        # 记录损失
        total_loss_G += loss_G.item()
        total_loss_F += loss_F.item()
        total_loss_D_photo += loss_D_photo.item()
        total_loss_D_label += loss_D_label.item()
        total_loss_cycle += loss_cycle.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            "G": f"{loss_G.item():.4f}",
            "F": f"{loss_F.item():.4f}",
            "D_photo": f"{loss_D_photo.item():.4f}",
            "D_label": f"{loss_D_label.item():.4f}",
            "Cycle": f"{loss_cycle.item():.4f}",
        })
        
        # 记录到tensorboard
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Train/Loss_G", loss_G.item(), global_step)
            writer.add_scalar("Train/Loss_F", loss_F.item(), global_step)
            writer.add_scalar("Train/Loss_D_photo", loss_D_photo.item(), global_step)
            writer.add_scalar("Train/Loss_D_label", loss_D_label.item(), global_step)
            writer.add_scalar("Train/Loss_Cycle", loss_cycle.item(), global_step)
            if lambda_identity > 0:
                writer.add_scalar("Train/Loss_Identity", loss_identity.item(), global_step)
    
    avg_loss_G = total_loss_G / num_batches if num_batches > 0 else 0.0
    avg_loss_F = total_loss_F / num_batches if num_batches > 0 else 0.0
    avg_loss_D_photo = total_loss_D_photo / num_batches if num_batches > 0 else 0.0
    avg_loss_D_label = total_loss_D_label / num_batches if num_batches > 0 else 0.0
    avg_loss_cycle = total_loss_cycle / num_batches if num_batches > 0 else 0.0
    
    return {
        "loss_G": avg_loss_G,
        "loss_F": avg_loss_F,
        "loss_D_photo": avg_loss_D_photo,
        "loss_D_label": avg_loss_D_label,
        "loss_cycle": avg_loss_cycle,
    }


@torch.no_grad()
def validate(
    generator_G, generator_F,
    dataloader, l1_loss, device, epoch, output_dir,
    writer=None, num_samples=10
):
    """验证并保存三联图"""
    generator_G.eval()
    generator_F.eval()
    
    total_loss_cycle = 0.0
    total_metrics = {"psnr": 0.0, "ssim": 0.0, "mae": 0.0}
    num_batches = 0
    saved_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    for batch_idx, batch in enumerate(pbar):
        label = batch["label"].to(device)
        photo = batch["photo"].to(device)
        name = batch["name"]
        
        # 前向传播
        fake_photo = generator_G(label)
        fake_label = generator_F(photo)
        
        # 循环一致性
        recovered_label = generator_F(fake_photo)
        recovered_photo = generator_G(fake_label)
        
        loss_cycle_forward = l1_loss(recovered_label, label)
        loss_cycle_backward = l1_loss(recovered_photo, photo)
        loss_cycle = loss_cycle_forward + loss_cycle_backward
        
        total_loss_cycle += loss_cycle.item()
        
        # 评估指标（Label→Photo方向）
        metrics = evaluate_batch(fake_photo, photo)
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        num_batches += 1
        
        # 保存前num_samples个样本的三联图
        if saved_samples < num_samples:
            # Label→Photo方向
            save_path = output_dir / "images" / f"epoch_{epoch:03d}_sample_{saved_samples:02d}_label2photo.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_triplet(label, fake_photo, photo, save_path, "Label | Generated Photo | Ground Truth Photo")
            
            # Photo→Label方向
            save_path = output_dir / "images" / f"epoch_{epoch:03d}_sample_{saved_samples:02d}_photo2label.png"
            save_triplet(photo, fake_label, label, save_path, "Photo | Generated Label | Ground Truth Label")
            
            # 循环一致性结果
            save_path = output_dir / "images" / f"epoch_{epoch:03d}_sample_{saved_samples:02d}_cycle.png"
            save_cycle_triplet(
                label, fake_photo, recovered_label,
                photo, fake_label, recovered_photo,
                save_path
            )
            
            saved_samples += 1
        
        pbar.set_postfix({
            "cycle_loss": f"{loss_cycle.item():.4f}",
            "psnr": f"{metrics['psnr']:.2f}",
        })
    
    avg_loss_cycle = total_loss_cycle / num_batches if num_batches > 0 else 0.0
    avg_metrics = {key: val / num_batches for key, val in total_metrics.items()}
    
    # 记录到tensorboard
    if writer is not None:
        writer.add_scalar("Val/Loss_Cycle", avg_loss_cycle, epoch)
        writer.add_scalar("Val/PSNR", avg_metrics["psnr"], epoch)
        writer.add_scalar("Val/SSIM", avg_metrics["ssim"], epoch)
        writer.add_scalar("Val/MAE", avg_metrics["mae"], epoch)
    
    return avg_loss_cycle, avg_metrics


def get_lr_scheduler(optimizer, num_epochs, start_decay_epoch=100):
    """线性学习率衰减（从start_decay_epoch开始衰减到0）"""
    def lr_lambda(epoch):
        if epoch < start_decay_epoch:
            return 1.0
        else:
            return 1.0 - (epoch - start_decay_epoch) / (num_epochs - start_decay_epoch)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train CycleGAN Model")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Data root directory")
    parser.add_argument("--split-index", type=Path, default=Path("data/splits/cityscapes_split_seed42.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--exp-name", type=str, default="cyclegan", help="Experiment name")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--start-decay-epoch", type=int, default=100, help="Start decay epoch")
    
    # 损失权重
    parser.add_argument("--lambda-cycle", type=float, default=10.0, help="Cycle consistency loss weight")
    parser.add_argument("--lambda-identity", type=float, default=0.5, help="Identity loss weight (0 to disable)")
    
    # 模型参数
    parser.add_argument("--n-residual-blocks", type=int, default=9, help="Number of residual blocks in generator")
    
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
    
    # 数据增强配置（训练集使用基础增强，验证集不使用增强）
    train_transform = build_transform(
        image_size=256,
        jitter=True,
        horizontal_flip=True,
        color_jitter=None,
        scale_range=None,
        normalize_mode="tanh"
    )
    val_transform = build_transform(
        image_size=256,
        jitter=False,
        horizontal_flip=False,
        color_jitter=None,
        scale_range=None,
        normalize_mode="tanh"
    )
    
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
    print("Creating models...")
    generator_G = CycleGANGenerator(
        in_channels=3,
        out_channels=3,
        n_residual_blocks=args.n_residual_blocks
    ).to(device)
    generator_F = CycleGANGenerator(
        in_channels=3,
        out_channels=3,
        n_residual_blocks=args.n_residual_blocks
    ).to(device)
    discriminator_photo = CycleGANDiscriminator(in_channels=3).to(device)
    discriminator_label = CycleGANDiscriminator(in_channels=3).to(device)
    
    print(f"Generator G parameters: {sum(p.numel() for p in generator_G.parameters()):,}")
    print(f"Generator F parameters: {sum(p.numel() for p in generator_F.parameters()):,}")
    print(f"Discriminator photo parameters: {sum(p.numel() for p in discriminator_photo.parameters()):,}")
    print(f"Discriminator label parameters: {sum(p.numel() for p in discriminator_label.parameters()):,}")
    
    # 损失函数和优化器
    gan_loss = GANLoss()
    l1_loss = nn.L1Loss()
    
    optimizer_G = optim.Adam(generator_G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_F = optim.Adam(generator_F.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_photo = optim.Adam(discriminator_photo.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_label = optim.Adam(discriminator_label.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    scheduler_G = get_lr_scheduler(optimizer_G, args.epochs, args.start_decay_epoch)
    scheduler_F = get_lr_scheduler(optimizer_F, args.epochs, args.start_decay_epoch)
    scheduler_D_photo = get_lr_scheduler(optimizer_D_photo, args.epochs, args.start_decay_epoch)
    scheduler_D_label = get_lr_scheduler(optimizer_D_label, args.epochs, args.start_decay_epoch)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=exp_dir / "logs" / "tensorboard")
    
    # 训练历史记录
    history = {
        "train_loss_G": [],
        "train_loss_F": [],
        "train_loss_D_photo": [],
        "train_loss_D_label": [],
        "train_loss_cycle": [],
        "val_loss_cycle": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_mae": [],
    }
    
    # 训练循环
    print(f"\nStarting training...")
    best_val_psnr = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_losses = train_epoch(
            generator_G, generator_F,
            discriminator_photo, discriminator_label,
            train_loader, gan_loss, l1_loss,
            optimizer_G, optimizer_F,
            optimizer_D_photo, optimizer_D_label,
            device, epoch,
            lambda_cycle=args.lambda_cycle,
            lambda_identity=args.lambda_identity,
            writer=writer
        )
        
        # 验证
        val_loss_cycle, val_metrics = validate(
            generator_G, generator_F,
            val_loader, l1_loss, device, epoch, exp_dir,
            writer=writer, num_samples=args.num_val_samples
        )
        
        # 学习率调度
        scheduler_G.step()
        scheduler_F.step()
        scheduler_D_photo.step()
        scheduler_D_label.step()
        current_lr = optimizer_G.param_groups[0]["lr"]
        writer.add_scalar("Train/LR", current_lr, epoch)
        
        # 记录历史
        history["train_loss_G"].append(train_losses["loss_G"])
        history["train_loss_F"].append(train_losses["loss_F"])
        history["train_loss_D_photo"].append(train_losses["loss_D_photo"])
        history["train_loss_D_label"].append(train_losses["loss_D_label"])
        history["train_loss_cycle"].append(train_losses["loss_cycle"])
        history["val_loss_cycle"].append(val_loss_cycle)
        history["val_psnr"].append(val_metrics["psnr"])
        history["val_ssim"].append(val_metrics["ssim"])
        history["val_mae"].append(val_metrics["mae"])
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train - G: {train_losses['loss_G']:.4f}, F: {train_losses['loss_F']:.4f}, "
              f"D_photo: {train_losses['loss_D_photo']:.4f}, D_label: {train_losses['loss_D_label']:.4f}, "
              f"Cycle: {train_losses['loss_cycle']:.4f}")
        print(f"  Val - Cycle Loss: {val_loss_cycle:.4f}, PSNR: {val_metrics['psnr']:.2f}, "
              f"SSIM: {val_metrics['ssim']:.4f}, MAE: {val_metrics['mae']:.4f}, LR: {current_lr:.6f}")
        
        # 保存checkpoint
        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            torch.save({
                "epoch": epoch,
                "generator_G_state_dict": generator_G.state_dict(),
                "generator_F_state_dict": generator_F.state_dict(),
                "discriminator_photo_state_dict": discriminator_photo.state_dict(),
                "discriminator_label_state_dict": discriminator_label.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_F_state_dict": optimizer_F.state_dict(),
                "optimizer_D_photo_state_dict": optimizer_D_photo.state_dict(),
                "optimizer_D_label_state_dict": optimizer_D_label.state_dict(),
                "scheduler_G_state_dict": scheduler_G.state_dict(),
                "scheduler_F_state_dict": scheduler_F.state_dict(),
                "scheduler_D_photo_state_dict": scheduler_D_photo.state_dict(),
                "scheduler_D_label_state_dict": scheduler_D_label.state_dict(),
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "val_mae": val_metrics["mae"],
            }, exp_dir / "checkpoints" / "best_model.pth")
        
        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "generator_G_state_dict": generator_G.state_dict(),
                "generator_F_state_dict": generator_F.state_dict(),
                "discriminator_photo_state_dict": discriminator_photo.state_dict(),
                "discriminator_label_state_dict": discriminator_label.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_F_state_dict": optimizer_F.state_dict(),
                "optimizer_D_photo_state_dict": optimizer_D_photo.state_dict(),
                "optimizer_D_label_state_dict": optimizer_D_label.state_dict(),
                "scheduler_G_state_dict": scheduler_G.state_dict(),
                "scheduler_F_state_dict": scheduler_F.state_dict(),
                "scheduler_D_photo_state_dict": scheduler_D_photo.state_dict(),
                "scheduler_D_label_state_dict": scheduler_D_label.state_dict(),
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "val_mae": val_metrics["mae"],
            }, exp_dir / "checkpoints" / f"checkpoint_epoch_{epoch:03d}.pth")
    
    # 保存最终模型和历史
    torch.save({
        "generator_G_state_dict": generator_G.state_dict(),
        "generator_F_state_dict": generator_F.state_dict(),
        "discriminator_photo_state_dict": discriminator_photo.state_dict(),
        "discriminator_label_state_dict": discriminator_label.state_dict(),
    }, exp_dir / "checkpoints" / "model_final.pth")
    
    with open(exp_dir / "logs" / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    writer.close()
    print(f"\n✅ Training complete! Best val PSNR: {best_val_psnr:.2f}")
    print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()




