"""
测试不同数据增强配置的可视化效果。

用法：
    python src/data/test_augmentation.py --output-dir outputs/figures/augmentation_test
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import CityscapesDataset
from src.data.transforms import build_transform


def tensor_to_image(tensor):
    """将 tensor [C, H, W] 转换为 numpy 图像 [H, W, C]，范围 [0, 1]"""
    if tensor.dim() == 3:
        img = tensor.permute(1, 2, 0).numpy()
    else:
        img = tensor.numpy()
    # 如果是归一化到 [-1, 1] 的，转回 [0, 1]
    if img.min() < 0:
        img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    return img


def visualize_augmentations(output_dir: Path, num_samples: int = 6):
    """可视化三种增强配置的效果"""
    data_root = Path("data")
    split_index = Path("data/splits/cityscapes_split_seed42.json")
    
    # 三种配置
    configs = {
        "no_aug": build_transform(
            image_size=256,
            jitter=False,
            horizontal_flip=False,
            color_jitter=None,
            scale_range=None,
            normalize_mode="01",  # 可视化用 01 模式
        ),
        "basic_aug": build_transform(
            image_size=256,
            jitter=True,
            horizontal_flip=True,
            color_jitter=None,
            scale_range=None,
            normalize_mode="01",
        ),
        "strong_aug": build_transform(
            image_size=256,
            jitter=True,
            horizontal_flip=True,
            color_jitter=(0.2, 0.2, 0.2, 0.05),
            scale_range=(0.8, 1.2),
            normalize_mode="01",
        ),
    }
    
    # 加载数据集（使用验证集，确保每次看到的是同一批图片）
    datasets = {
        name: CityscapesDataset(
            root=data_root,
            split="val",
            split_index=split_index,
            transform=transform,
        )
        for name, transform in configs.items()
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 对每个样本，展示三种增强配置的效果
    for sample_idx in range(min(num_samples, len(datasets["no_aug"]))):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Sample {sample_idx + 1}: Data Augmentation Comparison", fontsize=14)
        
        for col, (config_name, dataset) in enumerate(datasets.items()):
            # 获取同一张图片（多次调用会得到不同的随机增强结果）
            # 为了对比，我们固定随机种子（但这里展示的是随机效果）
            sample = dataset[sample_idx]
            label = sample["label"]
            photo = sample["photo"]
            
            # 转换为图像格式
            label_img = tensor_to_image(label)
            photo_img = tensor_to_image(photo)
            
            # 显示 label
            axes[0, col].imshow(label_img)
            axes[0, col].set_title(f"{config_name}\nLabel", fontsize=10)
            axes[0, col].axis("off")
            
            # 显示 photo
            axes[1, col].imshow(photo_img)
            axes[1, col].set_title(f"{config_name}\nPhoto", fontsize=10)
            axes[1, col].axis("off")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"aug_comparison_sample_{sample_idx + 1}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / f'aug_comparison_sample_{sample_idx + 1}.png'}")
    
    # 额外：展示同一张图片在不同增强下的多次随机结果（展示随机性）
    print("\nGenerating random augmentation samples for sample 0...")
    sample_idx = 0
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Random Augmentation Variants (Sample 0, Strong Augmentation)", fontsize=14)
    
    strong_dataset = datasets["strong_aug"]
    for row in range(3):
        for col in range(4):
            sample = strong_dataset[sample_idx]
            photo_img = tensor_to_image(sample["photo"])
            axes[row, col].imshow(photo_img)
            axes[row, col].set_title(f"Variant {row * 4 + col + 1}", fontsize=9)
            axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_dir / "aug_random_variants.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'aug_random_variants.png'}")
    
    print(f"\n✅ Augmentation visualization complete! Check {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data augmentation effects")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/augmentation_test"),
        help="Output directory for visualization images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of samples to visualize",
    )
    args = parser.parse_args()
    
    visualize_augmentations(args.output_dir, args.num_samples)













