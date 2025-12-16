"""
配对图像(Label/Photo)的常用变换。

提供：
- build_transform：基础 resize/随机jitter + 随机水平翻转。
- normalize_photo：支持 tanh 模式 [-1,1] 和 0-1 模式。

说明：
- label 与 photo 空间变换需同步；颜色变换仅对 photo。
"""

from typing import Callable, Optional, Tuple

import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import random


def normalize_photo(t: torch.Tensor, mode: str = "tanh") -> torch.Tensor:
    """
    归一化 photo：
    - tanh: [-1,1] => (x - 0.5) / 0.5
    - 01: 保持 [0,1]
    """
    if mode == "tanh":
        return (t - 0.5) / 0.5
    if mode == "01":
        return t
    raise ValueError(f"Unsupported normalize mode: {mode}")


def _random_crop_pair(label: Image.Image, photo: Image.Image, size: int) -> Tuple[Image.Image, Image.Image]:
    i, j, h, w = transforms.RandomCrop.get_params(photo, output_size=(size, size))
    label = F.crop(label, i, j, h, w)
    photo = F.crop(photo, i, j, h, w)
    return label, photo


def build_transform(
    image_size: int = 256,
    jitter: bool = True,
    normalize_mode: str = "tanh",
    horizontal_flip: bool = True,
    color_jitter: Optional[Tuple[float, float, float, float]] = None,
    scale_range: Optional[Tuple[float, float]] = None,
) -> Callable:
    """
    返回一个可调用 transform(label, photo) -> (label_t, photo_t)。

    参数：
    - image_size: 输出尺寸（square）。
    - jitter: 若为 True，先 resize 到 286 后随机裁剪回 image_size。
    - normalize_mode: photo 归一化模式，tanh 或 01。
    - horizontal_flip: 是否随机水平翻转（p=0.5）。
    - color_jitter: (brightness, contrast, saturation, hue)，仅应用于 photo。
      传 None 表示不做颜色抖动；例如 (0.2, 0.2, 0.2, 0.05)。
    - scale_range: (min_scale, max_scale)，在 resize+jitter 前随机缩放，保持最长边比例；
      仅做空间缩放，label/photo 同步；传 None 则不做额外缩放。
    """
    cj_transform = (
        transforms.ColorJitter(
            brightness=color_jitter[0],
            contrast=color_jitter[1],
            saturation=color_jitter[2],
            hue=color_jitter[3],
        )
        if color_jitter is not None
        else None
    )

    def _transform(label: Image.Image, photo: Image.Image):
        # 同步 resize/jitter
        if jitter:
            base_size = 286
            target_resize = base_size
            # 额外随机缩放（在 jitter 前进行）
            if scale_range is not None:
                scale = random.uniform(scale_range[0], scale_range[1])
                target_resize = int(base_size * scale)

            # 先缩放到 target_resize，之后再根据需要裁剪/补缩放回 image_size
            label = F.resize(label, target_resize, interpolation=transforms.InterpolationMode.BICUBIC)
            photo = F.resize(photo, target_resize, interpolation=transforms.InterpolationMode.BICUBIC)

            # 避免 target_resize < image_size 时随机裁剪报错
            crop_size = min(image_size, target_resize)
            label, photo = _random_crop_pair(label, photo, crop_size)

            # 若 crop_size 小于期望的 image_size，则再统一放缩到 image_size
            if crop_size != image_size:
                label = F.resize(label, image_size, interpolation=transforms.InterpolationMode.BICUBIC)
                photo = F.resize(photo, image_size, interpolation=transforms.InterpolationMode.BICUBIC)
        else:
            target_resize = image_size
            if scale_range is not None:
                scale = random.uniform(scale_range[0], scale_range[1])
                target_resize = int(image_size * scale)

            label = F.resize(label, target_resize, interpolation=transforms.InterpolationMode.BICUBIC)
            photo = F.resize(photo, target_resize, interpolation=transforms.InterpolationMode.BICUBIC)

            # 裁剪尺寸不能超过当前图像尺寸
            crop_size = min(image_size, target_resize)
            label, photo = _random_crop_pair(label, photo, crop_size)

            # 若裁剪结果尺寸小于 image_size，再放缩到统一尺寸
            if crop_size != image_size:
                label = F.resize(label, image_size, interpolation=transforms.InterpolationMode.BICUBIC)
                photo = F.resize(photo, image_size, interpolation=transforms.InterpolationMode.BICUBIC)

        # 同步水平翻转
        if horizontal_flip and random.random() < 0.5:
            label = F.hflip(label)
            photo = F.hflip(photo)

        # 仅对 photo 做颜色抖动
        if cj_transform is not None:
            photo = cj_transform(photo)

        # 转 tensor
        label_t = F.to_tensor(label)  # 归一到 [0,1]
        photo_t = F.to_tensor(photo)

        # 仅 photo 做归一化
        photo_t = normalize_photo(photo_t, mode=normalize_mode)

        return label_t, photo_t

    return _transform

