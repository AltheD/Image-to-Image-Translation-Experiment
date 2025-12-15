"""
配对图像(Label/Photo)的常用变换。

提供：
- build_transform：基础 resize/随机jitter + 随机水平翻转。
- normalize_photo：支持 tanh 模式 [-1,1] 和 0-1 模式。

说明：
- label 与 photo 空间变换需同步；颜色变换仅对 photo。
"""

from typing import Callable, Tuple

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
) -> Callable:
    """
    返回一个可调用 transform(label, photo) -> (label_t, photo_t)。

    参数：
    - image_size: 输出尺寸（square）。
    - jitter: 若为 True，先 resize 到 286 后随机裁剪回 image_size。
    - normalize_mode: photo 归一化模式，tanh 或 01。
    - horizontal_flip: 是否随机水平翻转（p=0.5）。
    """

    def _transform(label: Image.Image, photo: Image.Image):
        # 同步 resize/jitter
        if jitter:
            label = F.resize(label, 286, interpolation=transforms.InterpolationMode.BICUBIC)
            photo = F.resize(photo, 286, interpolation=transforms.InterpolationMode.BICUBIC)
            label, photo = _random_crop_pair(label, photo, image_size)
        else:
            label = F.resize(label, image_size, interpolation=transforms.InterpolationMode.BICUBIC)
            photo = F.resize(photo, image_size, interpolation=transforms.InterpolationMode.BICUBIC)

        # 同步水平翻转
        if horizontal_flip and random.random() < 0.5:
            label = F.hflip(label)
            photo = F.hflip(photo)

        # 转 tensor
        label_t = F.to_tensor(label)  # 归一到 [0,1]
        photo_t = F.to_tensor(photo)

        # 仅 photo 做归一化
        photo_t = normalize_photo(photo_t, mode=normalize_mode)

        return label_t, photo_t

    return _transform

