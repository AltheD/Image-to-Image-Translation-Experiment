"""
Cityscapes 配对数据集（已分割的 label/photo）。

- 读取 `data/processed/{split}/{label,photo}`。
- 支持外部传入 transform(label, photo) -> (label_t, photo_t)。
"""

from pathlib import Path
from typing import Callable, Dict, Tuple, Any, List

from PIL import Image
from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        split_index: Path,
        transform: Callable[[Image.Image, Image.Image], Tuple[Any, Any]] = None,
    ):
        """
        Args:
            root: 数据根目录，期望包含 processed/{split}/{photo,label}
            split: "train" 或 "val"
            split_index: 划分文件 JSON，包含 train/val 文件名列表
            transform: 可调用，接收 (label_img, photo_img) 返回 (label_t, photo_t)
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        import json

        with open(split_index, "r", encoding="utf-8") as f:
            splits = json.load(f)
        if split not in splits:
            raise ValueError(f"split '{split}' not found in {split_index}")

        self.files: List[str] = splits[split]
        self.photo_dir = self.root / "processed" / split / "photo"
        self.label_dir = self.root / "processed" / split / "label"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        name = self.files[idx]
        stem = Path(name).stem
        photo_path = self.photo_dir / f"{stem}_photo.jpg"
        label_path = self.label_dir / f"{stem}_label.png"

        photo = Image.open(photo_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        if self.transform:
            label, photo = self.transform(label, photo)

        return {"label": label, "photo": photo, "name": name}

