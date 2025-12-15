"""
Cityscapes 拼接图像分割脚本。

功能：
1. 读取 `data/raw/cityscapes/{train,val}` 下的拼接 JPG（左：photo，右：label）。
2. 分割并保存到 `data/processed/{split}/photo` 与 `data/processed/{split}/label`。
3. 生成划分索引文件 `data/splits/cityscapes_split_seed42.json`（仅 train/val）。

用法：
python src/data/prepare_cityscapes.py
    --raw-dir data/raw/cityscapes
    --out-dir data/processed
    --split-file data/splits/cityscapes_split_seed42.json
    [--overwrite]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image


def split_and_save(
    img_path: Path,
    photo_path: Path,
    label_path: Path,
    overwrite: bool = False,
) -> None:
    """Split a concatenated Cityscapes image into photo/label halves and save."""
    if not overwrite and photo_path.exists() and label_path.exists():
        return

    with Image.open(img_path) as img:
        width, height = img.size
        if width % 2 != 0:
            raise ValueError(f"Image width not divisible by 2: {img_path}")

        mid = width // 2
        photo = img.crop((0, 0, mid, height)).convert("RGB")
        label = img.crop((mid, 0, width, height)).convert("RGB")

        photo_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        photo.save(photo_path, format="JPEG")
        label.save(label_path, format="PNG")


def collect_files(raw_dir: Path) -> Dict[str, List[Path]]:
    """Collect train/val file paths under raw_dir."""
    splits = {}
    for split in ("train", "val"):
        split_dir = raw_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        files = sorted(split_dir.glob("*.jpg"))
        if not files:
            raise FileNotFoundError(f"No jpg files found in {split_dir}")
        splits[split] = files
    return splits


def write_split_file(split_file: Path, splits: Dict[str, List[Path]]) -> None:
    """Write split json using original filenames (e.g., '1.jpg')."""
    split_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: [p.name for p in v] for k, v in splits.items()}
    with split_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def process_dataset(raw_dir: Path, out_dir: Path, overwrite: bool = False) -> Dict[str, List[Path]]:
    """Split entire dataset and return split mapping."""
    splits = collect_files(raw_dir)
    for split, files in splits.items():
        for img_path in files:
            stem = img_path.stem  # e.g., "1"
            photo_path = out_dir / split / "photo" / f"{stem}_photo.jpg"
            label_path = out_dir / split / "label" / f"{stem}_label.png"
            split_and_save(img_path, photo_path, label_path, overwrite=overwrite)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Cityscapes concatenated images.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/cityscapes"),
        help="Directory containing raw concatenated images.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for split images.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("data/splits/cityscapes_split_seed42.json"),
        help="Path to save split index JSON (train/val).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = process_dataset(args.raw_dir, args.out_dir, overwrite=args.overwrite)
    write_split_file(args.split_file, splits)
    print("Completed processing.")
    for split, files in splits.items():
        print(f"{split}: {len(files)} images")


if __name__ == "__main__":
    main()

