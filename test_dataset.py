"""Self-check cho SegBaseDataset. Chay: python test_dataset.py

Kiem cai de vo trong im lang nhat: doc mask nhi phan. convert("1") truoc day
nguong o 128 + dither, nen mask luu duoi dang nhan 0/1 ra RONG HOAN TOAN —
khong exception, chi la Dice ~ 0 sau vai gio train.
"""

import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from usdsgen.data.datasets import SegBaseDataset

NAMES = ["b.png", "a.png", "c.png"]  # co tinh khong theo thu tu


def passthrough(**kwargs) -> dict:
    """Thay cho A.Compose: test nay soi khau doc mask, khong soi augmentation."""
    return kwargs


def build_split(root: Path, fg_value: int) -> None:
    img_dir = root / "training_set" / "image"
    mask_dir = root / "training_set" / "mask"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = fg_value  # 16 pixel foreground
    for name in NAMES:
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(img_dir / name)
        Image.fromarray(mask).save(mask_dir / name)


def main() -> None:
    # 1 = nhan lop, 100 = gia tri bat ky, 255 = anh nhi phan thong thuong.
    # Ca ba deu phai ra cung mot mask.
    for fg_value in (1, 100, 255):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_split(root, fg_value)
            cfg = OmegaConf.create(
                {
                    "num_classes": 2,
                    "path": {"root": str(root), "split": {"train": "training_set"}},
                }
            )
            ds = SegBaseDataset(cfg, "train", passthrough)

            mask = np.asarray(ds[0]["mask"])
            assert mask.sum() == 16, f"fg={fg_value}: {mask.sum()} pixel, ky vong 16"
            assert set(np.unique(mask)) <= {0, 1}, f"fg={fg_value}: {np.unique(mask)}"

            order = [Path(p).name for p in ds.image_list]
            assert order == sorted(NAMES), f"thu tu khong on dinh: {order}"

    print("ok")


if __name__ == "__main__":
    main()
