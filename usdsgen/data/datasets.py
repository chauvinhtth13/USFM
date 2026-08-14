import os

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import str_to_pil_interp
from torch.utils.data import Dataset
from torchvision import datasets

# Tham so chuan hoa cho nhanh Seg. Duoc ghi vao deploy checkpoint va dung lai
# o inference.py — lech mean/std la kieu loi lam ket qua te di ma khong bao gi.
SEG_NORM_MEAN = (0.485, 0.456, 0.406)
SEG_NORM_STD = (0.229, 0.224, 0.225)


def build_cls_dataset(config, logger):
    train_transforms = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (config.data.img_size, config.data.img_size),
                interpolation=str_to_pil_interp(config.data.interpolation),
            ),
            # T.RandomHorizontalFlip(),
            # A.RandomRotate90(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                std=torch.tensor(IMAGENET_DEFAULT_STD),
            ),
        ]
    )
    val_transforms = T.Compose(
        [
            T.Resize(
                (config.data.img_size, config.data.img_size),
                interpolation=str_to_pil_interp(config.data.interpolation),
            ),
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                std=torch.tensor(IMAGENET_DEFAULT_STD),
            ),
        ]
    )
    if config.data.type == "cls_imagenet":
        data_path = config.data.path
        dataset_train = datasets.ImageFolder(
            os.path.join(data_path.root, data_path.split.train),
            transform=train_transforms,
        )
        dataset_val = datasets.ImageFolder(
            os.path.join(data_path.root, data_path.split.val), transform=val_transforms
        )
        dataset_test = datasets.ImageFolder(
            os.path.join(data_path.root, data_path.split.test), transform=val_transforms
        )
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    logger.info(
        f"Build [Cls] dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}, test images = {len(dataset_test)}"
    )
    return dataset_train, dataset_val, dataset_test


def build_seg_dataset(config, logger):
    train_transforms = A.Compose(
        [
            A.Resize(width=config.data.img_size, height=config.data.img_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.ToFloat(max_value=255),
            A.Normalize(mean=SEG_NORM_MEAN, std=SEG_NORM_STD, max_pixel_value=1),
            ToTensorV2(),
        ]
    )
    val_transforms = A.Compose(
        [
            A.Resize(width=config.data.img_size, height=config.data.img_size),
            A.ToFloat(max_value=255),
            A.Normalize(mean=SEG_NORM_MEAN, std=SEG_NORM_STD, max_pixel_value=1),
            ToTensorV2(),
        ]
    )
    if config.data.type not in SEG_DATASETS:
        raise KeyError(
            f"Unknown data.type={config.data.type!r}. Ho tro: {sorted(SEG_DATASETS)}"
        )
    Dataset_class = SEG_DATASETS[config.data.type]

    dataset_train = Dataset_class(config.data, "train", train_transforms)
    dataset_val = Dataset_class(config.data, "val", val_transforms)
    dataset_test = Dataset_class(config.data, "test", val_transforms)
    logger.info(
        f"Build [Seg] dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}, test images = {len(dataset_test)}"
    )

    return dataset_train, dataset_val, dataset_test


class SegBaseDataset(Dataset):
    def __init__(self, DataConfig, stage, transforms):
        super().__init__()
        data_folder = os.path.join(DataConfig.path.root, DataConfig.path.split[stage])
        self.num_classes = DataConfig.num_classes
        self.update_datalist(data_folder)
        self.transforms = transforms

    def __getitem__(self, index):
        image_file = self.image_list[index]
        mask_file = self.mask_list[index]
        image = np.array(Image.open(image_file).convert("RGB"))
        if self.num_classes == 2:
            # KHONG dung convert("1"): no nguong o 128 va bat dither, nen mask
            # luu duoi dang nhan 0/1 se ra RONG HOAN TOAN — khong exception,
            # chi la Dice ~ 0. convert("L") + > 0 dung cho ca 0/1 va 0/255.
            mask = (np.array(Image.open(mask_file).convert("L")) > 0).astype(int)
        else:
            mask = np.array(Image.open(mask_file)).astype(int)
        image_mask = self.transforms(image=image, mask=mask)
        image_mask["img_path"] = image_file
        image_mask["mask_path"] = mask_file
        return image_mask

    def update_datalist(self, folder):
        image_path = os.path.join(folder, "image")
        mask_path = os.path.join(folder, "mask")
        # sorted: os.walk tra ve theo thu tu filesystem, doi theo may. Val/test
        # khong shuffle nen thu tu do chay thang vao thu tu dong segmetrics.csv.
        self.image_list = sorted(
            os.path.join(root, f) for root, _, files in os.walk(image_path) for f in files
        )
        self.mask_list = [i.replace(image_path, mask_path) for i in self.image_list]

    def __len__(self):
        return len(self.image_list)


# Dang ky dataset theo `data.type` trong configs/data/Seg/*.yaml.
# Them dataset moi = them mot dong o day (truoc day dung eval()).
SEG_DATASETS = {"SegBase": SegBaseDataset}
