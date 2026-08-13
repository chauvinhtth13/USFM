import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from .datasets import build_cls_dataset, build_seg_dataset


def build_loader(config, logger):
    if "cls" in config.data.type.lower():
        dataset_train, dataset_val, dataset_test = build_cls_dataset(config, logger)
    elif "seg" in config.data.type.lower():
        dataset_train, dataset_val, dataset_test = build_seg_dataset(config, logger)
    else:
        raise NotImplementedError("We only support seg and cls now.")

    logger.info(
        f"Finally build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}, test images = {len(dataset_test)}"
    )

    # dataloader will be setup by fabric, so no distribution sample here
    common = dict(
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.num_workers > 0,
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.data.batch_size,
        shuffle=True,  # FIX (audit): ban cu KHONG shuffle train
        drop_last=True,
        **common,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.data.batch_size,
        drop_last=False,
        **common,
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config.data.batch_size,
        drop_last=False,
        **common,
    )

    return dataloader_train, dataloader_val, dataloader_test
