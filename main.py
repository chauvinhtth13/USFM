# --- Workaround: Hydra 1.3.x vỡ với argparse của Python 3.14 ---
# Python 3.14 thêm _check_help() validate help-string khi add_argument;
# Hydra truyền object LazyCompletionHelp (class cục bộ trong
# get_args_parser, không monkeypatch trực tiếp được) không hỗ trợ
# toán tử `in` -> "ValueError: badly formed help string".
# Vô hiệu hóa bước validate này: nó chỉ kiểm tra format chuỗi help,
# không ảnh hưởng chức năng. Xóa được khi hydra-core > 1.3.5 vá.
import argparse

argparse._ActionsContainer._check_help = lambda self, action: None
# ----------------------------------------------------------------

import hydra
import torch
from omegaconf import DictConfig

torch.set_float32_matmul_precision("high")
# Input luon co kich thuoc co dinh (img_size vuong) nen autotune cua cudnn
# chi ton vai iteration dau, sau do lai toc do cho patch_embed + fpn conv.
torch.backends.cudnn.benchmark = True


@hydra.main(
    config_path="configs",
    config_name="train",
    version_base="1.2",
)
def main(config: DictConfig):
    trainer = getattr(
        __import__("usdsgen.trainer", fromlist=[""]), f"{config.task}Trainer"
    )(config)
    if config.mode == "train":
        trainer.fit()
    elif config.mode == "test":
        trainer.test()
    else:
        raise ValueError(f"Invalid mode: {config.mode}")


if __name__ == "__main__":
    main()
