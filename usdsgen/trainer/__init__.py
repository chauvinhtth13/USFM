from .cls_trainer import ClsTrainer
from .seg_trainer import SegTrainer

# main.py lay trainer bang getattr(module, f"{config.task}Trainer")
__all__ = ["ClsTrainer", "SegTrainer"]
