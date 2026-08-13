"""Model builder — de-mm version.

Thay the mmseg `MODELS.build(EncoderDecoder)` bang container nn.Module thuan.
Ten thuoc tinh `backbone` / `decode_head` giu nguyen de state_dict khop 100%
voi checkpoint cu (mmseg EncoderDecoder cung dat ten nhu vay).

LUU Y: nhanh Upernet (configs/model/Seg/Upernet.yaml) phu thuoc mmseg that su
va KHONG con duoc ho tro sau refactor nay.
"""

import torch.nn as nn
from omegaconf import OmegaConf
from timm.models import create_model

from usdsgen.modules.backbone.segbackbone import HVITBackbone4Seg
from usdsgen.modules.backbone.vision_transformer import build_vit
from usdsgen.modules.head.seg.ATMHead import ATMHead
from usdsgen.utils.modelutils import load_pretrained

SEG_BACKBONES = {"HVITBackbone4Seg": HVITBackbone4Seg}
SEG_HEADS = {"ATMHead": ATMHead}


class SegModel(nn.Module):
    """Thay the mmseg EncoderDecoder: chi chua backbone + decode_head."""

    def __init__(self, backbone: nn.Module, decode_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x):
        # Tien cho inference / torch.compile; SegTrainer.step() van goi
        # backbone / decode_head truc tiep nhu truoc.
        return self.decode_head(self.backbone(x))


def _pop_type(cfg: dict) -> tuple[str, dict]:
    cfg = dict(cfg)
    return cfg.pop("type"), cfg


def build_seg_model(model_cfg: dict, logger) -> SegModel:
    backbone_type, backbone_cfg = _pop_type(model_cfg["backbone"])
    head_type, head_cfg = _pop_type(model_cfg["decode_head"])

    if backbone_type not in SEG_BACKBONES:
        raise KeyError(f"Unknown backbone type: {backbone_type}")
    if head_type not in SEG_HEADS:
        raise KeyError(f"Unknown head type: {head_type}")

    # loss_decode cfg duoc ATMHead tu build (xem ATMHead.__init__)
    backbone = SEG_BACKBONES[backbone_type](**backbone_cfg)
    head = SEG_HEADS[head_type](**head_cfg)
    head.init_weights()  # mmseg goi ngam truoc day; giu dung hanh vi cu
    return SegModel(backbone, head)


def build_model(config, logger):
    if config.model.model_type == "FM":
        if config.task == "Cls":
            model = build_vit(config.model.model_cfg, logger)
        elif config.task == "Seg":
            model_cfg = OmegaConf.to_container(config.model.model_cfg, resolve=True)
            model = build_seg_model(model_cfg, logger)
            load_pretrained(config.model.model_cfg.backbone, model.backbone, logger)
        else:
            raise NotImplementedError(f"Unknown model: {config.model}")
    else:
        model = create_model(**config.model.model_cfg)
    return model
