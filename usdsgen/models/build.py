"""Model builder — de-mm version.

Thay the mmseg `MODELS.build(EncoderDecoder)` bang container nn.Module thuan.
Ten thuoc tinh `backbone` / `decode_head` giu nguyen de state_dict khop 100%
voi checkpoint cu (mmseg EncoderDecoder cung dat ten nhu vay).

LUU Y: chi ho tro SegVit (HVITBackbone4Seg + ATMHead). Cau hinh Upernet cu da
xoa vi UPerHead/FCNHead la component cua mmseg.
"""

import torch
from omegaconf import OmegaConf
from timm.models import create_model
from torch import nn

from usdsgen.modules.backbone.segbackbone import HVITBackbone4Seg
from usdsgen.modules.backbone.vision_transformer import build_vit
from usdsgen.modules.head.seg.ATMHead import ATMHead
from usdsgen.utils.modelutils import DEPLOY_FORMAT, load_pretrained

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

    def no_weight_decay(self):
        """Chuyen tiep tu backbone, them tien to cho khop ten param cua SegModel."""
        inner = getattr(self.backbone, "no_weight_decay", lambda: set())()
        return {f"backbone.{n}" for n in inner}


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
            if config.model.model_cfg.backbone.pretrained:
                load_pretrained(config.model.model_cfg.backbone, model.backbone, logger)
            elif logger is not None:
                logger.warning(
                    "backbone.pretrained = null -> train tu dau, KHONG dung trong so USFM."
                )
        else:
            raise NotImplementedError(f"Unknown model: {config.model}")
    else:
        model = create_model(**config.model.model_cfg)
    return model


def load_deploy_checkpoint(path, device="cpu", strict=True):
    """Dung lai model tu deploy checkpoint — khong can khai bao kien truc.

    Kien truc doc tu `model_cfg` nhung trong checkpoint (do
    `save_deploy_checkpoint` ghi luc train), nen inference khong the lech
    img_size / depth / num_classes so voi luc train nua.

    Tra ve (model o che do eval, dict metadata: img_size, num_classes, mean, std...).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != DEPLOY_FORMAT:
        raise ValueError(
            f"{path} khong phai deploy checkpoint ({DEPLOY_FORMAT}). "
            "Dung file export/*_deploy.pth sinh ra luc train."
        )

    task = ckpt.get("task", "Seg")
    if task == "Seg":
        model = build_seg_model(ckpt["model_cfg"], logger=None)
    elif task == "Cls":
        model = build_vit(OmegaConf.create(ckpt["model_cfg"]), logger=None)
    else:
        raise NotImplementedError(f"load_deploy_checkpoint chua ho tro task={task}")

    model.load_state_dict(ckpt["state_dict"], strict=strict)
    model.eval().to(device)

    meta = {k: v for k, v in ckpt.items() if k not in ("state_dict", "model_cfg")}
    return model, meta
