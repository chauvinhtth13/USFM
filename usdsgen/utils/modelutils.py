import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator as RGI

# Dinh danh format cua deploy/pretrain checkpoint. Tang so khi doi cau truc dict.
DEPLOY_FORMAT = "usdsgen-deploy-v1"
PRETRAIN_FORMAT = "usdsgen-pretrain-v1"


def unwrap_model(model):
    """Boc cac lop wrapper (Fabric _FabricModule, DDP) de lay nn.Module goc.

    Fabric boc model thanh _FabricModule; voi DDP con them mot lop nua. Peel
    `.module` cho den khi het thi duoc module that su giu state_dict "sach"
    (backbone.* / decode_head.*), khop voi cai build_seg_model tao ra.
    """
    while hasattr(model, "module"):
        model = model.module
    return model


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


# --------------------------------------------------------------------------- #
# Export checkpoint
# --------------------------------------------------------------------------- #
def save_deploy_checkpoint(path, model, model_cfg, task, meta=None):
    """Checkpoint tu chua kien truc: inference chi can load, khong viet model.

    Luu `model_cfg` (dict thuan) canh `state_dict`, nen ben load co the dung
    build_seg_model dung lai kien truc y het luc train. Kem theo tham so tien
    xu ly (img_size / mean / std) — day moi la thu hay lam sai ket qua inference
    ma khong bao loi.

    Toan bo gia tri la kieu co ban + tensor nen `torch.load(weights_only=True)`
    doc duoc, khong can tin tuong file pickle.
    """
    ckpt = {
        "format": DEPLOY_FORMAT,
        "task": task,
        "model_cfg": model_cfg,
        "state_dict": unwrap_model(model).state_dict(),
        **(meta or {}),
    }
    torch.save(ckpt, path)
    return path


# Cac tien to thuoc ve dau ra cua tac vu, khong phai bieu dien dung chung.
# Bo di khi xuat pretrain: chung gan chat voi num_classes cua tac vu cu.
_TASK_HEAD_PREFIXES = ("decode_head.", "auxiliary_head.", "head.", "fc_norm.")


def save_pretrain_checkpoint(path, model, backbone_cfg, meta=None):
    """Chi phan backbone, dung lam trong so khoi tao cho lan finetune sau.

    Format `{"model": ...}` khop voi cai `load_pretrained` doc, nen dung duoc
    truc tiep qua `model.model_cfg.backbone.pretrained=<file>`.

    - Model Seg (SegModel): lay thang thuoc tinh `.backbone`.
    - Model Cls (VisionTransformer): khong co `.backbone`, nen loc bo cac key
      thuoc classifier head.
    """
    inner = unwrap_model(model)
    backbone = getattr(inner, "backbone", None)
    if backbone is not None:
        state = backbone.state_dict()
    else:
        state = {
            k: v
            for k, v in inner.state_dict().items()
            if not k.startswith(_TASK_HEAD_PREFIXES)
        }

    ckpt = {
        "format": PRETRAIN_FORMAT,
        "model": state,
        "model_cfg": backbone_cfg,
        **(meta or {}),
    }
    torch.save(ckpt, path)
    return path


# --------------------------------------------------------------------------- #
# Load pretrained
# --------------------------------------------------------------------------- #
def load_pretrained(model_cfg, model, logger):
    def log(msg):
        if logger is not None:
            logger.info(msg)

    log(f">>>>>>>>>> Fine-tuned from {model_cfg.pretrained} ..........")
    # torch >= 2.6: weights_only=True la default; khai tuong minh cho ro rang.
    checkpoint = torch.load(model_cfg.pretrained, map_location="cpu", weights_only=True)
    checkpoint_model = (
        checkpoint.get("model", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    checkpoint_model = remap_pretrained_keys_vit(model, checkpoint_model, logger)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    missing = [k for k in msg.missing_keys if "relative_position_index" not in k]
    unexpected = list(msg.unexpected_keys)
    n_model = len(model.state_dict())
    n_loaded = n_model - len(missing)
    log(f"Missing keys ({len(missing)}): {missing}")
    log(f"Unexpected keys ({len(unexpected)}): {unexpected}")
    log(f"Loaded {n_loaded}/{n_model} tensors from pretrained")
    if len(missing) > 10 or len(unexpected) > 10:
        raise RuntimeError(
            f"Pretrained load looks broken: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys. "
            "Kiem tra qkv_bias / init_values trong model config co khop "
            "kien truc checkpoint khong."
        )

    del checkpoint_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f">>>>>>>>>> loaded successfully '{model_cfg.pretrained}'")


def remap_pretrained_keys_vit(model, checkpoint_model, logger):
    def log(msg):
        if logger is not None:
            logger.info(msg)

    # Checkpoint pre-train goc giu MOT bang rel_pos_bias dung chung; model
    # finetune giu rieng tung block. Checkpoint xuat ra tu mot lan finetune
    # (pretrain_*.pth) da o dang per-block roi -> khong co key nay, bo qua.
    shared_key = "rel_pos_bias.relative_position_bias_table"
    if shared_key in checkpoint_model:
        log("Expand the shared relative position embedding to each transformer block.")
        rel_pos_bias = checkpoint_model.pop(shared_key)
        for i in range(model.get_num_layers()):
            checkpoint_model[f"blocks.{i}.attn.relative_position_bias_table"] = (
                rel_pos_bias.clone()
            )

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                dst_patch_shape[1] * 2 - 1
            )
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                log(
                    "Position interpolate for %s from %dx%d to %dx%d"
                    % (key, src_size, src_size, dst_size, dst_size)
                )
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r**n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                all_rel_pos_bias = []

                xi, yi = np.meshgrid(dx, dy, indexing="ij")
                points = np.array([xi.ravel(), yi.ravel()]).T

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = RGI((x, y), z.T, method="cubic", bounds_error=False)
                    all_rel_pos_bias.append(
                        torch.Tensor(f(points).reshape(xi.shape))
                        .contiguous()
                        .view(-1, 1)
                        .to(rel_pos_bias.device)
                    )

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias

    return checkpoint_model
