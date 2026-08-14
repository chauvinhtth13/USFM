"""Inference script cho USFM segmentation (SegVit = HVITBackbone4Seg + ATMHead).

Chay doc lap voi trainer: chi can checkpoint + anh dau vao.
Tien xu ly KHOP CHINH XAC val_transforms cua pipeline train
(Resize -> /255 -> Normalize ImageNet -> CHW). Lech buoc nao thi ket qua
te hon luc validate ma KHONG bao loi.

Xuat ra:
  <output>/mask/       mask nhi phan 0/255, kich thuoc = anh goc
  <output>/overlay/    anh goc + vung du doan (to mau + vien)
  <output>/panel/      bang tong hop: goc | overlay | so voi GT | do tin cay
  <output>/report.csv  thong ke tung anh

Vi du:
    # Deploy checkpoint (khuyen dung): kien truc + img_size + mean/std nam san
    # trong file, khong can truyen --img-size / --num-classes
    python inference.py --ckpt logs/.../export/best_deploy.pth --input anh.png

    # ca thu muc
    python inference.py --ckpt logs/.../export/best_deploy.pth \\
        --input datasets/Seg/muscle_subj/test_set/image \\
        --output test_out

    # Checkpoint train thuong van chay duoc, nhung phai tu khai kien truc
    python inference.py --ckpt logs/.../best92.pth --input anh.png --img-size 512

Luu y: voi checkpoint train thuong, --img-size PHAI trung luc train (mac dinh 512).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from usdsgen.models.build import build_seg_model, load_deploy_checkpoint
from usdsgen.utils.modelutils import DEPLOY_FORMAT

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLR_PRED = (0, 255, 128)  # xanh la    — vung du doan
CLR_TP = (0, 220, 90)  # xanh la    — dung
CLR_FP = (255, 70, 70)  # do         — thua
CLR_FN = (70, 140, 255)  # xanh duong — bo sot


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def default_model_cfg(img_size: int, num_classes: int) -> dict:
    """Khop configs/model/Seg/SegVit.yaml (da bat qkv_bias / init_values)."""
    return {
        "backbone": {
            "type": "HVITBackbone4Seg",
            "pretrained": None,  # trong so lay tu checkpoint fine-tune
            "img_size": img_size,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "drop_path_rate": 0.0,  # inference: tat drop path
            "attn_drop_rate": 0.0,
            "drop_rate": 0.0,
            "out_indices": [5, 7, 11],
            "use_abs_pos_emb": False,
            "use_rel_pos_bias": True,
            "qkv_bias": True,
            "init_values": 0.1,
        },
        "decode_head": {
            "type": "ATMHead",
            "img_size": img_size,
            "in_channels": 768,
            "channels": 768,
            "num_classes": num_classes,
            "num_layers": 3,
            "num_heads": 12,
            "use_stages": 3,
            "embed_dims": 384,
            "loss_decode": {
                "type": "ATMLoss",
                "num_classes": num_classes,
                "dec_layers": 3,
                "loss_weight": 1.0,
            },
        },
    }


def extract_state_dict(ckpt) -> dict:
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint khong phai dict")
    sd = ckpt
    for key in ("model", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            sd = ckpt[key]
            break
    sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    for prefix in ("module.", "_orig_mod."):  # DDP / torch.compile
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix) :]: v for k, v in sd.items()}
    return sd


def load_model(ckpt_path: str, img_size: int, num_classes: int, device: str):
    """Nap model tu checkpoint. Tra ve (model, cfg) voi cfg gom img_size/mean/std.

    Uu tien deploy checkpoint (export/*_deploy.pth): kien truc va tham so tien
    xu ly nam san trong file, khong can doan --img-size / --num-classes nua.
    Checkpoint train thuong (best*.pth) van dung duoc theo duong cu.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # --- deploy checkpoint: tu chua kien truc ---
    if isinstance(ckpt, dict) and ckpt.get("format") == DEPLOY_FORMAT:
        model, meta = load_deploy_checkpoint(ckpt_path, device=device)
        cfg = {
            "img_size": int(meta.get("img_size", img_size)),
            "num_classes": int(meta.get("num_classes", num_classes)),
            "mean": np.array(meta.get("norm_mean", IMAGENET_MEAN), dtype=np.float32),
            "std": np.array(meta.get("norm_std", IMAGENET_STD), dtype=np.float32),
        }
        print(
            f"[model] deploy checkpoint : {Path(ckpt_path).name} "
            f"(epoch {meta.get('epoch')}, dice {meta.get('dice')})"
        )
        print(
            f"[model] kien truc doc tu checkpoint: img_size={cfg['img_size']} "
            f"num_classes={cfg['num_classes']}"
        )
        return model, cfg

    # --- checkpoint train thuong: phai tu khai kien truc ---
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    sd = extract_state_dict(ckpt)
    model = build_seg_model(default_model_cfg(img_size, num_classes), logger=None)
    msg = model.load_state_dict(sd, strict=False)
    missing = [k for k in msg.missing_keys if "relative_position_index" not in k]

    print(
        f"[model] checkpoint : {Path(ckpt_path).name}"
        + (f" (epoch {epoch})" if epoch is not None else "")
    )
    if missing or msg.unexpected_keys:
        print(
            f"[model] CANH BAO missing={len(missing)} "
            f"unexpected={len(msg.unexpected_keys)}"
        )
        if missing[:3]:
            print(f"          vi du missing   : {missing[:3]}")
        if list(msg.unexpected_keys)[:3]:
            print(f"          vi du unexpected: {list(msg.unexpected_keys)[:3]}")
        if len(missing) > 10:
            raise RuntimeError(
                "Checkpoint khong khop kien truc. Kiem tra --img-size / --num-classes."
            )
    else:
        print(f"[model] nap {len(sd)} tensors, khop hoan toan")

    model.eval().to(device)
    return model, {
        "img_size": img_size,
        "num_classes": num_classes,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }


# --------------------------------------------------------------------------- #
# Tien / hau xu ly
# --------------------------------------------------------------------------- #
def preprocess(path: Path, img_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Khop val_transforms. Tra ve (tensor CHW, (H_goc, W_goc))."""
    img = Image.open(path).convert("RGB")
    orig_hw = (img.height, img.width)
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)), orig_hw


@torch.no_grad()
def predict_batch(model, batch: torch.Tensor, device: str, use_amp: bool):
    batch = batch.to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        feats = model.backbone(batch)
        out = model.decode_head(feats)
    logits = out["pred"] if isinstance(out, dict) else out
    return logits.float()


def find_gt(img_path: Path, mask_dir: str | None) -> Path | None:
    """Tim mask GT: uu tien --mask-dir, khong thi doi .../image/... -> .../mask/..."""
    if mask_dir:
        p = Path(mask_dir) / img_path.name
        return p if p.exists() else None
    parts = list(img_path.parts)
    if "image" in parts:
        idx = len(parts) - 1 - parts[::-1].index("image")
        parts[idx] = "mask"
        p = Path(*parts)
        return p if p.exists() else None
    return None


def boundary(mask: np.ndarray) -> np.ndarray:
    """Vien 1px cua vung foreground (khong can scipy)."""
    m = mask > 0
    diff_v = m[:-1, :] != m[1:, :]
    diff_h = m[:, :-1] != m[:, 1:]
    edge = np.zeros_like(m)
    edge[:-1, :] |= diff_v
    edge[1:, :] |= diff_v
    edge[:, :-1] |= diff_h
    edge[:, 1:] |= diff_h
    return edge & m


def n_components(mask: np.ndarray):
    """(so vung lien thong, ty le vung lon nhat). (None, None) neu thieu scipy."""
    try:
        from scipy import ndimage
    except ImportError:
        return None, None
    lab, n = ndimage.label(mask > 0)
    if n == 0:
        return 0, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), float(sizes.max() / sizes.sum())


def seg_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred > 0, gt > 0
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    return {
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else 1.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 1.0,
        "tp_px": tp,
        "fp_px": fp,
        "fn_px": fn,
    }


# --------------------------------------------------------------------------- #
# Hinh anh
# --------------------------------------------------------------------------- #
def blend(img: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    layer = np.zeros_like(img)
    layer[:] = color
    return np.where((mask > 0)[..., None], img * (1 - alpha) + layer * alpha, img)


def make_overlay(img: np.ndarray, pred: np.ndarray, alpha=0.40) -> Image.Image:
    out = blend(img, pred, CLR_PRED, alpha)
    out[boundary(pred)] = CLR_PRED  # vien dam
    return Image.fromarray(out.astype(np.uint8))


def make_compare(img: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> Image.Image:
    """TP xanh la / FP do (thua) / FN xanh duong (sot)."""
    p, g = pred > 0, gt > 0
    out = img.astype(np.float32)
    out = blend(out, np.logical_and(p, g), CLR_TP, 0.40)
    out = blend(out, np.logical_and(p, ~g), CLR_FP, 0.55)
    out = blend(out, np.logical_and(~p, g), CLR_FN, 0.55)
    return Image.fromarray(out.astype(np.uint8))


def make_heat(prob: np.ndarray) -> Image.Image:
    """Do tin cay lop co: xanh duong (thap) -> vang -> do (cao)."""
    p = np.clip(prob, 0, 1)
    r = np.clip(2.0 * p - 0.6, 0, 1)
    g = np.clip(1.6 * p - 0.1, 0, 1) * (1 - np.clip(1.6 * p - 1.0, 0, 1) * 0.4)
    b = np.clip(1.0 - 2.2 * p, 0, 1)
    return Image.fromarray((np.stack([r, g, b], -1) * 255).astype(np.uint8))


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_panel(tiles, title: str, tile_w: int = 420) -> Image.Image:
    """Ghep cac o theo hang ngang, nhan duoi tung o + tieu de tren cung."""
    resized = []
    for label, im in tiles:
        h = max(1, round(im.height * tile_w / im.width))
        resized.append((label, im.resize((tile_w, h), Image.BILINEAR)))
    tile_h = max(im.height for _, im in resized)
    pad, bar, head = 8, 26, 30

    W = len(resized) * tile_w + (len(resized) + 1) * pad
    H = head + tile_h + bar + 2 * pad
    canvas = Image.new("RGB", (W, H), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), title, fill=(235, 235, 240), font=_font(16))

    x = pad
    for label, im in resized:
        canvas.paste(im, (x, head + pad))
        draw.text(
            (x + 4, head + pad + tile_h + 4), label, fill=(200, 200, 210), font=_font(14)
        )
        x += tile_w + pad
    return canvas


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def source_of(name: str) -> str:
    low = name.lower()
    if low.startswith("gastroc"):
        return "gastroc"
    if low.startswith("deepacsa"):
        return (
            "deepacsa_rectus"
            if "rectus" in low
            else ("deepacsa_vastus" if "vastus" in low else "deepacsa_khac")
        )
    if low.startswith("01nvb"):
        return "01NVb"
    if low.startswith("reboot"):
        return "reboot"
    return "khac"


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VALID_EXT)
    if not files:
        raise FileNotFoundError(f"Khong tim thay anh nao trong {input_path}")
    return files


def main():
    ap = argparse.ArgumentParser(description="USFM segmentation inference")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="Anh hoac thu muc anh")
    ap.add_argument("--output", default="inference_out")
    ap.add_argument(
        "--mask-dir",
        default=None,
        help="Thu muc mask GT (mac dinh: tu doi /image/ -> /mask/)",
    )
    ap.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="PHAI trung luc train (bo qua neu dung deploy checkpoint)",
    )
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-panel", action="store_true", help="Khong xuat panel tong hop")
    ap.add_argument("--no-amp", action="store_true", help="Tat bf16 autocast")
    ap.add_argument("--csv", default=None, help="Mac dinh <output>/report.csv")
    args = ap.parse_args()

    device = args.device
    use_amp = (device == "cuda") and not args.no_amp
    files = collect_inputs(Path(args.input))
    single = len(files) == 1

    out_dir = Path(args.output)
    (out_dir / "mask").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
    if not args.no_panel:
        (out_dir / "panel").mkdir(parents=True, exist_ok=True)

    model, mcfg = load_model(args.ckpt, args.img_size, args.num_classes, device)
    img_size = mcfg["img_size"]
    print(
        f"[info] {len(files)} anh | device={device} | "
        f"amp={'bf16' if use_amp else 'off'} | img_size={img_size}"
    )

    rows: list[dict] = []
    t_start = time.time()

    for start in range(0, len(files), args.batch_size):
        chunk = files[start : start + args.batch_size]
        tensors, sizes = zip(
            *(preprocess(p, img_size, mcfg["mean"], mcfg["std"]) for p in chunk)
        )

        t0 = time.time()
        logits = predict_batch(model, torch.stack(tensors), device, use_amp)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / len(chunk)

        for i, path in enumerate(chunk):
            h, w = sizes[i]
            # noi suy LOGITS ve kich thuoc goc TRUOC argmax (dung hon resize mask)
            lg = F.interpolate(
                logits[i : i + 1], size=(h, w), mode="bilinear", align_corners=False
            )
            prob = lg.softmax(dim=1)[0]
            conf_t, pred_t = prob.max(dim=0)
            pred = pred_t.cpu().numpy().astype(np.uint8)
            conf = conf_t.cpu().numpy()
            fg_prob = prob[1].cpu().numpy() if prob.shape[0] > 1 else conf

            img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            fg = pred > 0

            Image.fromarray(fg.astype(np.uint8) * 255).save(
                out_dir / "mask" / f"{path.stem}.png"
            )
            make_overlay(img, pred).save(out_dir / "overlay" / f"{path.stem}.png")

            ncomp, largest = n_components(pred)
            row = {
                "file": path.name,
                "source": source_of(path.name),
                "width": w,
                "height": h,
                "fg_px": int(fg.sum()),
                "fg_ratio": round(float(fg.mean()), 4),
                "n_components": ncomp,
                "largest_comp_ratio": None if largest is None else round(largest, 4),
                "conf_mean_fg": round(float(conf[fg].mean()), 4) if fg.any() else 0.0,
                "uncertain_ratio": round(float((conf < 0.7).mean()), 4),
                "infer_s": round(dt, 4),
            }

            gt_path = find_gt(path, args.mask_dir)
            gt = None
            if gt_path is not None:
                gt = np.asarray(Image.open(gt_path).convert("1")).astype(np.uint8)
                if gt.shape != pred.shape:
                    print(f"[canh bao] GT lech kich thuoc, bo qua: {gt_path.name}")
                    gt = None
                else:
                    row.update(seg_metrics(pred, gt))

            if not args.no_panel:
                tiles = [
                    ("anh goc", Image.fromarray(img.astype(np.uint8))),
                    ("du doan (overlay)", make_overlay(img, pred)),
                ]
                if gt is not None:
                    tiles.append(
                        ("TP xanh | FP do | FN duong", make_compare(img, pred, gt))
                    )
                tiles.append(("do tin cay lop co", make_heat(fg_prob)))
                title = (
                    f"{path.name}   |   {w}x{h}px   |   "
                    f"vung co {row['fg_ratio'] * 100:.1f}%"
                )
                if "dice" in row:
                    title += f"   |   Dice {row['dice']:.4f}  IoU {row['iou']:.4f}"
                make_panel(tiles, title).save(out_dir / "panel" / f"{path.stem}.png")

            rows.append(row)

            if single:
                print(f"\n=== {path.name} ===")
                print(f"  Nguon du lieu      : {row['source']}")
                print(f"  Kich thuoc         : {w} x {h} px")
                print(f"  Thoi gian suy luan : {dt * 1000:.0f} ms")
                print(
                    f"  Vung co            : {row['fg_px']:,} px "
                    f"({row['fg_ratio'] * 100:.2f}% anh)"
                )
                if ncomp is not None:
                    print(
                        f"  Vung lien thong    : {ncomp} "
                        f"(lon nhat chiem {largest * 100:.1f}%)"
                    )
                print(f"  Do tin cay TB (co) : {row['conf_mean_fg']:.4f}")
                print(
                    f"  Pixel khong chac   : {row['uncertain_ratio'] * 100:.2f}% "
                    f"(prob < 0.7)"
                )
                if "dice" in row:
                    print(f"  -- So voi GT: {gt_path.name} --")
                    print(
                        f"  Dice {row['dice']:.4f} | IoU {row['iou']:.4f} | "
                        f"Precision {row['precision']:.4f} | "
                        f"Recall {row['recall']:.4f}"
                    )
                    print(
                        f"  TP {row['tp_px']:,} | FP {row['fp_px']:,} (thua) "
                        f"| FN {row['fn_px']:,} (sot)"
                    )
                else:
                    print("  -- Khong tim thay mask GT, bo qua metric --")

        if not single:
            done = min(start + args.batch_size, len(files))
            print(f"\r[tien do] {done}/{len(files)}", end="", flush=True)

    elapsed = time.time() - t_start
    if not single:
        print(
            f"\n[xong] {len(files)} anh trong {elapsed:.1f}s "
            f"({len(files) / elapsed:.1f} anh/s)"
        )

    dices = [r["dice"] for r in rows if "dice" in r]
    if dices and not single:
        ious = [r["iou"] for r in rows if "iou" in r]
        print(f"[metric] Dice = {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
        print(f"[metric] IoU  = {np.mean(ious):.4f} +/- {np.std(ious):.4f}")
        groups: dict[str, list[float]] = {}
        for r in rows:
            if "dice" in r:
                groups.setdefault(r["source"], []).append(r["dice"])
        if len(groups) > 1:
            print("[metric] Theo nguon du lieu:")
            for src, vals in sorted(groups.items()):
                print(
                    f"   {src:16} n={len(vals):5}  Dice={np.mean(vals):.4f} "
                    f"+/- {np.std(vals):.4f}"
                )
        worst = sorted((r for r in rows if "dice" in r), key=lambda r: r["dice"])[:5]
        print("[metric] 5 anh te nhat (xem panel de soi loi):")
        for r in worst:
            print(f"   {r['dice']:.4f}  {r['file']}")

    csv_path = Path(args.csv) if args.csv else out_dir / "report.csv"
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[xuat] mask    : {out_dir / 'mask'}")
    print(f"[xuat] overlay : {out_dir / 'overlay'}")
    if not args.no_panel:
        print(f"[xuat] panel   : {out_dir / 'panel'}")
    print(f"[xuat] bao cao : {csv_path}")


if __name__ == "__main__":
    main()
