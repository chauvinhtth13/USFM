import datetime
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from timm.utils import AverageMeter
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

from usdsgen.data.datasets import SEG_NORM_MEAN, SEG_NORM_STD
from usdsgen.utils.file_manager import EXPORT_DIRNAME, Top_K_results_manager
from usdsgen.utils.metrics import get_seg_fromarray
from usdsgen.utils.modelutils import (
    get_grad_norm,
    save_deploy_checkpoint,
    save_pretrain_checkpoint,
)

from .basetrainer import BaseTrainer


def save_image(mask, path_dir, mask_path):
    file_name = os.path.join(path_dir, os.path.basename(mask_path))
    Image.fromarray(mask).save(file_name)


def save_seg_pre_gt(output_path, result, epoch, max_dice):
    # save the new checkpoint and folder
    mask_path_dir = os.path.join(output_path, f"best{epoch}_dice{max_dice:.3f}")

    mask_pre_path_dir = os.path.join(mask_path_dir, "mask_pre")
    mask_gt_path_dir = os.path.join(mask_path_dir, "mask_gt")
    os.makedirs(mask_path_dir, exist_ok=True)
    os.makedirs(mask_pre_path_dir, exist_ok=True)
    os.makedirs(mask_gt_path_dir, exist_ok=True)

    mask_pre_all = result["mask_pre_all"].numpy().astype(np.uint8)
    mask_path_all = result["mask_path_all"]
    mask_gt_all = result["mask_gt_all"].numpy().astype(np.uint8)
    for mask_gt, mask_pre, mask_path in zip(mask_gt_all, mask_pre_all, mask_path_all):
        save_image(mask_gt, mask_gt_path_dir, mask_path)
        save_image(mask_pre, mask_pre_path_dir, mask_path)

    return mask_path_dir


def save_segmetrics(
    output_path, val_result, epoch, max_dice, mask_path_dir, isbest=False
):
    # save segmetrics
    checkpoint_name = None
    list_dict = {
        k: (v[0].item(), v[1].item()) for k, v in val_result["segmetrics"].items()
    }
    df_metrics = pd.DataFrame.from_dict(list_dict, orient="index").reset_index()
    df_metrics.insert(0, "epoch", epoch)
    df_metrics.columns = ["epoch", "metrics", "mean", "std"]
    df_metrics.to_csv(os.path.join(output_path, "allsegmetrics.csv"), mode="a")
    if isbest:
        df_metrics.to_csv(os.path.join(mask_path_dir, "segmetrics.csv"))
        # chi giu MOT best*.pth: xoa ban cu truoc khi dat ten ban moi
        for bestpth in glob.glob(os.path.join(output_path, "best*.pth")):
            os.remove(bestpth)
        checkpoint_name = f"best{epoch}.pth"
    return checkpoint_name


class SegTrainer(BaseTrainer):
    def __init__(self, config: DictConfig) -> None:
        # base setting
        super().__init__(config)
        self.max_dice = 0.0
        self.state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,  # object, xem ghi chu basetrainer
            "max_dice": self.max_dice,
            "epoch": self.epoch,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }

        # Kien truc de nhung vao deploy/pretrain checkpoint. Bo `pretrained`:
        # do la duong dan tuyet doi tren may train, mang ckpt di may khac thi
        # tro vao file khong ton tai.
        self.deploy_model_cfg = OmegaConf.to_container(
            self.config.model.model_cfg, resolve=True
        )
        self.deploy_model_cfg.pop("pretrained", None)
        self.deploy_model_cfg.get("backbone", {}).pop("pretrained", None)
        self.top_k_results_manager = Top_K_results_manager(mode="max", max_len=5)
        self.Dice = GeneralizedDiceScore(
            num_classes=self.config.data.num_classes,
            include_background=False,
            weight_type="linear",
            input_format="index",
        ).to(self.fabric.device)
        self.IOU = MeanIoU(
            num_classes=self.config.data.num_classes,
            include_background=False,
            input_format="index",
        ).to(self.fabric.device)

    def fit(self):
        self.logger.info("Start fiting")
        self.check_resume()
        self.load_resume()
        isbest = False
        start_time = time.time()

        start_epoch = max(self.config.train.start_epoch, self.epoch)

        for epoch in range(start_epoch, self.config.train.epochs):
            self.epoch = epoch
            train_dice, train_iou, train_loss = self.train_one_epoch(
                self.dataloader_train
            )

            # train log
            tensorboard_log = {
                "loss": {"train_loss": train_loss},
                "dice": {"train_dice": train_dice},
                "iou": {"train_iou": train_iou},
            }

            # validation and save the best model
            if epoch % self.config.train.val_freq == 0:
                val_dice, val_iou, val_loss, val_result = self.validate(
                    self.dataloader_val
                )
                tensorboard_log["loss"]["val_loss"] = val_loss
                tensorboard_log["dice"]["val_dice"] = val_dice
                tensorboard_log["iou"]["val_iou"] = val_iou

                self.logger.info(
                    f"Dice of the network on all test images: {val_dice:.3f}"
                )
                self.fabric.barrier()
                self.fabric.all_reduce([train_loss, val_loss])
                self.fabric.all_reduce([train_dice, val_dice])
                self.fabric.all_reduce([train_iou, val_iou])
                val_result["segmetrics"] = self.fabric.all_reduce(
                    val_result["segmetrics"]
                )

                if val_dice > self.max_dice:
                    self.max_dice = val_dice.item()
                    isbest = True
                    self.logger.info(
                        f"Max Dice: {self.max_dice:.3f}, Max IoU: {val_iou:.3f}"
                    )
                self.save_checkpoint(epoch, self.max_dice, val_result, isbest)
                self.fabric.barrier()
                isbest = False
            # make log tensorboard
            self.fabric.log_dict(tensorboard_log, epoch)
            self.fabric.log("lr", self.optimizer.param_groups[-1]["lr"], epoch)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(f"Training time {total_time_str}")

    def train_one_epoch(self, data_loader):
        self.model.train()
        self.logger.info(
            f"Current learning rate for different parameter groups: {[it['lr'] for it in self.optimizer.param_groups]}"
        )

        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        iou_meter = AverageMeter()
        norm_meter = AverageMeter()

        start = time.time()
        end = time.time()
        accum = max(1, int(self.config.train.accumulation_steps))
        for idx, batch in enumerate(data_loader):
            loss, outputs, labels = self.step(batch)

            # Tich luy gradient: step sau moi `accum` micro-batch. Truoc day
            # vong lap step MOI batch trong khi basetrainer.make_optimizer van
            # nhan LR len accum lan -> dat accumulation_steps>1 chi lam LR sai.
            # Dieu kien idx+1 == num_steps: epoch chia khong het accum thi vai
            # micro-batch cuoi phai duoc step, khong thi gradient cua chung
            # treo lai sang epoch sau (zero_grad khong bao gio duoc goi).
            is_accumulating = (idx + 1) % accum != 0 and (idx + 1) != num_steps
            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                # Chia accum: khong chia thi gradient la TONG chu khong phai
                # trung binh, cong them LR da x accum thanh ra buoc cap nhat
                # lech accum^2 lan.
                self.fabric.backward(loss / accum)

            if not is_accumulating:
                if self.config.train.clip_grad:
                    grad_norm = self.fabric.clip_gradients(
                        self.model,
                        self.optimizer,
                        max_norm=self.config.train.clip_grad,
                        error_if_nonfinite=False,
                    )
                else:
                    grad_norm = get_grad_norm(self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step_update(self.epoch * num_steps + idx)
                norm_meter.update(grad_norm)

            loss_meter.update(loss.item(), labels.size(0))
            outputs = outputs.argmax(dim=1)
            dice_meter.update(self.Dice(outputs, labels).item())
            iou_meter.update(self.IOU(outputs, labels).item())
            batch_time.update(time.time() - end)
            end = time.time()

            # Truoc day chi log MOT lan sau ca epoch (43 phut o 768px) nen
            # khong doi chieu duoc toc do giua cac cau hinh, cung khong co ETA.
            # `print_freq` da nam trong config nhung khong duoc doc o dau ca.
            if (idx + 1) % self.config.print_freq == 0 or idx + 1 == num_steps:
                lr = self.optimizer.param_groups[-1]["lr"]
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                self.logger.info(
                    f"Train: [{self.epoch}/{self.config.train.epochs}][{idx}/{num_steps}]\t"
                    f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                    f"time {batch_time.avg:.4f} ({batch_time.val:.4f})\t"
                    f"loss {loss_meter.avg:.4f} ({loss_meter.val:.4f})\t"
                    f"DICE {dice_meter.avg:.3f} ({dice_meter.val:.3f})\t"
                    f"IOU {iou_meter.avg:.3f} ({iou_meter.val:.3f})\t"
                    f"grad_norm {norm_meter.avg:.4f} ({norm_meter.val:.4f})\t"
                    f"mem {memory_used:.0f}MB"
                )
        epoch_time = time.time() - start
        self.logger.info(
            f"EPOCH {self.epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
        )
        return dice_meter.avg, iou_meter.avg, loss_meter.avg

    @torch.no_grad()
    def validate(self, data_loader):
        self.model.eval()
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        mask_pre = []
        mask_gt = []
        mask_path = []

        end = time.time()
        for idx, batch in enumerate(data_loader):
            loss, outputs, labels = self.step(batch)
            # uint8 + CPU. Hai ly do tach roi nhau:
            # - uint8 thay int64: mask chi la chi so lop nen int64 ton gap 8
            #   lan vo ich (1217 anh o 768px: 5.7GB -> 717MB moi tensor).
            # - .cpu(): VRAM la thu dang thieu (L4 22GB, validate tung OOM
            #   dung o torch.cat), con RAM 64GB thi dang bo khong.
            # ponytail: van gom toan bo mask roi moi tinh metric mot the, vi
            # save_seg_pre_gt can ca bo. Val set rat lon hoac >255 lop thi
            # phai doi sang tinh metric tung batch va chi giu diem so.
            mask_gt.append(labels.to(torch.uint8).cpu())
            mask_pre.append(outputs.argmax(dim=1).to(torch.uint8).cpu())
            mask_path.append(batch["mask_path"])
            loss_meter.update(loss.item(), labels.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        mask_gt_all = torch.cat(mask_gt, dim=0)
        mask_pre_all = torch.cat(mask_pre, dim=0)

        mask_path_all = [item for sublist in mask_path for item in sublist]

        # Luu o RAM nhung TINH tren GPU: get_seg_fromarray lay device tu tensor
        # dau vao (metrics.py), de nguyen o CPU thi vong lap 1217 mau + HD95
        # cua monai cham hon nhieu bac. Ban uint8 nay chi ~717MB moi tensor.
        segmetrics = get_seg_fromarray(
            mask_gt_all.to(self.fabric.device), mask_pre_all.to(self.fabric.device)
        )

        dice = segmetrics["Dice"][0]
        iou = segmetrics["IoU"][0]

        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        self.logger.info(
            f"Test: [{idx}/{len(data_loader)}]\t"
            f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
            f"DICE {dice:.3f}\t"
            f"IOU {iou:.3f}"
            f"Mem {memory_used:.0f}MB"
        )

        val_result = {
            # da o CPU san, khong con .cpu() thua
            "mask_pre_all": mask_pre_all,
            "mask_path_all": mask_path_all,
            "mask_gt_all": mask_gt_all,
            "segmetrics": segmetrics,
        }

        # FIX (audit): tra ve loss trung binh, khong phai batch cuoi
        return dice, iou, loss_meter.avg, val_result

    @torch.no_grad()
    def test(self):
        if self.config.model.resume:
            self.load_resume()
        else:
            raise ValueError("No checkpoint loaded for testing")
        self.logger.info("Start testing")
        dice, iou, loss, test_result = self.validate(self.dataloader_test)
        self.logger.info(f"Test Dice: {dice:.3f}, Test IoU: {iou:.3f}")
        mask_path_dir = save_seg_pre_gt(self.config.output, test_result, "_test", dice)
        self.logger.info(f"Test result saved in {mask_path_dir}")

    def step(self, batch):
        samples, labels = batch["image"], batch["mask"]
        labels = labels.long()
        extra_features = self.model.module.backbone(samples)
        if hasattr(self.model.module.decode_head, "forward_with_loss"):
            # the [forward_with_loss] function is used for the model with custom training logic
            loss, outputs, labels = self.model.module.decode_head.forward_with_loss(
                extra_features, labels
            )
            return loss, outputs, labels
        else:
            samples, labels = batch["image"], batch["mask"]
            labels = labels.long()

            decoded = self.model.module.decode_head(extra_features)
            if hasattr(self.model.module, "auxiliary_head"):
                auxiliary_logits = self.model.module.auxiliary_head(extra_features)
                auxiliary_outputs = torch.nn.functional.interpolate(
                    auxiliary_logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                auxiliary_outputs = None

            outputs = torch.nn.functional.interpolate(
                decoded, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if auxiliary_outputs is not None:
                loss = 1 * self.criterion(outputs, labels) + 0.4 * self.criterion(
                    auxiliary_outputs, labels
                )
            else:
                loss = self.criterion(outputs, labels)
            return loss, outputs, labels

    def save_checkpoint(self, epoch, max_dice, val_result, isbest=False):
        """Ghi checkpoint. Moi lan train ket thuc se co ca `best` VA `last`.

        Ban cu chi ghi `last` khi epoch cuoi KHONG phai epoch tot nhat, nen neu
        model tot len den tan cuoi thi khong co file last nao ca.
        """
        is_last = epoch == self.config.train.epochs - 1

        with self.fabric.rank_zero_first():
            best_name = None
            if isbest:
                self.logger.info(
                    f"Saving the best model with dice {max_dice:.3f} at epoch {epoch}"
                )
                mask_path_dir = save_seg_pre_gt(
                    self.config.output, val_result, epoch, max_dice
                )
                best_name = save_segmetrics(
                    self.config.output,
                    val_result,
                    epoch,
                    max_dice,
                    mask_path_dir,
                    isbest,
                )
                self.top_k_results_manager.update(mask_path_dir, max_dice)

        self.fabric.barrier()
        best_name = self.fabric.broadcast(best_name, 0)

        self.state.update({"epoch": epoch, "max_dice": max_dice})

        # 1) checkpoint resume duoc (model + optimizer + scheduler)
        if best_name is not None:
            self.fabric.save(os.path.join(self.config.output, best_name), self.state)
            self.logger.info(f"Succeed to save checkpoint to {best_name}")
            self.export_checkpoints("best", epoch, max_dice, with_pretrain=False)

        if is_last:
            last_name = f"last{epoch}.pth"
            self.fabric.save(os.path.join(self.config.output, last_name), self.state)
            self.logger.info(f"Succeed to save checkpoint to {last_name}")
            # 2) + 3) ban deploy va ban pretrain, chi o epoch cuoi
            self.export_checkpoints("last", epoch, max_dice, with_pretrain=True)

        self.fabric.barrier()

    def export_checkpoints(self, kind, epoch, dice, with_pretrain=False):
        """Xuat ban dung ngoai vao <output>/export/.

        - `<kind>_deploy.pth`: kien truc + trong so + tham so tien xu ly ->
          inference chi can load, khong viet lai model.
        - `last_pretrain.pth`: rieng backbone, lam diem khoi dau cho lan
          finetune sau (truyen qua model.model_cfg.backbone.pretrained=...).
        """
        if self.fabric.global_rank != 0:
            return
        export_dir = os.path.join(self.config.output, EXPORT_DIRNAME)
        os.makedirs(export_dir, exist_ok=True)

        meta = {
            "epoch": epoch,
            "dice": float(dice),
            "img_size": int(self.config.data.img_size),
            "num_classes": int(self.config.data.num_classes),
            "norm_mean": list(SEG_NORM_MEAN),
            "norm_std": list(SEG_NORM_STD),
            "dataset": str(self.config.data.name),
        }

        deploy_path = os.path.join(export_dir, f"{kind}_deploy.pth")
        save_deploy_checkpoint(
            deploy_path, self.model, self.deploy_model_cfg, task="Seg", meta=meta
        )
        self.logger.info(f"Exported deploy checkpoint -> {deploy_path}")

        if with_pretrain:
            pretrain_path = os.path.join(export_dir, f"{kind}_pretrain.pth")
            save_pretrain_checkpoint(
                pretrain_path,
                self.model,
                self.deploy_model_cfg["backbone"],
                meta=meta,
            )
            self.logger.info(f"Exported pretrain checkpoint -> {pretrain_path}")


if __name__ == "__main__":
    SegTrainer()
