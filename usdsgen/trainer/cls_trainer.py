import datetime
import os
import pickle
import time

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from timm.utils import AverageMeter

from usdsgen.utils.file_manager import EXPORT_DIRNAME
from usdsgen.utils.logger import array_to_markdown
from usdsgen.utils.modelutils import (
    get_grad_norm,
    save_deploy_checkpoint,
    save_pretrain_checkpoint,
)

from .basetrainer import BaseTrainer


class ClsTrainer(BaseTrainer):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.max_accuracy = 0.0
        # task specific setting
        self.state = {
            "model": self.model,
            "optimizer": self.optimizer,
            # object, khong phai snapshot — xem ghi chu basetrainer
            "lr_scheduler": self.lr_scheduler,
            "max_accuracy": self.max_accuracy,
            # scaler da xoa khoi BaseTrainer: Fabric tu quan ly noi bo
            "epoch": self.epoch,
            # plain dict de load duoc voi torch>=2.6 (weights_only=True default)
            "config": OmegaConf.to_container(self.config, resolve=True),
        }

        # Kien truc de nhung vao deploy/pretrain checkpoint; bo `pretrained`
        # vi do la duong dan tuyet doi tren may train.
        self.deploy_model_cfg = OmegaConf.to_container(
            self.config.model.model_cfg, resolve=True
        )
        self.deploy_model_cfg.pop("pretrained", None)
        if isinstance(self.deploy_model_cfg.get("backbone"), dict):
            self.deploy_model_cfg["backbone"].pop("pretrained", None)

        # Verify if the num of classes in training datafolder is same to the config
        assert len(self.dataloader_val.dataset.classes) == self.config.data.num_classes, (
            "The num of classes in training datafolder is not same to the config."
        )
        self.logger.info(f"num of classes: {len(self.dataloader_val.dataset.classes)}")
        self.logger.info(f"class_to_index: {self.dataloader_val.dataset.class_to_idx}")
        # cm row name setting
        self.row_name = list(self.dataloader_val.dataset.class_to_idx.keys())
        # fabric setting datasets & use_distributed_sampler

    def fit(self):
        self.logger.info("Start fiting")
        self.check_resume()
        self.load_resume()

        isbest = False
        start_time = time.time()

        start_epoch = max(self.config.train.start_epoch, self.epoch)
        loginfo = None  # co the chua validate lan nao khi toi epoch cuoi
        for epoch in range(start_epoch, self.config.train.epochs):
            self.epoch = epoch
            isbest = False
            train_acc, _, train_loss = self.train_one_epoch(self.dataloader_train)
            tensorboard_log = {
                "loss": {
                    "train_loss": train_loss,
                },
                "acc": {
                    "train_acc": train_acc,
                },
            }

            # validation and save the best model
            if epoch % self.config.train.val_freq == 0:
                val_acc, _, val_loss, loginfo = self.validate(self.dataloader_val)
                tensorboard_log["loss"]["val_loss"] = val_loss
                tensorboard_log["acc"]["val_acc"] = val_acc
                if val_acc > self.max_accuracy:
                    self.max_accuracy = val_acc
                    isbest = True
                    self.logger.info(f"Max accuracy: {self.max_accuracy:.3f}")

            is_last = epoch == self.config.train.epochs - 1
            if (isbest or is_last) and loginfo is not None:
                self.save_checkpoint(
                    epoch, self.max_accuracy, loginfo, isbest=isbest, is_last=is_last
                )

            # make log tensorboard
            self.fabric.log_dict(tensorboard_log, epoch)
            self.fabric.log("lr", self.optimizer.param_groups[-1]["lr"], epoch)

        total_time = time.time() - start_time
        self.logger.info(f"Training time {datetime.timedelta(seconds=int(total_time))!s}")

    def train_one_epoch(self, data_loader):
        self.model.train()
        # self.logger.info(
        #     f'Current learning rate for different parameter groups: {[round(it["lr"], 2) for it in self.optimizer.param_groups]}'
        # )

        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        y_t = []
        y_p = []
        norm_meter = AverageMeter()

        start = time.time()
        end = time.time()
        accum = max(1, int(self.config.train.accumulation_steps))
        for idx, (samples, labels) in enumerate(data_loader):
            labels = labels.long()
            outputs = self.model(samples)

            # Tich luy gradient: step sau moi `accum` micro-batch.
            # Ban cu dung `idx % accum != 0` -> step ngay tai idx=0 (chua tich
            # luy gi) va crash ZeroDivisionError khi accum=0.
            is_accumulating = (idx + 1) % accum != 0
            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                loss = self.criterion(outputs, labels)
                # .backward() accumulates when .zero_grad() wasn't called
                self.fabric.backward(loss)

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
                if "plateau" not in type(self.lr_scheduler).__name__.lower():
                    self.lr_scheduler.step_update(self.epoch * num_steps + idx)
                # chi ghi nhan o buoc that su step (ban cu update 2 lan, va
                # doc `grad_norm` chua gan o micro-batch dau tien)
                norm_meter.update(grad_norm)

            loss_meter.update(loss.item(), labels.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            y_t.append(labels.detach())
            y_p.append(outputs.argmax(1).detach())

        y_t = torch.cat(y_t, dim=0)
        y_p = torch.cat(y_p, dim=0)
        y_t_all = self.fabric.all_gather(y_t).reshape(-1).cpu().numpy()
        y_p_all = self.fabric.all_gather(y_p).reshape(-1).cpu().numpy()

        cm = confusion_matrix(y_t_all, y_p_all)
        acc = balanced_accuracy_score(y_t_all, y_p_all)

        if "plateau" in type(self.lr_scheduler).__name__.lower():
            self.lr_scheduler.step(self.epoch, acc)

        lr = self.optimizer.param_groups[-1]["lr"]
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        etas = batch_time.avg * (num_steps - idx)
        self.logger.info(
            f"Train: [{self.epoch}/{self.config.train.epochs}]\t"
            f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
            f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
            f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
            f"acc {acc:2f}\t"
            f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
            f"mem {memory_used:.0f}MB"
        )
        epoch_time = time.time() - start
        self.logger.info(
            f"EPOCH {self.epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
        )

        self.writer.add_text(
            "cm/train",
            f"#train\n{array_to_markdown(cm, self.row_name, self.row_name)}\n\n",
            self.epoch,
        )
        return acc, cm, loss

    @torch.no_grad()
    def validate(self, data_loader):
        criterion = torch.nn.CrossEntropyLoss()
        self.model.eval()

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        y_t = []
        outputs_p = []
        y_p = []
        All_feature = []

        end = time.time()
        for idx, (samples, labels) in enumerate(data_loader):
            labels = labels.long()
            feature = self.model.module.forward_features(samples).detach()
            outputs = self.model(samples)
            All_feature.append(feature.detach())
            loss = criterion(outputs, labels)

            loss_meter.update(loss.item(), labels.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            y_t.append(labels.detach())
            outputs_p.append(outputs.detach())
            y_p.append(outputs.argmax(1).detach())

        y_t = torch.cat(y_t, dim=0)
        outputs_p = torch.cat(outputs_p, dim=0)
        y_p = torch.cat(y_p, dim=0)
        all_feature = torch.cat(All_feature, dim=0)

        y_t_all = self.fabric.all_gather(y_t).reshape(-1).cpu().numpy()
        y_p_all = self.fabric.all_gather(y_p).reshape(-1).cpu().numpy()
        outputs_p_all = self.fabric.all_gather(outputs_p).reshape(-1).cpu().numpy()
        all_feature = self.fabric.all_gather(all_feature).flatten(0, 1).cpu().numpy()

        cm = confusion_matrix(y_t_all, y_p_all)
        acc = balanced_accuracy_score(y_t_all, y_p_all)
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        self.logger.info(
            f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
            f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
            f"acc {acc:2f}\t"
            f"mem {memory_used:.0f}MB"
        )

        loginfo = {
            "y_t": y_t_all,
            "y_p": y_p_all,
            "outputs": outputs_p_all,
            "all_feature": all_feature,
        }

        self.writer.add_text(
            "cm/train",
            f"#test\n{array_to_markdown(cm, self.row_name, self.row_name)}",
            self.epoch,
        )
        return acc, cm, loss, loginfo

    @torch.no_grad()
    def test(self):
        if self.config.model.resume:
            self.load_resume()
        else:
            raise ValueError("No checkpoint loaded for testing")
        self.load_resume()
        self.logger.info("Start testing")
        acc, cm, loss, loginfo = self.validate(self.dataloader_test)
        self.logger.info(f"bAccuracy of the network on the test images: {acc:.3f}")
        self.logger.info(f"Confusion matrix: \n{cm}")
        self.logger.info(f"Loss: {loss:.3f}")
        np.savetxt(
            os.path.join(self.config.output, "prediction_result.csv"),
            np.concatenate(
                [loginfo["y_t"].reshape(-1, 1), loginfo["y_p"].reshape(-1, 1)], axis=1
            ),
            delimiter=",",
            fmt="%d",
        )

    def save_checkpoint(self, epoch, max_accuracy, loginfo, isbest=False, is_last=False):
        """Ghi ca `best_ckpt.pth` va `last_ckpt.pth`, kem ban deploy/pretrain."""
        self.fabric.barrier()
        self.state.update({"epoch": epoch, "max_accuracy": max_accuracy})

        if isbest:
            self.fabric.save(
                os.path.join(self.config.output, "best_ckpt.pth"), self.state
            )
            self.export_checkpoints("best", epoch, max_accuracy, with_pretrain=False)
        if is_last:
            self.fabric.save(
                os.path.join(self.config.output, "last_ckpt.pth"), self.state
            )
            self.export_checkpoints("last", epoch, max_accuracy, with_pretrain=True)

        if self.fabric.global_rank == 0:
            plot_path_dir = os.path.join(
                self.config.output, f"best{epoch}_acc{max_accuracy:.3f}"
            )
            os.makedirs(plot_path_dir, exist_ok=True)
            with open(os.path.join(plot_path_dir, "loginfo.pkl"), "wb") as f:
                pickle.dump(loginfo, f)
        self.fabric.barrier()

    def export_checkpoints(self, kind, epoch, acc, with_pretrain=False):
        """Xuat ban dung ngoai vao <output>/export/ — xem SegTrainer.export_checkpoints."""
        if self.fabric.global_rank != 0:
            return
        export_dir = os.path.join(self.config.output, EXPORT_DIRNAME)
        os.makedirs(export_dir, exist_ok=True)

        meta = {
            "epoch": epoch,
            "accuracy": float(acc),
            "img_size": int(self.config.data.img_size),
            "num_classes": int(self.config.data.num_classes),
            "class_names": list(self.row_name),
            "dataset": str(self.config.data.name),
        }

        deploy_path = os.path.join(export_dir, f"{kind}_deploy.pth")
        save_deploy_checkpoint(
            deploy_path, self.model, self.deploy_model_cfg, task="Cls", meta=meta
        )
        self.logger.info(f"Exported deploy checkpoint -> {deploy_path}")

        if with_pretrain:
            pretrain_path = os.path.join(export_dir, f"{kind}_pretrain.pth")
            save_pretrain_checkpoint(
                pretrain_path, self.model, self.deploy_model_cfg, meta=meta
            )
            self.logger.info(f"Exported pretrain checkpoint -> {pretrain_path}")


if __name__ == "__main__":
    ClsTrainer()
