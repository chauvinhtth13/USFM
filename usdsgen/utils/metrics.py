import torch
from monai.metrics import compute_hausdorff_distance


def dice_coefficient(pred, gt, smooth=1e-5):
    """dice = 2TP/(FP + 2TP + FN)

    LUU Y: nhi phan hoa bang so sanh (tao tensor moi), KHONG ghi de tai cho.
    Ban cu lam `pred[pred >= 1] = 1` tren mot *view* cua mask_pre_all, nen mask
    luu ra dia bi nhi phan hoa theo — sai voi bai toan nhieu hon 2 lop.
    """
    N = gt.shape[0]
    pred_flat = (pred >= 1).reshape(N, -1).float()
    gt_flat = (gt >= 1).reshape(N, -1).float()
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + smooth) / (unionset + smooth)
    return dice.sum() / N


def sespiou_coefficient2(pred, gt, all=False, smooth=1e-5):
    """sensitivity = TP/(TP+FN), specificity = TN/(FP+TN), iou = TP/(FP+TP+FN)"""
    N = gt.shape[0]
    pred_flat = (pred >= 1).reshape(N, -1).float()
    gt_flat = (gt >= 1).reshape(N, -1).float()
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    TN = ((1 - pred_flat) * (1 - gt_flat)).sum(1)
    FP = pred_flat.sum(1) - TP
    SE = (TP + smooth) / (TP + FN + smooth)
    SP = (TN + smooth) / (FP + TN + smooth)
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    Acc = (TP + TN + smooth) / (TP + FP + FN + TN + smooth)
    Precision = (TP + smooth) / (TP + FP + smooth)
    Recall = (TP + smooth) / (TP + FN + smooth)
    F1 = 2 * Precision * Recall / (Recall + Precision + smooth)
    if all:
        return (
            SE.sum() / N,
            SP.sum() / N,
            IOU.sum() / N,
            Acc.sum() / N,
            F1.sum() / N,
            Precision.sum() / N,
            Recall.sum() / N,
        )
    return IOU.sum() / N, Acc.sum() / N, SE.sum() / N, SP.sum() / N


def get_seg_fromarray(GT_array, Pre_array):
    device = GT_array.device
    B, H, W = GT_array.shape
    dice_scores = torch.zeros(B, device=device)
    hd95_scores = torch.zeros(B, device=device)
    iou_scores = torch.zeros(B, device=device)
    accuracy_scores = torch.zeros(B, device=device)
    sensitivity_scores = torch.zeros(B, device=device)
    specificity_scores = torch.zeros(B, device=device)
    diag = torch.tensor((H**2 + W**2) ** 0.5, device=device)

    for i in range(B):
        pred = Pre_array[i : i + 1, :, :]
        gt = GT_array[i : i + 1, :, :]

        dice_scores[i] = dice_coefficient(pred, gt)
        # HD95 bang monai thay package `hausdorff` (khong bao tri, khong co
        # wheel py3.14). LUU Y: so KHAC baseline cu (HD95 vs HD tho, euclid vs
        # manhattan) — thay doi co chu y, chuan hon trong y van.
        pred_1 = (pred[0] >= 1).float()
        gt_1 = (gt[0] >= 1).float()
        if pred_1.any() and gt_1.any():
            hd95_scores[i] = (
                compute_hausdorff_distance(
                    pred_1[None, None], gt_1[None, None], percentile=95
                )
                .squeeze()
                .to(device)
            )
        else:
            # mot trong hai mask rong: HD khong xac dinh, quy uoc duong cheo anh
            hd95_scores[i] = diag
        iou, acc, se, sp = sespiou_coefficient2(pred, gt, all=False)
        iou_scores[i] = iou
        accuracy_scores[i] = acc
        sensitivity_scores[i] = se
        specificity_scores[i] = sp

    return {
        "Dice": (torch.mean(dice_scores), torch.std(dice_scores)),
        "Hausdorff": (torch.mean(hd95_scores), torch.std(hd95_scores)),
        "IoU": (torch.mean(iou_scores), torch.std(iou_scores)),
        "Accuracy": (torch.mean(accuracy_scores), torch.std(accuracy_scores)),
        "Sensitivity": (torch.mean(sensitivity_scores), torch.std(sensitivity_scores)),
        "Specificity": (torch.mean(specificity_scores), torch.std(specificity_scores)),
    }
