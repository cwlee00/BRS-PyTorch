import numpy as np


def compute_iou(gt_mask, pred_mask):
    ignore_gt_mask = gt_mask != -1
    gt_mask = gt_mask == 1

    int_section = ((gt_mask & pred_mask) & ignore_gt_mask)
    uni_on = ((gt_mask | pred_mask) & ignore_gt_mask)
    iou_score = np.sum(int_section) / np.sum(uni_on)

    return iou_score
