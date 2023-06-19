import numpy as np
from scipy.ndimage.morphology import distance_transform_edt


def generate_click(gt_mask, pred_mask, click_map):
    not_ignore_mask = gt_mask != -1
    gt_mask = gt_mask == 1

    fn_map = (gt_mask & (~pred_mask)) & not_ignore_mask
    fp_map = ((~gt_mask) & pred_mask) & not_ignore_mask

    fn_map = np.pad(fn_map, ((1, 1), (1, 1)), 'constant')
    fndist_map = distance_transform_edt(fn_map)
    fndist_map = fndist_map[1:-1, 1:-1]
    fndist_map = np.multiply(fndist_map, 1 - click_map)

    fp_map = np.pad(fp_map, ((1, 1), (1, 1)), 'constant')
    fpdist_map = distance_transform_edt(fp_map)
    fpdist_map = fpdist_map[1:-1, 1:-1]
    fpdist_map = np.multiply(fpdist_map, 1 - click_map)

    if np.max(fndist_map) > np.max(fpdist_map):
        is_pos = 1
        usr_map = fndist_map
    else:
        is_pos = 0
        usr_map = fpdist_map

    y_mlist, x_mlist = np.where(usr_map == np.max(usr_map))
    yx_click = [y_mlist[0], x_mlist[0]]

    return is_pos, yx_click

