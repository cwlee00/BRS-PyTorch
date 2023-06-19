import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def create_meshgrid(image_size):
    y_linspace = np.linspace(0, image_size[0] - 1, image_size[0])
    x_linspace = np.linspace(0, image_size[1] - 1, image_size[1])
    x_meshgrid, y_meshgrid = np.meshgrid(x_linspace, y_linspace)

    return x_meshgrid, y_meshgrid


def save_visualization(in_img, pred_mask, yx_click):
    img_vis = in_img.copy()
    x_meshgrid, y_meshgrid = create_meshgrid(pred_mask.shape)

    r_img_vis = in_img[:, :, 0].copy()
    g_img_vis = in_img[:, :, 1].copy()
    b_img_vis = in_img[:, :, 2].copy()

    blend_ratio = 0.5

    r_img_vis[pred_mask] = (1 - blend_ratio) * r_img_vis[pred_mask] + blend_ratio * 0
    g_img_vis[pred_mask] = (1 - blend_ratio) * g_img_vis[pred_mask] + blend_ratio * 200
    b_img_vis[pred_mask] = (1 - blend_ratio) * b_img_vis[pred_mask] + blend_ratio * 200

    for yx_inst in yx_click:
        dist_map = np.sqrt(pow(y_meshgrid - yx_inst[0], 2) + pow(x_meshgrid - yx_inst[1], 2))
        outer_map = dist_map < 5
        inner_map = dist_map < 4.5

        r_img_vis[outer_map] = 255
        g_img_vis[outer_map] = 255
        b_img_vis[outer_map] = 255

        if yx_inst[2] == 1:
            r_img_vis[inner_map] = 191
            g_img_vis[inner_map] = 42
            b_img_vis[inner_map] = 42
        else:
            r_img_vis[inner_map] = 9
            g_img_vis[inner_map] = 33
            b_img_vis[inner_map] = 64

    img_vis[:, :, 0] = r_img_vis
    img_vis[:, :, 1] = g_img_vis
    img_vis[:, :, 2] = b_img_vis

    # img_vis_PIL = Image.fromarray(img_vis.astype('uint8'))

    return img_vis


def plot_ious(ious_noBRS, ious_withBRS, save_res_dir, sample_name, max_clicks):
    x = np.arange(1, max_clicks+1)
    total_area = metrics.auc(x, np.ones(max_clicks))
    auc_noBRS = metrics.auc(x, ious_noBRS)
    auc_withBRS = metrics.auc(x, ious_withBRS)
    auc_normalized_noBRS = auc_noBRS / total_area
    auc_normalized_withBRS = auc_withBRS / total_area

    plt.figure()
    plt.plot(x, ious_noBRS, label=f'NoBRS [{auc_normalized_noBRS:.3f}]')
    plt.plot(x, ious_withBRS, label=f'withBRS [{auc_normalized_withBRS:.3f}]')
    plt.xlabel('Number of Clicks')
    plt.ylabel('IoU score')
    plt.legend(loc=4)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.show()
    plt.savefig(os.path.join(save_res_dir, f'iou_{sample_name}.png'))