from PIL import Image
import numpy as np
import csv
import pandas as pd
import pickle

from network.model_vgg16_anysize import Model_BRS
from vis import save_visualization


def load_Model_BRS(state_dict, device):
    net_weight = state_dict['state_dict']
    model = Model_BRS()

    if state_dict['epoch'] < 20:
        model.withFD = False
    else:
        model.withFD = True

    model.load_state_dict(net_weight, strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def make_result_file(eval_exp_path, resume):
    csv_filename = eval_exp_path / 'results.csv'
    header = ['img_name', 'maxIoU', 'maxClick', 'time']

    if not resume:
        with open(csv_filename, 'a', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
        resume_index = 0
    else:
        df = pd.read_csv(csv_filename, header=None)
        resume_index = df.index.shape[0] - 1

    return csv_filename, header, resume_index


def save_sample_result(sample, results, save_dir):
    img_name = sample['meta']['img_name']
    clicks, sample_ious, pred_mask, iact, sample_time, maxIoU, maxClick = results

    # Save IoU Result
    np.savetxt(save_dir / f'{img_name}_IoU.csv', sample_ious, delimiter=',', fmt='%.5e')

    # Save Clicks
    clicks_res = save_dir / f'{img_name}_clicks.pkl'
    with open(clicks_res, "wb") as fp:
        pickle.dump(clicks, fp)

    # Visualize results
    draw = save_visualization(sample['image'], pred_mask, clicks)
    draw = np.concatenate((draw,
                           255 * pred_mask[:, :, np.newaxis].repeat(3, axis=2),
                           255 * (sample['instances_mask'] > 0)[:, :, np.newaxis].repeat(3, axis=2)
                           ), axis=1)
    draw_PIL = Image.fromarray(draw.astype('uint8'))
    draw_PIL.save(save_dir / f'{img_name}_results.png')

    # Visualize interaction map
    piact_PIL = Image.fromarray((np.clip(iact[0, 0], 0, 1) * 255).astype('uint8'))
    niact_PIL = Image.fromarray((np.clip(iact[0, 1], 0, 1) * 255).astype('uint8'))

    piact_PIL.save(save_dir / f'{img_name}_lastclick_piact.png')
    niact_PIL.save(save_dir / f'{img_name}_lastclick_niact.png')

    # Save results as csv file
    data = [{'img_name': img_name,
             'maxIoU': maxIoU,
             'maxClick': maxClick,
             'time': sample_time}]

    return data

