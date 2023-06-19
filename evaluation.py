import argparse
import csv
import torch
import numpy as np
from tqdm import tqdm
from time import time
from pathlib import Path
import torchvision

import utils
from metrics import compute_iou
from predictor import BasePredictor, BRSPredictor
from datasets.grabcut import GrabCutdataset
from datasets.berkeley import Berkeleydataset


def evaluate_dataset(dataset, predictor, save_dir, pred_thr, target_iou, resume):
    save_results = True
    # Make save directory
    if save_results:
        csv_filename, header, resume_id = utils.make_result_file(save_dir, resume)

    for index in tqdm(range(resume_id, len(dataset)), total=len(dataset), initial=resume_id, leave=False):
        sample = dataset[index]
        img_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ])
        image_nd = img_transform(sample['image'])

        # Set maximum number of clicks
        max_clicks = (sample['instances_mask'] > 0).sum()

        # Evaluate Sample
        sample_results = evaluate_sample(sample, image_nd, predictor, max_clicks, pred_thr, target_iou, save_dir)

        # Save Results
        if save_results:
            data = utils.save_sample_result(sample, sample_results, save_dir)
            with open(csv_filename, 'a', encoding='UTF8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writerows(data)


def evaluate_sample(sample, image_nd, predictor, max_clicks, pred_thr, max_iou_thr, save_dir, check_convergence=True):
    start_time = time()
    # Initialize prediction mask
    pred_mask = np.zeros_like(sample['instances_mask'])

    ious_list = []
    with torch.no_grad():
        # Set for sample
        predictor.set_input_image(image_nd, sample['instances_mask'], sample['meta']['img_name'], save_dir)

        max_iou = 0.0
        sameIoU_count = 0
        for click_id in range(max_clicks):
            pred_probs, iact = predictor.get_prediction(pred_mask)
            pred_mask = pred_probs > pred_thr

            iou = compute_iou(sample['instances_mask'], pred_mask)
            ious_list.append(iou)
            print(f'Click number {click_id+1} - IoU score: {iou}')

            # Check for maximum iou score
            if iou > max_iou:
                max_iou, max_id = iou, click_id + 1
                print(f'Max IoU score: {iou} at click {max_id}')

            # Stop when target IoU score is achieved
            if iou >= max_iou_thr:
                break

            # Convergence condition
            if check_convergence:
                if click_id >= 1:
                    if iou == ious_list[-2]:
                        sameIoU_count += 1
                    else:
                        sameIoU_count = 0

                    if sameIoU_count > 100:
                        break

    end_time = time()
    elapsed_time = end_time - start_time

    return predictor.click_list, np.array(ious_list, dtype=np.float32), pred_mask, iact, elapsed_time, max_iou, max_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='Try0')
    parser.add_argument('--exp_dir', type=str, default='./experiments')
    parser.add_argument('--dataset', type=str, default='GrabCut')
    parser.add_argument('--ckpt_path', type=str, default='vgg16_CD_model.pth')
    parser.add_argument('--mode', type=str, default='BRS')
    parser.add_argument('--pred_thr', type=float, default=0.5)
    parser.add_argument('--targ_iou', type=float, default=1.0)
    parser.add_argument('--resume', type=bool, default=False)
    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'GrabCut':
        dataset = GrabCutdataset(root='./datasets/GrabCut')
    elif args.dataset == 'Berkeley':
        dataset = Berkeleydataset(root='./datasets/Berkeley')

    # Load model
    ckpt = torch.load(args.ckpt_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = utils.load_Model_BRS(ckpt, device)

    if args.mode == 'NoBRS':
        predictor = BasePredictor(net, device)
    elif args.mode == 'BRS':
        predictor = BRSPredictor(net, device)

    exp_path = Path(args.exp_dir) / f'{args.exp_name}' / f'{args.dataset}' / f'{args.mode}'
    exp_path.mkdir(parents=True, exist_ok=True)

    evaluate_dataset(dataset, predictor, exp_path, pred_thr=args.pred_thr, target_iou=args.targ_iou, resume=args.resume)