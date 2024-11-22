"""Script to evalute performance on the val and test set."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import os
import sys

parent_dir = os.path.abspath('')
sys.path.append(parent_dir)

from utils.io import load_json, write_json
from utils.bboxes import box_xyxyzz_to_cxcyczwhd
from utils.visualization import save_pred_visualization
from data.dataloader import get_loader
from evaluator import DetectionEvaluator
from models.retinanet.retina_unet import RetinaUNet
from models.retinanet.mamba_detection import mambanet

from transoar.utils.io import get_config
import monai
import random
from torch.cuda.amp import GradScaler, autocast

class Tester:

    def __init__(self, args):
        path_to_run = Path('' + args.run)
        config = load_json(path_to_run / 'config.json')

        self._save_preds = args.save_preds
        self._full_labeled = args.full_labeled
        self._class_dict = config['labels']
        self._device = 'cuda' if args.num_gpu >= 0 else 'cpu'

        # Get path to checkpoint
        avail_checkpoints = [path for path in path_to_run.iterdir() if 'model_' in str(path)]
        avail_checkpoints.sort(key=lambda x: len(str(x)))
        if args.last:
            path_to_ckpt = avail_checkpoints[0]
        else:
            path_to_ckpt = avail_checkpoints[-1]

        # Build necessary components
        self._set_to_eval = 'val' if args.val else 'test'
        self._test_loader = get_loader(config, self._set_to_eval, batch_size=1)

        self._evaluator = DetectionEvaluator(
            classes=list(config['labels'].values()),
            classes_small=config['labels_small'],
            classes_mid=config['labels_mid'],
            classes_large=config['labels_large'],
            iou_range_nndet=(0.1, 0.5, 0.05),
            iou_range_coco=(0.5, 0.95, 0.05),
            sparse_results=False
        )
        # self._model = RetinaUNet(config).to(device=self._device)
        self._model = mambanet(config).to(device=self._device)

        # # Load checkpoint
        # checkpoint = torch.load(path_to_ckpt, map_location=self._device)
        # self._model.load_state_dict(checkpoint['model_state_dict'])
        # self._model.eval()

        # Load checkpoint
        checkpoint = torch.load(path_to_ckpt, map_location=self._device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k,v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self._model.load_state_dict(new_state_dict)
        self._model.eval()

        # Create dir to store results
        self._path_to_results = path_to_run / 'results' / path_to_ckpt.parts[-1][:-3]
        self._path_to_results.mkdir(parents=True, exist_ok=True)

        if self._save_preds:
            self._path_to_vis = self._path_to_results / ('vis_' + self._set_to_eval)
            self._path_to_vis.mkdir(parents=False, exist_ok=True)
  
    def run(self):
        with torch.no_grad():
            for idx, (data, _, bboxes, seg_mask, data_name) in enumerate(tqdm(self._test_loader)):
                # Put data to gpu
                # print('processing '+data_name)
                data = data.to(device=self._device)

                targets = defaultdict(list)
                for item in bboxes:
                    targets['target_boxes'].append(item[0].to(dtype=torch.float, device=self._device))
                    targets['target_classes'].append(item[1].to(device=self._device))
                targets['target_seg'] = seg_mask.squeeze(1).to(device=self._device)

                # # Only use complete data for performance evaluation
                # if self._full_labeled:
                #     if targets['target_classes'][0].shape[0] < len(self._class_dict):
                #         continue

                # Make prediction
                with autocast():
                    _, predictions = self._model.train_step(data, targets, evaluation=True)

                pred_boxes = predictions['pred_boxes'][0].detach().cpu().numpy()
                pred_classes = predictions['pred_labels'][0].detach().cpu().numpy()
                pred_scores = predictions['pred_scores'][0].detach().cpu().numpy()
                gt_boxes = targets['target_boxes'][0].detach().cpu().numpy()
                gt_classes = targets['target_classes'][0].detach().cpu().numpy()

                # Evaluate validation predictions based on metric
                self._evaluator.add(
                    pred_boxes=[pred_boxes],
                    pred_classes=[pred_classes],
                    pred_scores=[pred_scores],
                    gt_boxes=[gt_boxes],
                    gt_classes=[gt_classes],
                )
                # print(pred_boxes)
                print(pred_classes)
                # print(pred_scores)
                # print(gt_boxes)
                print(gt_classes)

                # # Just take most confident prediction per class
                best_ids = []
                for class_ in np.unique(pred_classes):
                    class_ids = (pred_classes == class_).nonzero()[0]
                    max_scoring_idx = pred_scores[class_ids].argmax()
                    best_ids.append(class_ids[max_scoring_idx])
                best_ids = torch.tensor(best_ids)

                if self._save_preds:
                    save_pred_visualization(
                        box_xyxyzz_to_cxcyczwhd(pred_boxes[best_ids], data.shape[2:]), pred_classes[best_ids] + 1,
                        box_xyxyzz_to_cxcyczwhd(gt_boxes, data.shape[2:]), gt_classes + 1,
                        seg_mask[0], self._path_to_vis, self._class_dict, data_name
                    )

                # if self._save_preds:
                #     save_pred_visualization(
                #         # box_xyxyzz_to_cxcyczwhd(pred_boxes[best_ids], data.shape[2:]), pred_classes[best_ids] + 1,
                #         box_xyxyzz_to_cxcyczwhd(pred_boxes, data.shape[2:]), pred_classes + 1,
                #         box_xyxyzz_to_cxcyczwhd(gt_boxes, data.shape[2:]), gt_classes + 1,
                #         seg_mask[0], self._path_to_vis, self._class_dict, data_name
                #     )
                #     # print(data_name)

            # Get and store final results
            metric_scores = self._evaluator.eval()
            write_json(metric_scores, self._path_to_results / ('results_' + self._set_to_eval))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    # Add necessary args
    parser.add_argument('--run', default='prior_retina_unet_amos', type=str, help='Name of experiment in transoar/runs.')
    parser.add_argument('--num_gpu', type=int, default=1, help='gpu or cpu.')
    parser.add_argument('--val', action='store_true', help='Evaluate performance on test set.')
    parser.add_argument('--last', action='store_true', help='Use model_last instead of model_best.')
    parser.add_argument('--save_preds', action='store_true', default='True',help='Save predictions.')
    parser.add_argument('--full_labeled', action='store_true',help='Use only fully labeled data.')
    parser.add_argument('--coco_map', action='store_true', default='True', help='Use coco map.')
    parser.add_argument("--config", type=str, default='attn_fpn_retina_unet_amos', help="Config to use for training located in /config.")
    args = parser.parse_args()

    # Get relevant configs
    config = get_config(args.config)
    # To get reproducable results
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    monai.utils.set_determinism(seed=config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.benchmark = True # performance vs. reproducibility
    torch.backends.cudnn.deterministic = True

    tester = Tester(args)
    tester.run()
