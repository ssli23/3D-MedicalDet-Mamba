"""Script for training the transoar project."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import monai
import sys


import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


parent_dir = os.path.abspath('/home/kb535/lss/codes/objection/transoar')
sys.path.append(parent_dir)

from transoar.trainer2 import Trainer2
from transoar.data.dataloader import get_loader,modified_get_loader
from transoar.utils.io import get_config, write_json, get_meta_data
# from transoar.models.retinanet.retina_unet import RetinaUNet
from transoar.models.retinanet.mamba_detection import mambanet


def match(n, keywords):
        out = False
        for b in keywords:
            if b in n:
                out = True
                break
        return out

def train(config, args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')

    # os.environ["CUDA_VISIBLE_DEVICES"] = config['device'][-1]
    device = torch.device(f'cuda:{args.local_rank}')

    # device = 'cuda' # TODO: fix this hack for def detr cuda module

    # Build necessary components
    # train_loader = get_loader(config, 'train')
    train_loader = modified_get_loader(config, 'train')

    if config['overfit']:
        val_loader = get_loader(config, 'train')
    else:
        val_loader = get_loader(config, 'val', batch_size=1)

    # model = RetinaUNet(config)
    model = mambanet(config)

    # # # init
    # checkpoint = torch.load('/home/kb535/lss/codes/objection/transoar/runs/model_best_0.169.pt')
    # # Unpack and load content
    # model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device=device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)

    # Analysis of model parameter distribution
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_backbone_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and match(n, ['attn_fpn']))
    num_head_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and match(n, ['head', 'segmenter']))
    num_head_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and match(n, ['head', 'segmenter']))

    print(num_params)
    print(f'num_head_params\t\t{num_head_params:>10}\t{num_head_params/num_params:.4f}%') # TODO: Incorp into logging
    print(f'num_backbone_params\t{num_backbone_params:>10}\t{num_backbone_params/num_params:.4f}%')

    param_dicts = [
        {
            'params': [
                p for n, p in model.named_parameters() if not match(n, ['attn_fpn']) and p.requires_grad
            ],
            'lr': float(config['lr'])
        },
        {
            'params': [p for n, p in model.named_parameters() if match(n, ['attn_fpn']) and p.requires_grad],
            'lr': float(config['lr_backbone'])
        } 
    ]


    optim = torch.optim.AdamW(
        param_dicts, lr=float(config['lr']), weight_decay=float(config['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, config['lr_drop'])

    # Load checkpoint if applicable
    if args.resume is not None:
        checkpoint = torch.load(Path(args.resume))

        # Unpack and load content
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

        checkpoint['scheduler_state_dict']['step_size'] = config['lr_drop']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        metric_start_val = checkpoint['metric_max_val']
    else:
        epoch = 0
        metric_start_val = 0


    # Init logging
    path_to_run = Path('/home/kb535/lss/codes/objection/transoar/runs')/ config['experiment_name']
    path_to_run.mkdir(exist_ok=True)

    # Get meta data and write config to run
    config.update(get_meta_data())
    write_json(config, path_to_run / 'config.json')

    # Build trainer and start training
    trainer = Trainer2(
        train_loader, val_loader, model, optim, scheduler, device, config, 
        path_to_run, epoch, metric_start_val
    )
    trainer.run()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    # Add minimal amount of args (most args should be set in config files)
    parser.add_argument("--config", type=str, default='attn_fpn_retina_unet_t2', help="Config to use for training located in /config.")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to use.", default=None)
    parser.add_argument('--local_rank', type=int, default=0, help = 'local rank. necessary for using the torch.distributed.launch utility')  ## my add
    args = parser.parse_args()

    # Get relevant configs
    config = get_config(args.config)

    # To get reproducable results
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    monai.utils.set_determinism(seed=config['seed'])
    random.seed(config['seed'])
    #
    torch.backends.cudnn.benchmark = True  # performance vs. reproducibility
    torch.backends.cudnn.deterministic = True

    train(config, args)
