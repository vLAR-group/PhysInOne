import argparse

import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import yaml
from shutil import copy
from train import train

from utils.seed import setup_seed
from utils.logger import Logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb
import os.path as osp
import timeit
import math
from PIL import Image
import sys
import random
from einops import rearrange

if __name__ == '__main__':
    cudnn.enabled = True
    cudnn.benchmark = True

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--postfix", default="")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--set-start", default=True)
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--log_dir",default='./logs_training/diffusion', help="path to log into")
    parser.add_argument("--config",default="./config/smmnist64.yaml",help="path to config")
    parser.add_argument("--device_ids", default="0,1", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--flowae_checkpoint", # use the flowae_checkpoint pretrained model provided by Snap
                        default="",
                        help="path to flowae_checkpoint checkpoint")
    parser.add_argument("--verbose", default=False, help="Print model architecture")
    parser.add_argument("--fp16", default=False)
    
    args = parser.parse_args()

    setup_seed(int(args.random_seed))

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.postfix == '':
        postfix = ''
    else:
        postfix = '_' + args.postfix

    log_dir = os.path.join(args.log_dir, os.path.basename(args.config).split('.')[0]+postfix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
        copy(args.config, log_dir)

    # the directory to save checkpoints
    config["snapshots"] = os.path.join(log_dir, 'snapshots')
    os.makedirs(config["snapshots"], exist_ok=True)
    # the directory to save images of training results
    config["imgshots"] = os.path.join(log_dir, 'imgshots')
    os.makedirs(config["imgshots"], exist_ok=True)
    # vidshots
    config["vidshots"] = os.path.join(log_dir, 'vidshots')
    os.makedirs(config["vidshots"], exist_ok=True)
    # samples
    config["samples"] = os.path.join(log_dir, 'samples')
    os.makedirs(config["samples"], exist_ok=True)

    config["set_start"] = args.set_start

    train_params   = config['diffusion_params']['train_params']
    model_params   = config['diffusion_params']['model_params']
    dataset_params = config['dataset_params'] 

    log_txt = os.path.join(log_dir,
                           "B"+format(train_params['batch_size'], "04d")+
                           "E"+format(train_params['max_epochs'], "04d")+".log")
    sys.stdout = Logger(log_txt, sys.stdout)

    if os.environ.get("WANDB_MODE", "disabled") != "disabled":
        wandb.login()
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project="EXtdm",
        config=config,
        name=f"{config['experiment_name']}{postfix}",
        dir=log_dir,
        tags=["diffusion"],
        # mode='disabled'
    )

    print("postfix:", postfix)
    print("checkpoint:", args.checkpoint)
    print("flowae checkpoint:", args.flowae_checkpoint)
    print("batch size:", train_params['batch_size'])

    config['flowae_checkpoint'] = args.flowae_checkpoint

    print("Training...")
    train(
        config, 
        dataset_params,
        train_params, 
        log_dir, 
        args.checkpoint, 
        args.device_ids
    )
    