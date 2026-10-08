# Estimate flow and occlusion mask via RegionMM for MHAD dataset
# this code is based on RegionMM from Snap Inc.
# https://github.com/snap-research/articulated-animation

import os
import sys
import math
import yaml
from argparse import ArgumentParser
from shutil import copy

import wandb
import datetime

from model.LFAE.generator import Generator
from model.LFAE.bg_motion_predictor import BGMotionPredictor
from model.LFAE.region_predictor import RegionPredictor

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random

from train import train

from utils.logger import Logger
from utils.seed import setup_seed

if __name__ == "__main__":
    cudnn.enabled = True
    cudnn.benchmark = True

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--postfix", default="")
    parser.add_argument("--config",default="./config/smmnist64.yaml",help="path to config")
    parser.add_argument("--log_dir",default='./logs_training/diffusion', help="path to log into")
    parser.add_argument("--checkpoint", # use the pretrained Taichi model provided by Snap
                        default="",
                        help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0,1", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    
    parser.add_argument("--random-seed", default=1234)
    parser.add_argument("--set-start", default=False)
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--verbose", default=False, help="Print model architecture")
    parser.add_argument('--local-rank', default=0, type=int, help='local rank')

    args = parser.parse_args()

    setup_seed(int(args.random_seed))

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.postfix == '':
        postfix = ''
    else:
        postfix = '_' + args.postfix

    log_dir = os.path.join(args.log_dir, os.path.basename(args.config).split('.')[0]+postfix)
    os.makedirs(log_dir, exist_ok=True)
    config_copy_path = os.path.join(log_dir, os.path.basename(args.config))
    if not os.path.exists(config_copy_path):
        try:
            copy(args.config, log_dir)
        except FileExistsError:
            pass

    # the directory to save checkpoints
    config["snapshots"] = os.path.join(log_dir, 'snapshots')
    os.makedirs(config["snapshots"], exist_ok=True)
    # the directory to save images of training results
    config["imgshots"] = os.path.join(log_dir, 'imgshots')
    os.makedirs(config["imgshots"], exist_ok=True)

    config["set_start"] = args.set_start

    train_params   = config['flow_params']['train_params']
    model_params   = config['flow_params']['model_params']
    dataset_params = config['dataset_params']   

    log_txt = os.path.join(log_dir,
                           "B"+format(train_params['batch_size'], "04d")+
                           "E"+format(train_params['max_epochs'], "04d")+".log")
    sys.stdout = Logger(log_txt, sys.stdout)
    
    if os.environ.get("WANDB_MODE", "disabled") != "disabled":
        wandb.login()
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project="EXtDM",
        config=config,
        name=f"{config['experiment_name']}{postfix}",
        dir=log_dir,
        tags=["flow"],
        # mode='disabled'
    )

    print("postfix:", postfix)
    print("checkpoint:", args.checkpoint)
    print("batch size:", train_params['batch_size'])

    generator = Generator(num_regions=model_params['num_regions'],
                          num_channels=model_params['num_channels'],
                          revert_axis_swap=model_params['revert_axis_swap'],
                          **model_params['generator_params'])

    if torch.cuda.is_available():
        generator.to(args.device_ids[0])
    if args.verbose:
        print(generator)

    region_predictor = RegionPredictor(num_regions=model_params['num_regions'],
                                       num_channels=model_params['num_channels'],
                                       estimate_affine=model_params['estimate_affine'],
                                       **model_params['region_predictor_params'])

    if torch.cuda.is_available():
        region_predictor.to(args.device_ids[0])

    if args.verbose:
        print(region_predictor)

    bg_predictor = BGMotionPredictor(num_channels=model_params['num_channels'],
                                     **model_params['bg_predictor_params'])
    if torch.cuda.is_available():
        bg_predictor.to(args.device_ids[0])
    if args.verbose:
        print(bg_predictor)

    print("Training...")
    train(
        config, 
        dataset_params,
        train_params, 
        generator, 
        region_predictor, 
        bg_predictor, 
        log_dir, 
        args.checkpoint, 
        args.device_ids
    )