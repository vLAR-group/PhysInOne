import torch
import os.path
import numpy as np
import math
import sys
from shutil import copy2
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from torch.optim.lr_scheduler import LambdaLR

from data.two_frames_dataset import DatasetRepeater

from utils.misc import grid2fig, conf2fig
from utils.meter import AverageMeter
from utils.visualize import sample_img
from utils.seed import setup_seed

from einops import rearrange, repeat
from PIL import Image
import timeit
import wandb
from einops import rearrange
import imageio

import torch.backends.cudnn as cudnn
from data.video_dataset import VideoDataset, dataset2videos ,PhysBenchVideoDataset
from utils.visualize import visualize
# from model.BaseDM_adaptor.VideoFlowDiffusion_multi import FlowDiffusion
# from model.BaseDM_adaptor.VideoFlowDiffusion_multi1248 import FlowDiffusion
# from model.BaseDM_adaptor_for_MACs.VideoFlowDiffusion_multi_w_ref import FlowDiffusion
# from model.BaseDM_adaptor.VideoFlowDiffusion_multi_w_ref import FlowDiffusion
from model.BaseDM_adaptor.VideoFlowDiffusion_multi_w_ref_u22 import FlowDiffusion

# Configure evaluator

# Initialize


from accelerate import Accelerator, DistributedDataParallelKwargs

os.environ['NCCL_DEBUG'] = "INFO"
os.environ['NCCL_P2P_DISABLE']='1'

def train(
        config, 
        dataset_params,
        train_params,
        log_dir, 
        checkpoint,
        device_ids
    ):

    print(config)
    print(device_ids)
    # accelerator = Accelerator()
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    model = FlowDiffusion(
        config=config,
        pretrained_pth=config['flowae_checkpoint'],
        is_train=True,
    )

    def count_parameters(model):
        res = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"count_training_parameters: {res}")
        res = sum(p.numel() for p in model.parameters())
        print(f"count_all_parameters:      {res}")
    
    count_parameters(model)

    model.cuda()

    train_dataset = PhysBenchVideoDataset(
        root_dir=dataset_params['root_dir'],
        type='train', 
        total_videos=-1,
        cond_frames=dataset_params['train_params']['cond_frames'] , 
        image_size=64, 
        num_frames=dataset_params['train_params']['cond_frames'] + dataset_params['train_params']['pred_frames'],
        scene_list_txt=dataset_params.get('train_scene_list_txt')
    )

    valid_dataset = PhysBenchVideoDataset(
        root_dir=dataset_params['root_dir'],
        type='test', 
        total_videos=10,
        cond_frames=dataset_params['train_params']['cond_frames'] , 
        image_size=64, 
        scene_list_txt=dataset_params.get('test_scene_list_txt')
    )

    # 计算一个 epoch 有多少 step 
    steps_per_epoch = math.ceil(train_params['num_repeats'] * len(train_dataset) / float(train_params['batch_size']))
    # 多少 step 保存一次模型
    save_ckpt_freq = train_params['save_ckpt_freq']
    print("save ckpt freq:", save_ckpt_freq)
    
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        train_dataset = DatasetRepeater(train_dataset, train_params['num_repeats'])

    optimizer = torch.optim.AdamW(
        model.diffusion.parameters(), 
        lr=train_params['lr'],
        betas=(0.9, 0.999),
        eps=1.0e-08,
        weight_decay=0.0,
        amsgrad=False
    )

    start_epoch = 0
    start_step = 0
    best_fvd = 1e5

    if checkpoint is not None:
        if os.path.isfile(checkpoint):
            print("=> loading checkpoint '{}'".format(checkpoint))
            device = accelerator.device
            ckpt = torch.load(checkpoint , map_location=device)
            if config["set_start"]:
                start_step = int(math.ceil(ckpt['example'] / train_params['batch_size'])) - 1
                start_epoch = ckpt['epoch']

                print("start_step", start_step)
                print("start_epoch", start_epoch)

            model_ckpt = model.diffusion.state_dict()
            for name, _ in model_ckpt.items():
                model_ckpt[name].copy_(ckpt['diffusion'][name])
            model.diffusion.load_state_dict(model_ckpt)
            print("=> loaded checkpoint '{}'".format(checkpoint))
            if "optimizer" in list(ckpt.keys()):
                optimizer.load_state_dict(ckpt['optimizer'])
            del ckpt, model_ckpt
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint))
    else:
        print("NO checkpoint found!")

    scheduler = MultiStepLR(optimizer, last_epoch=start_step - 1, **train_params['scheduler_param'])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_params['batch_size'],
        shuffle=True, 
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=100,
        
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=train_params['valid_batch_size'],
        shuffle=False, 
        num_workers=1,
        pin_memory=True, 
        drop_last=False,
        prefetch_factor=100,
    )

    # 使用 Accelerate 封装模型、优化器、数据加载器
    print("开始封装")
    model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader
    )

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_rec = AverageMeter()
    losses_warp = AverageMeter()

    cnt = 0
    epoch_cnt = start_epoch
    actual_step = start_step
    final_step = steps_per_epoch * train_params["max_epochs"]

    print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer.param_groups[0]["lr"]))
    
    while actual_step < final_step:
        iter_end = timeit.default_timer()

        for i_iter, batch in enumerate(train_dataloader):
            actual_step = int(start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)

            real_vids, real_names = batch
            # (b t c h)/(b t h w c) -> (b t c h w)
            real_vids = dataset2videos(real_vids)
            real_vids = rearrange(real_vids, 'b t c h w -> b c t h w')

            # print(real_vids.shape)
            # torch.Size([8, 20, 3, 64, 64])
            # torch.Size([bs, c, length, h, w])
            ref_imgs = real_vids[:, :, dataset_params['train_params']['cond_frames']-1, :, :].clone().detach()
            real_vid = real_vids[:, :, :dataset_params['train_params']['cond_frames'] + dataset_params['train_params']['pred_frames']].cuda()
            
            optimizer.zero_grad()
            ret = model(real_vid)

            # print(ret['loss'].mean().item())
            # print(ret['rec_loss'].mean().item())
            # print(ret['rec_warp_loss'].mean().item())

            loss_ = ret['loss'].mean()
            loss_rec = ret['rec_loss'].mean()
            loss_rec_warp = ret['rec_warp_loss'].mean()

            # if model.module.only_use_flow:
            #     # loss_.backward()
            #     accelerator.backward(loss_)
            # else:
            #     # (loss_ + loss_rec + loss_rec_warp).backward()
            #     accelerator.backward(loss_ + loss_rec + loss_rec_warp)
            #     # loss_.backward()
            #     # loss_rec.backward()
            #     # loss_rec_warp.backward()
            accelerator.backward(loss_ + loss_rec + loss_rec_warp)
            # print('反向传播成功！')
            optimizer.step()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            bs = real_vids.size(0)
            losses.update(loss_.item(), bs)
            losses_rec.update(loss_rec.item(), bs)
            losses_warp.update(loss_rec_warp.item(), bs)

            if actual_step % train_params["print_freq"] == 0 and cnt != 0:
                print('iter: [{0}]{1}/{2}\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'loss_rec {loss_rec.val:.3f} ({loss_rec.avg:.3f})\t'
                      'loss_warp {loss_warp.val:.3f} ({loss_warp.avg:.3f})\t'
                      'time {batch_time.val:.2f}({batch_time.avg:.2f})'
                    .format(
                    cnt, actual_step, final_step,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_rec=losses_rec,
                    loss_warp=losses_warp,
                ))
                if accelerator.is_main_process:
                    wandb.log({
                        "actual_step": actual_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss": losses.val, 
                        "loss_rec": losses_rec.val,
                        "loss_warp": losses_warp.val,
                        "batch_time": batch_time.avg
                    })

            # if actual_step % train_params['save_img_freq'] == 0 and cnt != 0:
            #     msk_size = ref_imgs.shape[-1]
            #     save_src_img = sample_img(ref_imgs)
            #     save_tar_img = sample_img(real_vids[:, :, dataset_params['train_params']['cond_frames']+dataset_params['train_params']['pred_frames']//2, :, :])
            #     save_real_out_img = sample_img(ret['real_out_vid'][:, :, dataset_params['train_params']['cond_frames']+dataset_params['train_params']['pred_frames']//2, :, :])
            #     save_real_warp_img = sample_img(ret['real_warped_vid'][:, :, dataset_params['train_params']['cond_frames']+dataset_params['train_params']['pred_frames']//2, :, :])
            #     save_fake_out_img = sample_img(ret['fake_out_vid'][:, :, dataset_params['train_params']['pred_frames']//2, :, :])
            #     save_fake_warp_img = sample_img(ret['fake_warped_vid'][:, :, dataset_params['train_params']['pred_frames']//2, :, :])
            #     save_real_grid = grid2fig(ret['real_vid_grid'][0, :, dataset_params['train_params']['cond_frames']+dataset_params['train_params']['pred_frames']//2].permute((1, 2, 0)).data.cpu().numpy(),
            #                               grid_size=12, img_size=msk_size)
            #     save_fake_grid = grid2fig(ret['fake_vid_grid'][0, :, dataset_params['train_params']['pred_frames']//2].permute((1, 2, 0)).data.cpu().numpy(),
            #                               grid_size=12, img_size=msk_size)
            #     save_real_conf = conf2fig(ret['real_vid_conf'][0, :, dataset_params['train_params']['cond_frames']+dataset_params['train_params']['pred_frames']//2], img_size=dataset_params['frame_shape'])
            #     save_fake_conf = conf2fig(ret['fake_vid_conf'][0, :, dataset_params['train_params']['pred_frames']//2], img_size=dataset_params['frame_shape'])
            #     new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                
            #     # imgshot
            #     # -------------------------------------------------------
            #     # | src | real_out  | real_warp | real_grid | real_conf |
            #     # -------------------------------------------------------
            #     # | tar | fake_out  | fake_warp | fake_grid | fake_conf |
            #     # -------------------------------------------------------
                
            #     new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
            #     new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                
            #     new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
            #     new_im.paste(Image.fromarray(save_fake_out_img, 'RGB'), (msk_size, msk_size))
                
            #     new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size * 2, 0))
            #     new_im.paste(Image.fromarray(save_fake_warp_img, 'RGB'), (msk_size * 2, msk_size))
                
            #     new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
            #     new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                
            #     new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
            #     new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                
            #     new_im_name = 'B' + format(train_params["batch_size"], "04d") + '_S' + format(actual_step, "06d") \
            #                   + '_' + format(real_names[0], "06d") + ".png"
            #     new_im_file = os.path.join(config["imgshots"], new_im_name)
            #     new_im.save(new_im_file)
            #     if accelerator.is_main_process:
            #         wandb.log({
            #             "save_img": wandb.Image(new_im)
            #         })

            if actual_step % train_params['save_vid_freq'] == 0 and cnt != 0:
                i_pred_video = model.module.sample_one_video(cond_scale=1.0, real_vid=real_vids[:,:,:dataset_params['train_params']['cond_frames']])['sample_out_vid'].clone().detach().cpu()
                i_pred_video= rearrange(i_pred_video, 'b c t h w -> b t c h w')
                gt_vid = rearrange(real_vids, 'b c t h w -> b t c h w')
                visualize(
                    save_path=f"{log_dir}/video_result",
                    origin=gt_vid ,
                    result=i_pred_video ,
                    save_pic_num=8,
                    select_method='linspace',
                    grid_nrow=4,
                    save_gif_grid=False,
                    save_gif=True,
                    save_pic_row=False,
                    save_pic=False,
                    epoch_or_step_num=actual_step, 
                    cond_frame_num=dataset_params['train_params']['cond_frames'],
                    suffix=f"iter{i_iter}"
                )


            # save model
            if actual_step % train_params['save_ckpt_freq'] == 0 and cnt != 0:
                print('taking snapshot ...')
                # valid(config, valid_dataloader, model, log_dir, actual_step , accelerator)
                torch.save({
                        'example': actual_step * train_params["batch_size"],
                        'epoch': epoch_cnt,
                        'diffusion': model.module.diffusion.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(config["snapshots"],'flowdiff_' + format(train_params["batch_size"], "04d") + '_S' + format(actual_step, "06d") + '.pth'))

            # update saved model
            if actual_step % train_params['update_ckpt_freq'] == 0 and cnt != 0:
                print('updating saved snapshot ...')
                checkpoint_save_path=os.path.join(config["snapshots"], 'flowdiff.pth')
                # valid(config, valid_dataloader, model, log_dir, actual_step , accelerator)
                torch.save({
                        'example': actual_step * train_params["batch_size"],
                        'epoch': epoch_cnt,
                        'diffusion': model.module.diffusion.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    checkpoint_save_path)
                # Private validation is performed by the official leaderboard.
                # if metrics['metrics/fvd'] < best_fvd:
                #     best_fvd = metrics['metrics/fvd']
                #     copy2(os.path.join(config["snapshots"], 'flowdiff.pth'), os.path.join(config["snapshots"], f'flowdiff_best_{best_fvd:.3f}.pth'))
                # if accelerator.is_main_process:
                #     wandb.log(metrics)

            if actual_step >= final_step:
                break

            cnt += 1
            # 按 step 进行 warmup 策略
            scheduler.step()
            
        epoch_cnt += 1
        print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer.param_groups[0]["lr"]))

    print('save the final model ...')
    torch.save({
        'example': actual_step * train_params["batch_size"],
        'diffusion': model.module.diffusion.state_dict(),
        'optimizer': optimizer.state_dict()
        },
        os.path.join(config["snapshots"], 'flowdiff_' + format(train_params["batch_size"], "04d") + '_S' + format(actual_step, "06d") + '.pth'))


