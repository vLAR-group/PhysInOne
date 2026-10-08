import einops
import os
import mediapy as media
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.misc import grid2fig
from PIL import Image
import imageio

def visualize(save_path, origin, result, epoch_or_step_num=0, cond_frame_num=10, skip_pic_num=1, save_pic_num=8, select_method='top', grid_nrow=4, suffix = '',save_pic=True, save_pic_row=True, save_gif=True, save_gif_grid=True):
    # 输入: origin [b t c h w] + result [b t c h w]
    
    # 数据判定
    print("save", save_pic_num, "samples")
    assert origin.shape == result.shape, f"origin ({origin.shape}) result ({result.shape}) shape are not equal."
    assert cond_frame_num <= origin.shape[1], f"cond_frame_num ({cond_frame_num}) is too big for video length ({origin.shape[1]})."
    
    # 确认文件夹存在
    epoch_or_step_save_path = os.path.join(save_path, f"{epoch_or_step_num}")
    os.makedirs(epoch_or_step_save_path, exist_ok=True)

    # 确认输出图像数量
    save_pic_num = min (len(origin), save_pic_num)
    if len(origin) < save_pic_num:
        save_pic_num = min (len(origin), save_pic_num)
        print(f"video batchsize({len(origin)}) is too small, save_num is set to {save_pic_num}")
    
    # 选择需要输出的视频索引（前n个、均等取n个、直接根据索引提取n个）
    index = None
    if select_method == 'top':
        index = [int(i) for i in range(save_pic_num)]
    elif select_method == 'linspace':
        index = [int(i) for i in torch.linspace(0, len(origin)-1, save_pic_num)]
    elif isinstance(select_method, list):
        assert len(select_method) == save_pic_num
        index = [int(i) for i in select_method]
    
    print(index)

    origin = origin[index].cpu()
    result = result[index].cpu()
    
    # 输出视频单帧
    if save_pic:
        save_pic_path = os.path.join(epoch_or_step_save_path, "pic")
        os.makedirs(save_pic_path, exist_ok=True)
        
        origin_output = einops.rearrange(origin, "b t c h w -> b t h w c")
        result_output = einops.rearrange(result, "b t c h w -> b t h w c")
        
        for i in range(save_pic_num):
            for t in range(origin.shape[1]):
                # save_path = os.path.join(save_pic_path, str(index[i]), "origin")
                # os.makedirs(save_path, exist_ok=True)
                # media.write_image(os.path.join(save_path, f"pic_origin_{index[i]}_{t}.png"), origin_output[i, t].squeeze().numpy())
                save_path = os.path.join(save_pic_path, str(index[i]), "result")
                os.makedirs(save_path, exist_ok=True)
                media.write_image(os.path.join(save_path, f"pic_result_{index[i]}_{t}.png"), result_output[i, t].squeeze().numpy())
        
        
    # 输出视频分解对比图
    if save_pic_row:
        save_pic_row_path = os.path.join(epoch_or_step_save_path, "pic_row")
        os.makedirs(save_pic_row_path, exist_ok=True)
        
        all_video = torch.stack([origin, result])
        
        for i in range(save_pic_num):
            two_video = all_video[:, i]
            two_video[1, :cond_frame_num] = 1.0

            two_video = two_video[:,::skip_pic_num]

            # result of cond_frame set to blank frame
            two_video = einops.rearrange(two_video, "b t c h w -> (b h) (t w) c")
            save_path = save_pic_row_path
            # os.makedirs(save_path, exist_ok=True)
            # print(save_path)
            media.write_image(os.path.join(save_path, f"pic_row_{index[i]}.png"), two_video.squeeze().numpy())

    # 输出视频
    if save_gif:
        save_gif_path = os.path.join(epoch_or_step_save_path, "gif")
        os.makedirs(save_gif_path, exist_ok=True)
        
        origin_output = einops.rearrange(origin, "b t c h w -> b t h w c")
        result_output = einops.rearrange(result, "b t c h w -> b t h w c")

        for i in range(save_pic_num):
            save_path = os.path.join(save_gif_path, str(index[i]))
            os.makedirs(save_path, exist_ok=True)
            media.write_video(os.path.join(save_path, f"gif_origin_{index[i]}_{suffix}.gif"), origin_output[i].squeeze().numpy(), codec='gif', fps=20)
            save_path = os.path.join(save_gif_path, str(index[i]))
            os.makedirs(save_path, exist_ok=True)
            media.write_video(os.path.join(save_path, f"gif_result_{index[i]}_{suffix}.gif"), result_output[i].squeeze().numpy(), codec='gif', fps=20)

    # 输出视频对比网格
    if save_gif_grid:
        save_gif_grid_path = os.path.join(epoch_or_step_save_path, "gif_grid")
        os.makedirs(save_gif_grid_path, exist_ok=True)

        cond_color = [3/255,87/255,127/255]
        pred_color = [254/255,85/255,1/255]

        if origin.shape[2] == 1:
            origin_output = einops.repeat(origin, "b t c h w -> b t (3 c) h w")
            result_output = einops.repeat(result, "b t c h w -> b t (3 c) h w")
            origin_output = einops.rearrange(origin_output, "b t c h w -> b t h w c").numpy()
            result_output = einops.rearrange(result_output, "b t c h w -> b t h w c").numpy()
        else:
            origin_output = einops.rearrange(origin, "b t c h w -> b t h w c").numpy()
            result_output = einops.rearrange(result, "b t c h w -> b t h w c").numpy()

        videos_grids = []

        for i in range(save_pic_num):
            origin_video = origin_output[i]
            result_video = result_output[i]

            video_grids=[]

            for t in range(len(origin_video)):
                
                if t < cond_frame_num:
                    origin_img = cv2.copyMakeBorder(origin_video[t], 2,2,2,2, cv2.BORDER_CONSTANT, value=cond_color)
                    result_img = cv2.copyMakeBorder(result_video[t], 2,2,2,2, cv2.BORDER_CONSTANT, value=cond_color)
                else:
                    origin_img = cv2.copyMakeBorder(origin_video[t], 2,2,2,2, cv2.BORDER_CONSTANT, value=cond_color)
                    result_img = cv2.copyMakeBorder(result_video[t], 2,2,2,2, cv2.BORDER_CONSTANT, value=pred_color)

                video_grid = np.stack([origin_img, result_img])
                video_grid = einops.rearrange(video_grid, "n h w c -> h (n w) c")

                video_grids.append(video_grid)
                
            video_grids = np.stack(video_grids)

            videos_grids.append(video_grids)

        videos_grids = np.stack(videos_grids)

        videos_grids = torch.from_numpy(videos_grids)
        videos_grids = einops.rearrange(videos_grids, "b t h w c-> t b c h w")


        final_grids = []

        for t in range(len(videos_grids)):
            # make_grid need [b c h w]
            final_grids.append(make_grid(videos_grids[t], nrow=grid_nrow, padding=10, pad_value=1))

        final_grids = torch.stack(final_grids)
        final_grids = einops.rearrange(final_grids, "t c h w -> t h w c")

        save_path = save_gif_grid_path
        # os.makedirs(save_path, exist_ok=True)
        media.write_video(os.path.join(save_path, f"gif_grid.gif"), final_grids.numpy(), codec='gif', fps=20)

def visualize_ori_pre_flow(save_path, origin, result, origin_flow, result_flow, epoch_or_step_num=0):
    # 输入: 四个 tensor 形状均为 [b t c h w]
    # 输出：四个视频图片序列，标题为[视频号_序列号]。
    
    # 确认文件夹存在
    epoch_or_step_save_path = os.path.join(save_path, f"{epoch_or_step_num}")
    os.makedirs(epoch_or_step_save_path, exist_ok=True)
    save_pic_path = os.path.join(epoch_or_step_save_path, "pic")
    os.makedirs(save_pic_path, exist_ok=True)
    
    origin_output = einops.rearrange(origin, "b t c h w -> b t h w c")
    result_output = einops.rearrange(result, "b t c h w -> b t h w c")
    origin_flow_output = einops.rearrange(origin_flow, "b t c h w -> b t h w c")
    result_flow_output = einops.rearrange(result_flow, "b t c h w -> b t h w c")
    
    for i in tqdm(range(len(origin_output))):
        for t in range(origin.shape[1]):
            save_path = os.path.join(save_pic_path, str(i), "origin")
            os.makedirs(save_path, exist_ok=True)
            media.write_image(os.path.join(save_path, f"pic_origin_{i}_{t}.png"), origin_output[i, t].squeeze().numpy())
            save_path = os.path.join(save_pic_path, str(i), "origin_flow")
            os.makedirs(save_path, exist_ok=True)
            media.write_image(os.path.join(save_path, f"pic_origin_flow_{i}_{t}.png"), origin_flow_output[i, t].squeeze().numpy())
            save_path = os.path.join(save_pic_path, str(i), "result")
            os.makedirs(save_path, exist_ok=True)
            media.write_image(os.path.join(save_path, f"pic_result_{i}_{t}.png"), result_output[i, t].squeeze().numpy())
            save_path = os.path.join(save_pic_path, str(i), "result_flow")
            os.makedirs(save_path, exist_ok=True)
            media.write_image(os.path.join(save_path, f"pic_result_flow_{i}_{t}.png"), result_flow_output[i, t].squeeze().numpy())

def visualize_ori_pre_flow_conf(save_path, origin, result, flow, conf, cond_num, epoch_or_step_num=0):
    # 输入: 四个 tensor 形状均为 [b t c h w]，gt、result、flow、occu
    # 输出：四个视频图片序列，标题为[视频号_序列号]。
    
    # 确认文件夹存在
    epoch_or_step_save_path = os.path.join(save_path, f"{epoch_or_step_num}")
    os.makedirs(epoch_or_step_save_path, exist_ok=True)
    save_pic_path = os.path.join(epoch_or_step_save_path, "pic")
    os.makedirs(save_pic_path, exist_ok=True)
    
    # b t c h w -> n b t c h w
    videos_output = torch.stack([origin, result, flow, conf])
    
    # n b t c h w -> b n t h w c
    videos_output = einops.rearrange(videos_output, "n b t c h w -> b n t h w c")[:, :, cond_num:]
    
    for i in tqdm(range(len(videos_output))):
        # get [n t h w c] (n=4)
        video_output = videos_output[i]
        video_output = einops.rearrange(video_output, "n t h w c -> (n h) (t w) c")
        media.write_image(os.path.join(save_path, f"pic_ori_res_flow_conf_{i}.png"), video_output.squeeze().numpy())

def visualize_ori_pre_flow_conf_save_pic(save_path, origin, result, flow, conf, cond_num, epoch_or_step_num=0):
    # 输入: 四个 tensor 形状均为 [b t c h w]，gt、result、flow、occu
    # 输出：四个图片序列
    
    # 确认文件夹存在
    epoch_or_step_save_path = os.path.join(save_path, f"{epoch_or_step_num}")
    os.makedirs(epoch_or_step_save_path, exist_ok=True)

    save_pic_path = os.path.join(epoch_or_step_save_path, "pic")
    os.makedirs(save_pic_path, exist_ok=True)

    origin_output = einops.rearrange(origin, "b t c h w -> b t h w c")
    result_output = einops.rearrange(result, "b t c h w -> b t h w c")
    flow_output   = einops.rearrange(flow,   "b t c h w -> b t h w c")
    conf_output   = einops.rearrange(conf,   "b t c h w -> b t h w c")

    for i in tqdm(range(len(origin_output))):
        for t in range(origin.shape[1]):
            save_path = os.path.join(save_pic_path, str(i), "result")
            os.makedirs(save_path, exist_ok=True)
            media.write_image(os.path.join(save_path, f"pic_result_{i}_{t}.png"), result_output[i, t].squeeze().numpy())
            
            if t >= cond_num:
                save_flow_path = os.path.join(save_pic_path, str(i), "flow")
                os.makedirs(save_flow_path, exist_ok=True)
                media.write_image(os.path.join(save_flow_path, f"pic_flow_{i}_{t}.png"), flow_output[i, t].squeeze().numpy())
                save_conf_path = os.path.join(save_pic_path, str(i), "conf")
                os.makedirs(save_conf_path, exist_ok=True)
                media.write_image(os.path.join(save_conf_path, f"pic_conf_{i}_{t}.png"), conf_output[i, t].squeeze().numpy())
        
def visualize_ori_pre_flow_diff(save_path, origin, result, origin_flow, result_flow, video_diff, flow_diff, epoch_or_step_num=0, cond_frame_num=10, skip_pic_num=1):
    # 输入: 五个 tensor 形状均为 [b t c h w]
    # 输出：五个合并的视频，标题为[序列号]_psnr[大小]。
    
    # 确认文件夹存在
    
    # 先计算psnr
    from metrics.calculate_psnr import calculate_psnr2
    psnr_results = calculate_psnr2(origin[:, cond_frame_num:], result[:, cond_frame_num:])
    
    # 后续工作
    epoch_or_step_save_path = os.path.join(save_path, f"{epoch_or_step_num}")
    os.makedirs(epoch_or_step_save_path, exist_ok=True)

    save_gif_grid_path = os.path.join(epoch_or_step_save_path, "gif_grid")
    os.makedirs(save_gif_grid_path, exist_ok=True)

    cond_color = [3/255,87/255,127/255]
    pred_color = [254/255,85/255,1/255]

    # origin, result, origin_flow, result_flow, diff
    
    videos = torch.stack([origin, result, video_diff, origin_flow, result_flow, flow_diff])
    videos = einops.rearrange(videos, '(n r) b t c h w -> b t (n h) (r w) c', n=2, r=3).numpy()

    for i in range(len(videos)):
        video_output = []
        video = videos[i]
        for t in range(len(video)):
            if t < cond_frame_num:
                output = cv2.copyMakeBorder(video[t], 2,2,2,2, cv2.BORDER_CONSTANT, value=cond_color)
            else:
                output = cv2.copyMakeBorder(video[t], 2,2,2,2, cv2.BORDER_CONSTANT, value=pred_color)
            video_output.append(output)
        video_output = np.stack(video_output)
        save_path = save_gif_grid_path
        media.write_video(os.path.join(save_path, f"{i:03}_psnr{str(psnr_results[i])}.gif"), video_output, codec='gif', fps=2)
        
def LFAE_visualize(
            ground, prediction, video_names,  save_path, 
            deformed=None, optical_flow=None, occlusion_map=None,
            save_num=8, epoch_or_step_num=0, image_size=64
        ):
    
    # 确认文件夹存在
    epoch_or_step_save_path = os.path.join(save_path, f"{epoch_or_step_num}")
    os.makedirs(epoch_or_step_save_path, exist_ok=True)

    index = [int(i) for i in torch.linspace(0, ground.size(0)-1, save_num)]

    if deformed is None or optical_flow is None or occlusion_map is None:
        print(ground.shape, prediction.shape)
        for batch_idx in index:
            new_im_list = []
            for frame_idx in range(ground.size(2)):
                # cond+real
                save_tar_img = sample_img(ground[:, :, frame_idx], batch_idx)
                # prediction
                save_out_img = sample_img(prediction[frame_idx], batch_idx)
                # save img_list
                new_im = Image.new('RGB', (image_size * 2, image_size))
                new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, 0))
                new_im.paste(Image.fromarray(save_out_img, 'RGB'), (image_size, 0))
                new_im_list.append(new_im)
            imageio.mimsave(os.path.join(epoch_or_step_save_path, f"{video_names[batch_idx]}.gif"), new_im_list)
    else:
        for batch_idx in index:
            new_im_list = []
            for frame_idx in range(ground.size(2)):
                # cond+real
                save_tar_img = sample_img(ground[:, :, frame_idx], batch_idx)
                # prediction
                save_out_img = sample_img(prediction[frame_idx], batch_idx)
                # deformed
                save_warped_img = sample_img(deformed[frame_idx], batch_idx)
                # optical_flow
                save_warped_grid = grid2fig(optical_flow[frame_idx, batch_idx].data.cpu().numpy(),grid_size=32, img_size=image_size)
                # occlusion_map
                save_conf_map = occlusion_map[frame_idx, batch_idx].unsqueeze(dim=0)
                save_conf_map = save_conf_map.data.cpu()
                save_conf_map = F.interpolate(save_conf_map, size=ground.shape[3:5]).numpy()
                save_conf_map = np.transpose(save_conf_map, [0, 2, 3, 1])
                save_conf_map = np.array(save_conf_map[0, :, :, 0]*255, dtype=np.uint8)
                # save img_list
                new_im = Image.new('RGB', (image_size * 5, image_size))
                new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, 0))
                new_im.paste(Image.fromarray(save_out_img, 'RGB'), (image_size, 0))
                new_im.paste(Image.fromarray(save_warped_img, 'RGB'), (image_size * 2, 0))
                new_im.paste(Image.fromarray(save_warped_grid), (image_size * 3, 0))
                new_im.paste(Image.fromarray(save_conf_map, "L"), (image_size * 4, 0))
                new_im_list.append(new_im)
            imageio.mimsave(os.path.join(epoch_or_step_save_path, f"{video_names[batch_idx]}.gif"), new_im_list)

def sample_img(rec_img_batch, index=0):
    rec_img = rec_img_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array((0.0, 0.0, 0.0))/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)