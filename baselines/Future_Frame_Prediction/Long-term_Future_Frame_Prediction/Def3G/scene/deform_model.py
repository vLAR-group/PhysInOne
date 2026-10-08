import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, CodeField
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, quaternion_multiply
from utils.velocity_field_utils import VelBasis, VelocityWarpper, AccBasis, SegVel
import einops

class DeformModel:
    def __init__(self, is_blender=False, is_6dof=False, max_time=0.7, vel_start_time=0.0):
        deform_code_dim = 16 # 16
        self.deform = DeformNetwork(D=8, W=256, input_ch=3, hyper_ch=deform_code_dim, multires=8,
                                    is_blender=is_blender, is_6dof=is_6dof, gated=False).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.max_time = max_time
        self.vel_start_time = vel_start_time

    def _empty_outputs(self, xyz):
        d_xyz = torch.zeros_like(xyz)
        d_rotation = torch.zeros((xyz.shape[0], 4), device=xyz.device, dtype=xyz.dtype)
        d_scale = torch.zeros_like(xyz)
        return d_xyz, d_rotation, d_scale

    def step(self, xyz, time_emb, deform_code, dt=1/60, max_time=0.75):
        if xyz.shape[0] == 0 or time_emb.shape[0] == 0:
            return self._empty_outputs(xyz)
        d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
        return d_xyz, d_rotation, d_scale

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        deform_weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(deform_weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

