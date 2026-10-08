import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, CodeField
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, quaternion_multiply
from utils.physics_utils import Integrator, TRD
import einops

class DeformModel:
    def __init__(self, is_blender=True, is_6dof=False, max_time=0.7, vel_start_time=0.0, light=True, physics_code=16, freegave=False):
        motion_patterns = physics_code # 16
        self.freegave = freegave
        if freegave:
            self.code_field = CodeField(D=4, W=128, input_ch=3, output_ch=motion_patterns, multires=8).cuda()

        self.deform = DeformNetwork(D=8, W=256, input_ch=3, hyper_ch=motion_patterns, multires=8,
                                    is_blender=is_blender, is_6dof=is_6dof, freegave=freegave).cuda()
        if light:
            print("light mode")
            self.vel_net = TRD(motion_patterns=motion_patterns, hidden_dim=128, layers=6, encode_dim=3, freegave=freegave).cuda()
        else:
            print("complex mode")
            self.vel_net = TRD(motion_patterns=motion_patterns, hidden_dim=256, layers=8, encode_dim=6, freegave=freegave).cuda()
        self.vel = Integrator(self.vel_net)
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.max_time = max_time
        self.vel_start_time = vel_start_time

    def _empty_outputs(self, xyz):
        d_xyz = torch.zeros_like(xyz)
        d_rotation = torch.zeros((xyz.shape[0], 4), device=xyz.device, dtype=xyz.dtype)
        d_scale = torch.zeros_like(xyz)
        return d_xyz, d_rotation, d_scale

    def step_no_rot(self, xyz, time_emb, dt=1/60, max_time=0.75):
        if xyz.shape[0] == 0 or time_emb.shape[0] == 0:
            return self._empty_outputs(xyz)

        max_time = self.max_time
        min_time = max(self.vel_start_time, dt)
        if time_emb[0, 0] >= min_time and time_emb[0, 0] <= max_time:
            sign = 1 if torch.rand(1)[0] > 0.5 else -1
            deform_time = (time_emb - dt * sign).clamp(self.vel_start_time, max_time)
            if self.freegave:
                deform_code = self.code_field(xyz.detach())
                deform_seg = self.code_field.seg(deform_code)
            else:
                deform_code = None
                deform_seg = None
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            xyz_vel = (self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=False))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        elif time_emb[0, 0] > max_time:
            # in training, we only consider to integrate once, so we need to modify dt here
            deform_time = torch.ones_like(time_emb) * max_time
            dt = time_emb[0, 0] - max_time
            if self.freegave:
                deform_code = self.code_field(xyz.detach())
                deform_seg = self.code_field.seg(deform_code)
            else:
                deform_code = None
                deform_seg = None
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            xyz_vel = (
                self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=False))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
        else:
            if self.freegave:
                deform_code = self.code_field(xyz.detach())
            else:
                deform_code = None
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
        return d_xyz, d_rotation, d_scale

    def step_full_rot(self, xyz, time_emb, dt=1/60, max_time=0.75):
        if xyz.shape[0] == 0 or time_emb.shape[0] == 0:
            return self._empty_outputs(xyz)

        max_time = self.max_time
        if time_emb[0, 0] >= dt and time_emb[0, 0] <= max_time:
            sign = 1 if torch.rand(1)[0] > 0.5 else -1
            deform_time = (time_emb - dt * sign).clamp(0, max_time)
            if self.freegave:
                deform_code = self.code_field(xyz.detach())
                deform_seg = self.code_field.seg(deform_code)
            else:
                deform_code = None
                deform_seg = None
            d_xyz_deform, d_rotation, d_scale =  self.deform(xyz.detach(), deform_time, deform_code)
            xyz_vel, d_rotation2 = (self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=True))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            d_rotation = quaternion_multiply(d_rotation2, d_rotation)
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        elif time_emb[0, 0] > max_time:
            deform_time = torch.ones_like(time_emb) * max_time
            if self.freegave:
                deform_code = self.code_field(xyz.detach())
                deform_seg = self.code_field.seg(deform_code)
            else:
                deform_code = None
                deform_seg = None
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            xyz_vel, d_rotation2 = (
                self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=True))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            d_rotation = quaternion_multiply(d_rotation2, d_rotation)
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        else:
            if self.freegave:
                deform_code = self.code_field(xyz.detach())
            else:
                deform_code = None
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
        return d_xyz, d_rotation, d_scale

    def step(self, xyz, time_emb, dt=1/60, max_time=0.75):
        if xyz.shape[0] == 0 or time_emb.shape[0] == 0:
            return self._empty_outputs(xyz)

        max_time = self.max_time
        if time_emb[0, 0] > max_time:
            deform_time = torch.ones_like(time_emb) * max_time
            if self.freegave:
                deform_code = self.code_field(xyz.detach())
                deform_seg = self.code_field.seg(deform_code)
            else:
                deform_code = None
                deform_seg = None
            d_xyz_deform, d_rotation, d_scale = self.deform(xyz.detach(), deform_time, deform_code)
            xyz_vel, d_rotation2 = (
                self.vel.integrate_pos(deform_seg, xyz.detach() + d_xyz_deform.detach(), deform_time, time_emb, dt, rot=True))
            d_xyz_vel = xyz_vel - d_xyz_deform.detach() - xyz.detach()
            d_xyz = d_xyz_vel + d_xyz_deform
            d_rotation = quaternion_multiply(d_rotation2, d_rotation)
            # d_rotation = quaternion_multiply(d_rotation, d_rotation2)
        else:
            if self.freegave:
                deform_code = self.code_field(xyz.detach())
            else:
                deform_code = None
            d_xyz, d_rotation, d_scale = self.deform(xyz.detach(), time_emb, deform_code)
        return d_xyz, d_rotation, d_scale

    def train_setting(self, training_args):
        if self.freegave:
            params = list(self.deform.parameters()) + list(self.vel.parameters()) + list(self.code_field.parameters())
        else:
            params = list(self.deform.parameters()) + list(self.vel.parameters())
        l = [
            {'params': params,
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
        if self.freegave:
            torch.save(self.code_field.state_dict(), os.path.join(out_weights_path, 'code_field.pth'))
        torch.save(self.vel.state_dict(), os.path.join(out_weights_path, 'vel.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        deform_weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(deform_weights_path))
        if self.freegave:
            code_field_weights_path = os.path.join(model_path, "deform/iteration_{}/code_field.pth".format(loaded_iter))
            self.code_field.load_state_dict(torch.load(code_field_weights_path))
        vel_weights_path = os.path.join(model_path, "deform/iteration_{}/vel.pth".format(loaded_iter))
        self.vel.load_state_dict(torch.load(vel_weights_path))
        # try:
        #     self.vel.load_state_dict(torch.load(vel_weights_path))
        # except:
        #     print("No velocity weights found")

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

