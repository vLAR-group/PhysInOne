import torch
import torch.nn as nn
from functorch import vmap, jacrev
import numpy as np
import einops
from einops.layers.torch import Rearrange
from numpy.ma.core import identity
from utils.general_utils import rotation_to_quaternion


class PositionEncoder(nn.Module):

    def __init__(self, encode_dim, log_sampling=True):
        super(PositionEncoder, self).__init__()

        self.encode_dim = encode_dim
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                0.0,
                self.encode_dim - 1,
                self.encode_dim,
                dtype=torch.float32
            )
        else:
            frequency_bands = torch.linspace(
                2.0 ** 0.0,
                2.0 ** (self.encode_dim - 1),
                self.encode_dim,
                dtype=torch.float32
            )
        self.register_buffer('frequency_bands', frequency_bands)

    def forward(self, x):

        encoding = [x]

        for freq in self.frequency_bands:
            encoding.append(torch.sin(x * freq))
            encoding.append(torch.cos(x * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


class Integrator(nn.Module):

    def __init__(self, trd):
        super(Integrator, self).__init__()
        self.vel_net = trd

    def update_weights(self, weights, a, dt):
        if self.vel_net.freegave:
            dt_ = dt.unsqueeze(-1)
        else:
            dt_ = dt
        weights_new = weights + dt_ * a
        return weights_new

    def integrate_pos(self, pattern_weights, pos_init, t1, t2, dt_max, rot=False):
        dt_max = torch.ones_like(t1) * dt_max
        time_offset = t1 - t2
        xyz_prev = pos_init
        t_curr = t1
        unfinished = (time_offset.abs() > 0).squeeze(-1)
        if rot:
            rot_prev = torch.zeros(pos_init.shape[0], 3, 3, dtype=pos_init.dtype, device=pos_init.device)
            rot_prev[:, 0, 0] = 1
            rot_prev[:, 1, 1] = 1
            rot_prev[:, 2, 2] = 1

        weights, a = self.vel_net.get_weights(pattern_weights, torch.cat([xyz_prev, t_curr], dim=-1))

        while unfinished.any():
            # get time step
            dt = time_offset.sign() * torch.minimum(time_offset.abs(), dt_max)
            # Runge-Kutta 2
            xyzt_prev = torch.cat([xyz_prev, t_curr], dim=-1)

            velocity = self.vel_net.get_vel(pattern_weights, xyzt_prev, weights)
            p_mid = xyz_prev - 0.5 * dt * velocity
            t_mid = t_curr - 0.5 * dt
            pt_mid = torch.cat([p_mid, t_mid], dim=-1)

            weights = self.update_weights(weights, a, -0.5 * dt)

            velocity = self.vel_net.get_vel(pattern_weights, pt_mid, weights)
            if rot:
                drot = self.vel_net.rodrigues(pattern_weights, weights, -dt)
                rot_prev = einops.einsum(drot, rot_prev, '... i j, ... j k -> ... i k')

            xyz_cur = xyz_prev - dt * velocity
            xyz_prev = xyz_cur
            time_offset = time_offset - dt
            t_curr = t_curr - dt
            weights = self.update_weights(weights, a, -0.5 * dt)
            unfinished = (time_offset.abs() > 0).squeeze(-1)

        if rot:
            rot_prev = rotation_to_quaternion(rot_prev)
            return xyz_prev, rot_prev
        return xyz_prev # + dxyz_acc


class TRD(nn.Module):
    def __init__(self, motion_patterns=8, hidden_dim=128, layers=6, encode_dim=3, freegave=False):
        super(TRD, self).__init__()
        self.K = motion_patterns

        self.freegave = freegave
        if freegave:
            in_dim = 1 + 1 * 2 * encode_dim
            print(f"Decompose into {self.K} moiton patterns by FreeGave decomposition.")
            out_dim = (6 + 6) * self.K
        else:
            in_dim = 4 + 4 * 2 * encode_dim
            out_dim = 6 + 6
        self.embedder = PositionEncoder(encode_dim)
        self.weight_net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        for i in range(layers):
            self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, out_dim)))
        if freegave:
            self.weight_net.append(Rearrange('... (K dim) -> ... K dim', K=self.K))

    def forward(self, pattern_weights, xt):
        self.get_vel(pattern_weights, xt)

    def get_vel(self, pattern_weights, xt, weights=None):
        p, t = xt[..., :3], xt[..., -1:]
        if weights is None:
            weights, a = self.get_weights(pattern_weights, xt)

        weights_v, weights_w = torch.chunk(weights, 2, dim=-1)
        if self.freegave:
            p = einops.rearrange(p, '... dim -> ... 1 dim')
            v = weights_v + torch.cross(weights_w, p, dim=-1)
            v = einops.einsum(pattern_weights, v, '... k, ... k j -> ... j')
        else:
            v = weights_v + torch.cross(weights_w, p, dim=-1)
        return v

    def get_weights(self, pattern_weights, xt):
        p, t = xt[..., :3], xt[..., -1:]
        if self.freegave:
            embed = self.embedder(t)
        else:
            embed = self.embedder(xt)
        weights_a = self.weight_net(embed)
        weights, a = weights_a[..., :6], weights_a[..., 6:]
        return weights, a * torch.abs(a) / 10

    def rodrigues(self, pattern_weights, weights, dt):
        weights_v, weights_w = torch.chunk(weights, 2, dim=-1)
        if self.freegave:
            weights_w = einops.einsum(pattern_weights, weights_w, '... k, ... k j -> ... j')
        angular_v = weights_w.norm(dim=-1, keepdim=True)
        axis = weights_w / (angular_v + 1e-8)
        axis_x = self.get_cross_product_matrix(axis)
        
        identity = torch.zeros_like(axis_x)
        identity[..., 0, 0] = 1
        identity[..., 1, 1] = 1
        identity[..., 2, 2] = 1

        theta = (angular_v * dt).unsqueeze(-1)
        
        R = identity + torch.sin(theta) * axis_x + (1 - torch.cos(theta)) * (axis_x @ axis_x)
        # R = identity + theta * axis_x
        return R
    
    def get_cross_product_matrix(self, vec):
        x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
        zeros = x * 0.

        b1 = torch.stack([zeros, -z, y], dim=-1)
        b2 = torch.stack([z, zeros, -x], dim=-1)
        b3 = torch.stack([-y, x, zeros], dim=-1)

        return torch.stack([b1, b2, b3], dim=-2)
