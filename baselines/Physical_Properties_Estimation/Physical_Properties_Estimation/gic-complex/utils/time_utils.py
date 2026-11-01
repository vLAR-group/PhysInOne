import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from utils.rigid_utils import exp_se3


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class TimeEncoding(nn.Module):
    def __init__(self, t_multires=6, timenet=True, time_out=30):
        super(TimeEncoding, self).__init__()
        # self.t_multires = 6 if is_blender else 10
        self.embed_time_fn, self.time_input_ch = get_embedder(t_multires, 1)

        if timenet:
            self.timenet = nn.Sequential(
                nn.Linear(self.time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, time_out)
            )
            self.time_input_ch = time_out
        else:
            self.timenet = None
    
    def forward(self, t):
        t_emb = self.embed_time_fn(t)
        if self.timenet:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        return t_emb


class PositionEncoding(nn.Module):
    def __init__(self, x_multires=10):
        super(PositionEncoding, self).__init__()
        self.embed_fn, self.xyz_input_ch = get_embedder(x_multires, 3)

    def forward(self, x):
        x_emb = self.embed_fn(x)
        return x_emb


class MotionBasis(nn.Module):
    def __init__(self, time_input_ch, num_basis=10, D=8, W=256, num_attribute=10):
        super(MotionBasis, self).__init__()
        self.skips = [D // 2]
        self.num_basis = num_basis
        self.num_attribute = num_attribute

        self.linear_t = nn.ModuleList(
            [nn.Linear(time_input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + time_input_ch, W)
                for i in range(D - 1)]
        )

        for i in range(self.num_basis):
            self.__setattr__(f'basis{i}', nn.Linear(W, self.num_attribute))

    def forward(self, t_emb):
        h = t_emb
        for i, l in enumerate(self.linear_t):
            h = self.linear_t[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([t_emb, h], -1)
        basis = []
        for i in range(self.num_basis):
            basis.append(self.__getattr__(f'basis{i}')(h))
        basis = torch.stack(basis, dim=-1)
        return basis


class CoefficientNet(nn.Module):
    def __init__(self, xyz_input_ch, time_input_ch, num_basis=10, num_coeff_set_per_basis=1, W=256, D=1, num_attribute=10, softmax=True):
        super(CoefficientNet, self).__init__()
        self.num_basis = num_basis
        self.num_attribute = num_attribute  # 3 + 4 + 3
        self.num_coeff_set_per_basis = num_coeff_set_per_basis
        self.softmax = softmax

        self.linear_x = nn.ModuleList(
            [nn.Linear(xyz_input_ch + time_input_ch, W)] +  
            [nn.Linear(W, W) for _ in range(D)] + 
            [nn.Linear(W, self.num_basis*self.num_coeff_set_per_basis*self.num_attribute)]
        )

    def forward(self, x_emb, t_emb):
        Np = x_emb.shape[0]
        Nt = t_emb.shape[0]
        x_emb = x_emb[None].expand(Nt, -1, -1)
        t_emb = t_emb[:, None].expand(-1, Np, -1)
        c = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear_x):
            c = self.linear_x[i](c)
            c = F.relu(c) if i != len(self.linear_x) - 1 else c
        if self.softmax:
            c = torch.softmax(c, dim=1)
        return c


class DeformNetwork(nn.Module):
    def __init__(self, 
        x_multires=10, 
        t_multires=6, timenet=True, time_out=30, 
        num_basis=10, num_coeff_set_per_basis=10, channel_mb=256, depth_mb=8, 
        channel_cn=256, depth_cn=1, softmax=False, num_attribute=10, dxyz_scale=1.0, 
    ): 
        super(DeformNetwork, self).__init__()
        self.pe = PositionEncoding(x_multires)
        self.te = TimeEncoding(t_multires, timenet=timenet, time_out=time_out)
        self.mb = MotionBasis(self.te.time_input_ch, num_basis=num_basis, 
                              W=channel_mb, D=depth_mb, num_attribute=num_attribute)
        self.cn = CoefficientNet(self.pe.xyz_input_ch, self.te.time_input_ch, 
                                 num_basis=num_basis, num_coeff_set_per_basis=num_coeff_set_per_basis, 
                                 W=channel_cn, D=depth_cn, softmax=softmax, num_attribute=num_attribute)
        self.dxyz_scale = dxyz_scale
    
    def forward_embs(self, x, t):
        x_emb = self.pe(x)
        t_emb = self.te(t)
        return x_emb, t_emb
    
    def forward_coeff(self, x, t):
        x_emb, t_emb = self.forward_embs(x, t)
        c = self.cn(x_emb, t_emb)
        return c
    
    def forward_motion_basis(self, t):
        t_emb = self.te(t)
        basis = self.mb(t_emb)
        return basis

    def forward(self, x, t):
        x_emb, t_emb = self.forward_embs(x, t)
        basis = self.mb(t_emb)
        c = self.cn(x_emb, t_emb)
        c_ = rearrange(c, 
                       'nt np (d1 d2 d3) -> nt np d1 d2 d3', 
                       d1=self.cn.num_coeff_set_per_basis, 
                       d2=self.cn.num_attribute, 
                       d3=self.cn.num_basis)
        basis = rearrange(basis, '(nt np d1) d2 d3 -> nt np d1 d2 d3', np=1, d1=1)
        # flag = torch.zeros_like(basis).to(basis)
        # flag[..., 9] = 1
        # basis = basis * flag
        attributes = torch.mean(c_ * basis, dim=2)  # mean on coefficient set
        # attributes = torch.sum(attributes, dim=-1)  # sum on basis
        attributes = torch.mean(attributes, dim=-1)  # mean on basis
        d_xyz = attributes[..., 0:3].squeeze(dim=0) * self.dxyz_scale
        if self.cn.num_attribute == 10:
            rotation = attributes[..., 3:7].squeeze(dim=0)
            scaling = attributes[..., 7:10].squeeze(dim=0)
        elif self.cn.num_attribute == 4:  # isotropic gaussian
            rotation = 0.0
            scaling = attributes[..., 3:4].squeeze(dim=0)
        else:
            raise ValueError('# attribute is either 10 or 4.')
        return d_xyz, rotation, scaling


class Deformable3DGaussians(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False):
        super(Deformable3DGaussians, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 1)

    def forward(self, x, t):
        N = x.shape[0]
        t = t.expand(N, -1)
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling