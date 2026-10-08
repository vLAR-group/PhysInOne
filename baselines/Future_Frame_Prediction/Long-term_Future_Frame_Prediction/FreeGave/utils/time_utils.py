import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3
from utils.general_utils import build_rotation
import einops


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


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, hyper_ch=8, multires=10, is_blender=False, is_6dof=False, gated=True):
        print(f"DeformNetwork is blender {is_blender}")
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.hyper_ch = hyper_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, input_ch)
        self.input_ch = xyz_input_ch + time_input_ch + hyper_ch

        # Better for D-NeRF Dataset
        self.time_out = 30

        self.timenet = nn.Sequential(
            nn.Linear(time_input_ch, W), nn.ReLU(inplace=True),
            nn.Linear(W, self.time_out))

        self.linear = nn.ModuleList(
            [nn.Linear(xyz_input_ch + self.time_out + hyper_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out + hyper_ch, W)
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
        self.gaussian_scaling = nn.Linear(W, 3)

        self.gated = gated
        # self.gate_func = lambda t: torch.tanh(20 * t ** 2)
        self.gate_func = nn.Sequential(
            nn.Linear(hyper_ch, W), nn.ReLU(inplace=False),
            nn.Linear(W, 1), nn.Sigmoid()
        )

    def get_feature(self, x, t, motion_code):
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb, motion_code], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, motion_code, h], -1)
        if self.gated:
            gate = self.gate_func(motion_code)
        else:
            gate = None
        return h, gate

    def get_gate(self, motion_code):
        gate = self.gate_func(motion_code)
        return gate

    def get_translation(self, x, t, motion_code):
        h, gate = self.get_feature(x, t, motion_code)
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

        if self.gated:
            # gate = self.gate_func(t)
            d_xyz = d_xyz * gate
        return d_xyz

    def forward(self, x, t, motion_code):
        h, gate = self.get_feature(x, t, motion_code)

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
        # scaling = 0.0
        rotation = self.gaussian_rotation(h)

        if self.gated:
            # gate = self.gate_func(t)
            d_xyz = d_xyz * gate
            # rotation[..., 1:] = rotation[..., 1:] * gate
            rotation = rotation * gate
            rotation[..., 1:] = rotation[..., 1:] + 1

            scaling = scaling * gate

        rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)

        return d_xyz, rotation, scaling


class CodeField(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=8, multires=10):
        super(CodeField, self).__init__()
        self.D = D
        self.W = W
        self.skips = [D // 2]

        self.embed_fn, xyz_input_ch = get_embedder(multires, input_ch)

        self.linear = nn.ModuleList(
            [nn.Linear(xyz_input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch, W)
                for i in range(D - 1)]
        )
        self.output = nn.Sequential(
            nn.Linear(W, output_ch),
            # nn.Softmax(dim=-1)
        )

        self.seg = nn.Sequential(
            nn.Linear(output_ch, output_ch * 4),
            nn.ReLU(inplace=False),
            nn.Linear(output_ch * 4, output_ch * 4),
            nn.ReLU(inplace=False),
            nn.Linear(output_ch * 4, output_ch),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, h], -1)
        motion_code = self.output(h)
        return motion_code
