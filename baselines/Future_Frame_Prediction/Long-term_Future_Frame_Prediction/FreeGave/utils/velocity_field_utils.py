import torch
import torch.nn as nn
from functorch import vmap, jacrev
import numpy as np
import einops
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


class VelocityWarpper(nn.Module):

    def __init__(self, vel_net):
        super(VelocityWarpper, self).__init__()
        self.vel_net = vel_net

    def integrate_pos(self, deform_code, pos_init, t1, t2, dt_max, deform=False, rot=False):
        assert ~deform | ~rot
        dt_max = torch.ones_like(t1) * dt_max
        time_offset = t1 - t2
        xyz_prev = pos_init
        t_curr = t1
        unfinished = (time_offset.abs() > 0).squeeze(-1)
        if deform:
            deform_prev = torch.zeros(pos_init.shape[0], 3, 3, dtype=pos_init.dtype, device=pos_init.device)
            deform_prev[:, 0, 0] = 1
            deform_prev[:, 1, 1] = 1
            deform_prev[:, 2, 2] = 1
        elif rot:
            rot_prev = torch.zeros(pos_init.shape[0], 3, 3, dtype=pos_init.dtype, device=pos_init.device)
            rot_prev[:, 0, 0] = 1
            rot_prev[:, 1, 1] = 1
            rot_prev[:, 2, 2] = 1
            identity = torch.eye(3, dtype=pos_init.dtype, device=pos_init.device).unsqueeze(0).expand(pos_init.shape[0], 3, 3)
        while unfinished.any():
            # get time step
            dt = time_offset[unfinished].sign() * torch.minimum(time_offset[unfinished].abs(), dt_max[unfinished])
            # Runge-Kutta 2
            xyzt_prev = torch.cat([xyz_prev[unfinished], t_curr[unfinished]], dim=-1)
            velocity = self.vel_net.get_vel(deform_code[unfinished], xyzt_prev)
            p_mid = xyz_prev[unfinished] - 0.5 * dt * velocity
            t_mid = t_curr[unfinished] - 0.5 * dt
            pt_mid = torch.cat([p_mid, t_mid], dim=-1)
            if deform:
                # def u_func(deform_code, xyzt):
                #     u = self.vel_net.get_vel(deform_code[unfinished], xyzt)
                #     return u, u
                # jac_v, velocity = vmap(jacrev(u_func, argnums=1, has_aux=True))(deform_code[unfinished], pt_mid)
                velocity, jac_v = self.vel_net.get_vel_jac(deform_code[unfinished], pt_mid)
                drotdt = torch.bmm(jac_v[..., :3, :3], deform_prev[unfinished])
                deform_prev[unfinished] = deform_prev[unfinished] - dt.unsqueeze(-1) * drotdt
            elif rot:
                # def u_func(deform_code, xyzt):
                #     u = self.vel_net.get_vel(deform_code, xyzt)
                #     return u, u
                # jac_v, velocity = vmap(jacrev(u_func, argnums=1, has_aux=True))(deform_code[unfinished], pt_mid)
                velocity, jac_v = self.vel_net.get_vel_jac(deform_code[unfinished], pt_mid)
                drot = identity[unfinished] - dt.unsqueeze(-1) * jac_v[..., :3, :3]
                rot_prev[unfinished] = torch.bmm(drot, rot_prev[unfinished])
            else:
                velocity = self.vel_net.get_vel(deform_code, pt_mid)

            xyz_cur = xyz_prev[unfinished] - dt * velocity
            xyz_prev[unfinished] = xyz_cur
            time_offset[unfinished] = time_offset[unfinished] - dt
            t_curr[unfinished] = t_curr[unfinished] - dt
            unfinished = (time_offset.abs() > 0).squeeze(-1)
        if deform:
            return xyz_prev, deform_prev
        elif rot:
            rot_prev = rotation_to_quaternion(rot_prev)
            return xyz_prev, rot_prev
        return xyz_prev

    def get_vel_loss(self, deform_code, xyz, t):
        # TODO: modify
        xyzt = torch.cat([xyz, t], dim=-1)

        def u_func(deform_code, xyzt):
            u = self.vel_net.get_vel(deform_code, xyzt)
            return u, u
        jac, vel = vmap(jacrev(u_func, argnums=1, has_aux=True))(deform_code, xyzt)
        a = self.vel_net.get_acc(deform_code, xyzt)

        # calculate the divergence
        divergence = jac[..., 0, 0] + jac[..., 1, 1] + jac[..., 2, 2]
        # calculate the transport equation
        transport = einops.einsum(jac[..., :3, :3], vel, '... o i, ... i -> ... o') + jac[..., :3, 3] - a

        loss = (
                torch.mean(transport ** 2)
                # + torch.mean(divergence ** 2)
        )
        return loss

    # def get_vel_loss(self, deform_code, xyz, device='cuda', begin=0.):
    #     # TODO: modify
    #     n_pts = deform_code.shape[0]
    #     t = torch.rand(int(n_pts), 1, device=device) * (1 - begin) + begin
    #     xyz_noise = torch.randn_like(xyz) * 0.1
    #     xyzt = torch.cat([xyz + xyz_noise, t], dim=-1)
    #
    #     def u_func(deform_code, xyzt):
    #         u = self.vel_net.get_vel(deform_code, xyzt)
    #         return u, u
    #     jac, vel = vmap(jacrev(u_func, argnums=1, has_aux=True))(deform_code, xyzt)
    #     a = self.vel_net.get_acc(deform_code, xyzt)
    #
    #     # calculate the divergence
    #     divergence = jac[..., 0, 0] + jac[..., 1, 1] + jac[..., 2, 2]
    #     # calculate the transport equation
    #     transport = einops.einsum(jac[..., :3, :3], vel, '... o i, ... i -> ... o') + jac[..., :3, 3] - a
    #
    #     loss = (
    #         torch.mean(divergence ** 2)
    #         + torch.mean(transport ** 2)
    #     )
    #     return loss

    # def get_vel_loss(self, aabb, n_pts=32768., device='cuda', begin=0.):
    #     # TODO: modify
    #     min_corner, max_corner = aabb
    #     points = torch.rand(int(n_pts), 3, device=device) * (max_corner - min_corner) + min_corner
    #     t = torch.rand(int(n_pts), 1, device=device) * (1 - begin) + begin
    #     xyzt = torch.cat([points, t], dim=-1)
    #
    #     def u_func(deform_code, xyzt):
    #         u = self.vel_net.get_vel(deform_code, xyzt)
    #         return u, u
    #     jac, u = vmap(jacrev(u_func, argnums=0, has_aux=True))(xyzt)
    #     vel, a = u[..., :3], u[..., 3:]
    #
    #     # calculate the divergence
    #     divergence = jac[..., 0, 0] + jac[..., 1, 1] + jac[..., 2, 2]
    #     # calculate the transport equation
    #     transport = einops.einsum(jac[..., :3, :3], vel, '... o i, ... i -> ... o') + jac[..., :3, 3] - a
    #
    #     loss = (
    #         torch.mean(divergence ** 2) * 5
    #         + torch.mean(transport ** 2) * 0.1
    #     )
    #     return loss


class GroupLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, group_dim: int) -> None:
        super(GroupLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.group_dim = group_dim
        self.linear = nn.Conv2d(in_dim * group_dim, out_dim * group_dim, kernel_size=(1,1), groups=group_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: [..., group_dim, in_dim]]
        :return:
        """
        x = einops.rearrange(x, '... group in_dim -> ... (group in_dim) 1 1')
        x = self.linear(x)
        x = einops.rearrange(x, '... (group out_dim) 1 1 -> ... group out_dim', out_dim=self.out_dim)
        return x


class SegVel(nn.Module):
    def __init__(self, deform_code_dim=8, hidden_dim=64, layers=4, encode_dim=3):
        super(SegVel, self).__init__()
        self.K = deform_code_dim
        in_dim = 1 + 1 * 2 * encode_dim
        self.embedder = PositionEncoder(encode_dim)
        self.weight_net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        for i in range(layers - 1):
            self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, 6 * self.K)))

        # a_in_dim = 3 + 3 * 2 * encode_dim
        # self.a_weight_net = nn.Sequential(GroupLinear(a_in_dim, hidden_dim, self.K), nn.ReLU())
        # for i in range(4):
        #     self.a_weight_net.append(nn.Sequential(GroupLinear(hidden_dim, hidden_dim, self.K), nn.ReLU()))
        # self.a_weight_net.append(nn.Sequential(GroupLinear(hidden_dim, 6, self.K)))

        self.a_weight_bank = nn.Parameter(torch.randn(self.K, 6) / np.sqrt(self.K))

    def forward(self, deform_code, xt):
        v_basis, a_basis = self.get_basis(xt)
        t_embed = self.embedder(xt[..., -1:])
        weights = self.weight_net(t_embed)
        weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self.K)

        # x_embed = self.embedder(xt[..., :-1])
        # a_weights = self.a_weight_net(einops.rearrange(x_embed, '... dim -> ... group dim', group=self.K))
        v = torch.einsum('...ij,...ki->...kj', v_basis, weights)
        a = torch.einsum('...ij,...ki->...kj', a_basis, self.a_weight_bank)
        v = torch.einsum('...k,...kj->...j', deform_code, v)
        a = torch.einsum('...k,...kj->...j', deform_code, a)
        return torch.cat([v, a], dim=-1)

    def get_vel(self, deform_code, xt):
        v_basis, _ = self.get_basis(xt)
        t_embed = self.embedder(xt[..., -1:])
        weights = self.weight_net(t_embed)
        weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self.K)
        v = torch.einsum('...ij,...ki->...kj', v_basis, weights)
        v = torch.einsum('...k,...kj->...j', deform_code, v)
        return v

    def get_weights(self, deform_code, xt):
        v_basis, _ = self.get_basis(xt)
        t_embed = self.embedder(xt[..., -1:])
        weights = self.weight_net(t_embed)
        weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self.K)
        v = torch.einsum('...k,...kj->...j', deform_code, weights)
        return v

    def get_vel_jac(self, deform_code, xt):
        v_basis, jac_basis = self.get_basis_jac(xt)
        t_embed = self.embedder(xt[..., -1:])
        weights = self.weight_net(t_embed)
        weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self.K)
        v = torch.einsum('...ij,...ki->...kj', v_basis, weights)
        v = torch.einsum('...k,...kj->...j', deform_code, v)
        jac = torch.einsum('...imn,...ki->...kmn', jac_basis, weights)
        jac = torch.einsum('...k,...kmn->...mn', deform_code, jac)
        return v, jac

    def get_acc(self, deform_code, xt):
        _, a_basis = self.get_basis(xt)
        # x_embed = self.embedder(xt[..., :-1])
        # a_weights = self.a_weight_net(einops.rearrange(x_embed, '... dim -> ... group dim', group=self.K))
        a = torch.einsum('...ij,...ki->...kj', a_basis, self.a_weight_ban)
        a = torch.einsum('...k,...kj->...j', deform_code, a)
        return a

    def get_basis(self, xt):
        # x, y, z = xt[..., 0].clamp(-1., 1.), xt[..., 1].clamp(-1., 1.), xt[..., 2].clamp(-1., 1.)
        x, y, z = xt[..., 0], xt[..., 1], xt[..., 2]
        zeros = xt[..., -1] * 0.
        ones = zeros + 1.

        b1 = torch.stack([ones, zeros, zeros], dim=-1)
        b2 = torch.stack([zeros, ones, zeros], dim=-1)
        b3 = torch.stack([zeros, zeros, ones], dim=-1)
        b4 = torch.stack([zeros, z, -y], dim=-1)
        b5 = torch.stack([-z, zeros, x], dim=-1)
        b6 = torch.stack([y, -x, zeros], dim=-1)

        a4 = torch.stack([zeros, -y, -z], dim=-1)
        a5 = torch.stack([-x, zeros, -z], dim=-1)
        a6 = torch.stack([-x, -y, zeros], dim=-1)
        return torch.stack([b1, b2, b3, b4, b5, b6], dim=-2), torch.stack([b1, b2, b3, a4, a5, a6], dim=-2)

    def get_basis_jac(self, xt):
        x, y, z = xt[..., 0], xt[..., 1], xt[..., 2]
        zeros = xt[..., -1] * 0.
        ones = zeros + 1.

        b1 = torch.stack([ones, zeros, zeros], dim=-1)
        b2 = torch.stack([zeros, ones, zeros], dim=-1)
        b3 = torch.stack([zeros, zeros, ones], dim=-1)
        b4 = torch.stack([zeros, z, -y], dim=-1)
        b5 = torch.stack([-z, zeros, x], dim=-1)
        b6 = torch.stack([y, -x, zeros], dim=-1)

        zeros_vec = torch.stack([zeros, zeros, zeros], dim=-1)

        jac_1 = torch.stack([zeros_vec, zeros_vec, zeros_vec], dim=-2)
        jac_4 = torch.stack([zeros_vec, b3, -b2], dim=-2)
        jac_5 = torch.stack([-b3, zeros_vec, b1], dim=-2)
        jac_6 = torch.stack([b2, -b1, zeros_vec], dim=-2)

        # jac_1 = torch.stack([
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        # ], dim=-2)
        # jac_2 = torch.stack([
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        # ], dim=-2)
        # jac_3 = torch.stack([
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        # ], dim=-2)
        # jac_4 = torch.stack([
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, ones], dim=-1),
        #     torch.stack([zeros, -ones, zeros], dim=-1),
        # ], dim=-2)
        # jac_5 = torch.stack([
        #     torch.stack([zeros, zeros, -ones], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([ones, zeros, zeros], dim=-1),
        # ], dim=-2)
        # jac_6 = torch.stack([
        #     torch.stack([zeros, ones, zeros], dim=-1),
        #     torch.stack([-ones, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        # ], dim=-2)

        return torch.stack([b1, b2, b3, b4, b5, b6], dim=-2), torch.stack([jac_1, jac_1, jac_1, jac_4, jac_5, jac_6], dim=-3)


class VelBasis(nn.Module):
    def __init__(self, deform_code_dim=32):
        super(VelBasis, self).__init__()
        encode_dim = 3
        in_dim = 4 + deform_code_dim + 4 * 2 * encode_dim
        hidden_dim = 128
        self.embedder = PositionEncoder(encode_dim)
        self.weight_net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        for i in range(4):
            self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, 6)))

        a_in_dim = 3 + deform_code_dim + 3 * 2 * encode_dim
        self.a_weight_net = nn.Sequential(nn.Linear(a_in_dim, hidden_dim), nn.ReLU())
        for i in range(4):
            self.a_weight_net.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.a_weight_net.append(nn.Sequential(nn.Linear(hidden_dim, 6)))

    def forward(self, deform_code, xt):
        v_basis, a_basis = self.get_basis(xt)
        xt_embed = self.embedder(xt)
        weights = self.weight_net(torch.cat([deform_code, xt_embed], dim=-1))
        x_embed = self.embedder(xt[..., :-1])
        a_weights = self.a_weight_net(torch.cat([deform_code, x_embed], dim=-1))
        v = torch.einsum('...ij,...i->...j', v_basis, weights)
        a = torch.einsum('...ij,...i->...j', a_basis, a_weights)
        return torch.cat([v, a], dim=-1)

    def get_vel(self, deform_code, xt):
        v_basis, _ = self.get_basis(xt)
        xt_embed = self.embedder(xt)
        weights = self.weight_net(torch.cat([deform_code, xt_embed], dim=-1))
        v = torch.einsum('...ij,...i->...j', v_basis, weights)
        return v

    def get_acc(self, deform_code, xt):
        _, a_basis = self.get_basis(xt)
        x_embed = self.embedder(xt[..., :-1])
        a_weights = self.a_weight_net(torch.cat([deform_code, x_embed], dim=-1))
        a = torch.einsum('...ij,...i->...j', a_basis, a_weights)
        return a

    def get_basis(self, xt):
        # x, y, z = xt[..., 0].clamp(-1., 1.), xt[..., 1].clamp(-1., 1.), xt[..., 2].clamp(-1., 1.)
        x, y, z = xt[..., 0], xt[..., 1], xt[..., 2]
        zeros = xt[..., -1] * 0.
        ones = zeros + 1.

        b1 = torch.stack([ones, zeros, zeros], dim=-1)
        b2 = torch.stack([zeros, ones, zeros], dim=-1)
        b3 = torch.stack([zeros, zeros, ones], dim=-1)
        b4 = torch.stack([zeros, z, -y], dim=-1)
        b5 = torch.stack([-z, zeros, x], dim=-1)
        b6 = torch.stack([y, -x, zeros], dim=-1)

        a4 = torch.stack([zeros, -y, -z], dim=-1)
        a5 = torch.stack([-x, zeros, -z], dim=-1)
        a6 = torch.stack([-x, -y, zeros], dim=-1)
        return torch.stack([b1, b2, b3, b4, b5, b6], dim=-2), torch.stack([b1, b2, b3, a4, a5, a6], dim=-2)


class AccBasis(nn.Module):
    def __init__(self, deform_code_dim=32):
        super(AccBasis, self).__init__()
        encode_dim = 2
        in_dim = deform_code_dim + 3 + (deform_code_dim + 3) * 2 * encode_dim
        hidden_dim = 128
        self.a_weight_net = nn.Sequential(PositionEncoder(encode_dim), nn.Linear(in_dim, hidden_dim), nn.ReLU())
        for i in range(4):
            self.a_weight_net.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.a_weight_net.append(nn.Sequential(nn.Linear(hidden_dim, 6)))

    def forward(self, z, x):
        zx = torch.cat([z, x], dim=-1)
        a_basis = self.get_basis(x)
        a_weights = self.a_weight_net(zx)
        a = torch.einsum('...ij,...i->...j', a_basis, a_weights)
        return a

    def get_basis(self, pos):
        # x, y, z = pos[..., 0].clamp(-1., 1.), pos[..., 1].clamp(-1., 1.), pos[..., 2].clamp(-1., 1.)
        x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
        zeros = pos[..., -1] * 0.
        ones = zeros + 1.

        b1 = torch.stack([ones, zeros, zeros], dim=-1)
        b2 = torch.stack([zeros, ones, zeros], dim=-1)
        b3 = torch.stack([zeros, zeros, ones], dim=-1)

        a4 = torch.stack([zeros, -y, -z], dim=-1)
        a5 = torch.stack([-x, zeros, -z], dim=-1)
        a6 = torch.stack([-x, -y, zeros], dim=-1)
        return torch.stack([b1, b2, b3, a4, a5, a6], dim=-2)
