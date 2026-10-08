# the code from https://github.com/lucidrains/video-diffusion-pytorch
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial, reduce, lru_cache
from operator import mul
import numpy as np
from einops import rearrange
from einops_exts import rearrange_many

from rotary_embedding_torch import RotaryEmbedding
from timm.models.layers import DropPath, trunc_normal_
from model.BaseDM_adaptor.text import BERT_MODEL_DIM


# helpers functions
def exists(x):
    return x is not None


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# relative positional bias
class RelativePositionBias(nn.Module):
    def __init__(
            self,
            heads=8,
            num_buckets=32,
            max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim, use_deconv=True, padding_mode="reflect"):
    if use_deconv:
        return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            nn.Conv3d(dim, dim, (1, 3, 3), (1, 1, 1), (0, 1, 1), padding_mode=padding_mode)
        )


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)



class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x
    

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias,
    ):
        b, n, device = x.shape[0], x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # split out heads
        q, k, v = rearrange_many(qkv, 'b m n (h d) -> (b m) h n d', h=self.heads)
        # scale
        q = q * self.scale
        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # out = xops.memory_efficient_attention(q, k, v)

        # similarity
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        # relative positional bias
        if exists(pos_bias):
            sim = sim + pos_bias

        # numerical stability
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '(b m) h n d -> b m n (h d)', b=b)
        return self.to_out(out)
    

    
class AttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None,
        norm_layer=nn.LayerNorm,
        drop=0.,
        act_layer=nn.GELU,
        drop_path=0.,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb)

    def forward(self, x, pos_bias=None):
        r = x
        x = self.norm(x)
        x = self.attn(x, pos_bias)
        x = r + x
        return x

def get_window_size(x_size, window_size, shift_size):
    use_window_size = list(window_size)
    use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)
    
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, dim_head, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., rotary_emb=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = qk_scale or dim_head ** -0.5
        self.rotary_emb = rotary_emb
        hidden_dim = dim_head * num_heads

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, hidden_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class STWAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        window_size=(2,4,4),
        shift_size=(0,0,0),
        heads=8,
        dim_head=32,
        rotary_emb=None,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.attn = WindowAttention3D(dim, window_size=window_size, num_heads=heads, dim_head=dim_head, rotary_emb=rotary_emb)

    def forward(self, x):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        mask_matrix = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(self.dim_head*self.heads,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x



class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y

class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

# class MotionAdaptor(nn.Module):
#     def __init__(self, channel_in, channel_hid, channel_out, N_T=4, incep_ker = [3,5,7,11], groups=8):
#         super(MotionAdaptor, self).__init__()

#         self.N_T = N_T
#         enc_layers = []
        
#         # channel in->channel hid
#         enc_layers.append(Inception(channel_in, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        
#         # channel hid->channel hid
#         for _ in range(1, N_T):
#             enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))

#         dec_layers = []
        
#         # channel hid->channel hid
#         dec_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        
#         # channel hid + skip -> channel hid
#         for _ in range(1, N_T-1):
#             dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
            
#         # channel hid + skip -> channel output
#         dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_out, incep_ker=incep_ker, groups=groups))

#         self.enc = nn.Sequential(*enc_layers)
#         self.dec = nn.Sequential(*dec_layers)

#     def forward(self, x):
#         B, C, T, H, W = x.shape
#         x = rearrange(x, 'b c t h w -> b (t c) h w')

#         # encoder
#         skips = []
#         z = x
#         for i in range(self.N_T):
#             z = self.enc[i](z)
#             if i < self.N_T - 1:
#                 skips.append(z)

#         # decoder
#         z = self.dec[0](z)
#         for i in range(1, self.N_T):
#             z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

#         y = rearrange(z, 'b (t c) h w -> b c t h w', c=C)
#         return y

def compute_layer(tm, tp, l=None):
    factor = (tp+1)/tm
    num_layers = max(1, int(math.ceil(math.log2(factor))))
    num_layers = l if l else num_layers
    num_frames = (2**num_layers - 1)*tm
    return num_layers, num_frames
    
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class adaptor(nn.Module):
    def __init__(self, dim, num_layer=2):
        super(adaptor, self).__init__()
        self.predictor = Residual(PreNorm(dim, nn.Conv3d(dim,dim,1)))
        self.extrapolators = nn.ModuleList([])
        for _ in range(num_layer):
            self.extrapolators.append(
                Residual(zero_module(nn.Conv3d(dim,dim, kernel_size=(1,3,3), padding=(0,1,1), bias=False)))
                # Residual(PreNorm(dim, zero_module(nn.Conv3d(dim,dim,1, bias=False)))),
                # Residual(PreNorm(dim, zero_module(nn.Conv3d(dim,dim,1, bias=False)))),
            )
        
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 5)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1, 1)
        return feat_mean, feat_std
    
    def forward(self, xm):
        tm = xm.size(2)
        xm = self.predictor(xm)
        x = xm
        for extrapolator in self.extrapolators:
            r = x
            x_m, x_v = self.calc_mean_std(x)
            x_h = (x - x_m) / x_v
            x_h = extrapolator(x_h)
            # x_m = m_estimator(x_h) + x_m
            # x_v = (1 + v_estimator(x_h)) * x_v
            x = x_h * x_v + x_m
            x = torch.cat([r,x],dim=2)
        return x[:,:,tm:]

class MotionAdaptor(nn.Module):
    def __init__(self, dim, tc, tp, l=None):
        super(MotionAdaptor, self).__init__()
        self.tm = tc
        self.tp = tp
        self.dim = dim
        num_layers, num_frames = compute_layer(self.tm, self.tp, l)
        self.adaptors = adaptor(dim, num_layers)
        self.Tmodulator = nn.Conv2d(dim*num_frames, dim*self.tp, 1)
        self.fuser = PreNorm(dim*2, nn.Conv3d(dim*2,dim,1))
    
    def forward(self, x):
        xm, xp = x[:,:,:self.tm], x[:,:,self.tm:]
        xm2p = self.adaptors(xm)
        xm2p_t = rearrange(xm2p, 'N C T H W->N (T C) H W')
        xm2p_t = self.Tmodulator(xm2p_t)
        xm2p = rearrange(xm2p_t, 'N (T C) H W->N C T H W', T=self.tp)
        
        xm2p = torch.cat([xm2p, xp], dim=1)
        # print(xm2p.shape, x.shape, )
        xp = self.fuser(xm2p) + xp
        x = torch.cat([xm, xp], dim=2)
        return x
    
class ScaledDotProductAttention(nn.Module):
    

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value), attention

class MultiHeadAttentionOp(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttentionOp, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attn = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y, attn

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

class TrajWarp(nn.Module):
    def __init__(self, dim, tc, tp, heads=8, bias=True):
        super(TrajWarp, self).__init__()
        self.tm = tc
        self.tp = tp
        self.cross_att = MultiHeadAttentionOp(in_features=dim, head_num=heads)
        self.fuser = nn.Conv3d(dim*2,dim,1)
        self.down = nn.MaxPool3d((1,2,2),(1,2,2))

    def forward(self, xp, f):
        fm, fp = f[:,:,:self.tm], f[:,:,self.tm:]
        h, w = fp.shape[3:]
        xp = self.down(xp)
        
        fm_reshape = rearrange(fm, 'n c t h w->n (t h w) c')
        xp_reshape = rearrange(xp, 'n c t h w->n (t h w) c')
        fm2p, att_map = self.cross_att(q=xp_reshape, k=fm_reshape, v=fm_reshape)
        fm2p = rearrange(fm2p,'n (t h w) c->n c t h w',t=self.tp,h=h,w=w)

        fp = torch.cat([fp,fm2p], dim=1)
        fp = self.fuser(fp)
        f = torch.cat([fm, fp], dim=2)
        
        return f

# class MotionAdaptor(nn.Module):
#     def __init__(self, dim, tc, tp, heads=8, bias=True):
#         super(MotionAdaptor, self).__init__()
#         self.tm = tc-1
#         self.tp = tp
#         self.dim = dim
#         self.norm = nn.LayerNorm(dim)
#         self.heads = heads
#         self.linear_q = nn.Linear(dim, dim, bias)
#         self.linear_k = nn.Linear(dim, dim, bias)
#         self.linear_v = nn.Linear(dim, dim, bias)
#         self.linear_o = nn.Linear(dim, dim, bias)
#         # num_layers, num_frames = compute_layer(self.tm, self.tp)
#         # self.adaptors = adaptor(dim, num_layers)
#         # self.Tmodulator = nn.Conv2d(dim*num_frames, dim*self.tp, 1)
#         # self.fuser = PreNorm(dim*2, nn.Conv3d(dim*2,dim,1))
    
#     def forward(self, x):
#         xm, xp = x[:,:,:self.tm], x[:,:,self.tm:]
#         h, w = xm.shape[3:]

#         r = xp
#         xm_reshape = rearrange(xm, 'n c t h w->n (t h w) c')
#         xp = rearrange(xp, 'n c t h w->n (t h w) c')
#         xp = self.norm(xp)
#         qkv = self.linear_q(xp), self.linear_k(xm_reshape), self.linear_v(xm_reshape)
#         q, k, v = rearrange_many(qkv, 'b k (h d) -> b k h d', h=self.heads)
#         xp = xops.memory_efficient_attention(q, k, v)
        
#         xp = rearrange(xp, 'n (t h w) head d->n (head d) t h w', t=self.tp, h=h, w=w)
#         xp = xp + r
#         x = torch.cat([xm, xp], dim=2)
#         return x

# model
class Unet3D(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim=None,
            out_grid_dim=2,
            out_conf_dim=1,
            window_size= (4, 4, 4),
            dim_mults=(1, 2, 4),
            channels=3,
            cond_channels=3,
            attn_heads=8,
            attn_dim_head=16,
            use_bert_text_cond=False,
            init_dim=None,
            init_kernel_size=7,
            resnet_groups=8,
            use_final_activation=False,
            learn_null_cond=False,
            use_deconv=True,
            padding_mode="zeros",
            cond_num=0,
            pred_num=0,
            l=None,
            framesize=32,
    ):
        self.tc = cond_num
        self.tp = pred_num
        print(dim_mults)

        super().__init__()
        self.null_cond_mask = None
        self.channels = channels
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        
        # temporal attention and its relative positional encoding
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)
        # TODO: 
        temporal_attn = lambda dim: EinopsToAndFrom('b c t h w', 'b (h w) t c',
                                                    AttentionLayer(dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))
        m_adaptor = lambda dim: MotionAdaptor(dim, tc=cond_num, tp=pred_num, l=l)
        
        # initial conv
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        
        self.init_conv = nn.Conv3d(channels, init_dim, kernel_size=(1, init_kernel_size, init_kernel_size), padding=(0, init_padding, init_padding))
        self.init_noise_conv = nn.Conv3d(3, 256, kernel_size=(1, init_kernel_size, init_kernel_size), padding=(0, init_padding, init_padding))
        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))
        self.cond_temporal_attn = Residual(PreNorm(256, temporal_attn(256)))
        self.cond_adaptor = m_adaptor(256)
        # self.init_traj = TrajWarp(256, self.tc, self.tp)

        # dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # time conditioning
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        # modified by nhm
        self.learn_null_cond = learn_null_cond
        if self.learn_null_cond:
            self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None
        else:
            self.null_cond_emb = torch.zeros(1, cond_dim).cuda() if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.motion_enc = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_motion_cond = partial(ResnetBlock, groups=resnet_groups, time_emb_dim=cond_dim)
        
        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass_motion_cond(dim_in, dim_out),
                Residual(PreNorm(dim_out, STWAttentionLayer(dim_out, window_size=self.window_size, shift_size=self.shift_size, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))),
                block_klass_motion_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, STWAttentionLayer(dim_out, window_size=self.window_size, heads=attn_heads, dim_head=attn_dim_head,rotary_emb=rotary_emb))),
                m_adaptor(dim_out) if ind>1 else nn.Identity(),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_motion_cond(mid_dim, mid_dim)
        self.mid_attn1 = Residual(PreNorm(mid_dim, STWAttentionLayer(mid_dim, window_size=self.window_size, shift_size=self.shift_size, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb)))
        self.mid_block2 = block_klass_motion_cond(mid_dim, mid_dim)
        self.mid_attn2 = Residual(PreNorm(mid_dim, STWAttentionLayer(mid_dim, window_size=self.window_size, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb)))
        self.mid_adaptor = m_adaptor(mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass_motion_cond(dim_out * 2, dim_in),
                Residual(PreNorm(dim_in, STWAttentionLayer(dim_in, window_size=window_size, shift_size=self.shift_size, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))),
                block_klass_motion_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, STWAttentionLayer(dim_in, window_size=self.window_size, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))),
                m_adaptor(dim_in) if ind>1 else nn.Identity(),
                Upsample(dim_in, use_deconv, padding_mode) if not is_last else nn.Identity()
            ]))
        
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_grid_dim, 1)
        )

        # added by nhm
        self.use_final_activation = use_final_activation
        
        if self.use_final_activation:
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

        # added by nhm for predicting occlusion mask
        self.occlusion_map = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_conf_dim, 1)
        )

    def forward_with_cond_scale(self, *args, cond_scale=2.,  **kwargs ):
        if cond_scale == 0:
            null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
            return null_logits

        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, time, cond_frames, cond_fea=None, cond=None, null_cond_prob=0., none_cond_mask=None):
        # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device
        tc, tp = cond_frames.shape[2], x.shape[2]
        x = torch.cat([cond_frames, x], dim=2)
        assert tc == self.tc
        assert tp == self.tp
        assert cond_fea.shape[2] == tc+tp
        time_rel_pos_bias   = self.time_rel_pos_bias(tc+tp, device=x.device)
        # b c t h w
        # x = torch.cat([x, cond_frames], dim=2)
        x = self.init_noise_conv(x)
        # x = self.init_adaptor(x)
        # cond_fea = self.init_traj(x[:,:,tc:], cond_fea)
        cond_fea = self.cond_adaptor(cond_fea)
        cond_fea = self.cond_temporal_attn(cond_fea, pos_bias=time_rel_pos_bias)
        
        if not cond_fea is None:
            cond_fea = rearrange(cond_fea, 'n c t h w->(n t) c h w')
            cond_fea = F.interpolate(cond_fea, size=x.shape[-2:], mode='bilinear')
            cond_fea = rearrange(cond_fea, '(n t) c h w->n c t h w',t=self.tc+self.tp)
            x = torch.cat([x, cond_fea], dim=1)


        x = self.init_conv(x)
        r = x.clone()
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)


        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance
        if self.has_cond:
            batch, device = x.shape[0], x.device
            self.null_cond_mask = prob_mask_like((batch,), null_cond_prob, device=device)
            if none_cond_mask is not None:
                self.null_cond_mask = torch.logical_or(self.null_cond_mask, torch.tensor(none_cond_mask).cuda())
            cond = torch.where(rearrange(self.null_cond_mask, 'b -> b 1'), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim=-1)

        h = []

        for block1, STW_attn1 ,block2, STW_attn2, adaptor, downsample in self.downs:
            x = block1(x, t)
            x = STW_attn1(x)
            x = block2(x, t)
            x = STW_attn2(x)
            x = adaptor(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn1(x)
        x = self.mid_block2(x, t)
        x = self.mid_attn2(x)
        x = self.mid_adaptor(x)

        for block1, STW_attn1 ,block2, STW_attn2, adaptor, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = STW_attn1(x)
            x = block2(x, t)
            x = STW_attn2(x)
            x = adaptor(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x_fin = self.final_conv(x)[:,:,tc:]
        x_occ = self.occlusion_map(x)[:,:,tc:]
        return torch.cat((x_fin, x_occ), dim=1)