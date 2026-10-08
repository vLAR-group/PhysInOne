# the code from https://github.com/lucidrains/video-diffusion-pytorch
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange
from model.BaseDM_adaptor.text import tokenize, bert_embed


# helpers functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])




def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        channels=3,
        timesteps=1000,
        sampling_timesteps=250,
        ddim_sampling_eta=1.,
        loss_type='l1',
        use_dynamic_thres=True,  # from the Imagen paper
        dynamic_thres_percentile=0.9,
        null_cond_prob=0.1
    ):
        super().__init__()
        self.null_cond_prob = null_cond_prob
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        if self.is_ddim_sampling:
            print("using ddim samping with %d steps" % sampling_timesteps)
        self.ddim_sampling_eta = ddim_sampling_eta

        # register buffer helper function that casts float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters
        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_cond, x, cond_fea, t, clip_denoised: bool, cond=None, cond_scale=7.5):
        # x_c = torch.cat([x_cond,x],dim=2)
        epsilon_noise = self.denoise_fn.forward_with_cond_scale(x, t, cond_frames=x_cond, cond=cond, cond_fea=cond_fea, cond_scale=cond_scale)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon_noise)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x_cond, x, cond_fea, t, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x_cond=x_cond, x=x, cond_fea=cond_fea, t=t,
                                                                 clip_denoised=clip_denoised, cond=cond,
                                                                 cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, x_cond, shape, cond_fea, cond=None, cond_scale=1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(x_cond, img, torch.full((b,), i, device=device, dtype=torch.long), cond_fea=cond_fea, cond=cond,
                                cond_scale=cond_scale)

        return img
        # return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, x_cond, cond_fea, cond=None, cond_scale=1., batch_size=16):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls).to(device)

        batch_size = x_cond.shape[0] if exists(x_cond) else batch_size
        image_size = self.image_size
        channels = 3
        num_frames = self.num_frames - x_cond.size(2)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(x_cond, (batch_size, channels, num_frames, x_cond.shape[3], x_cond.shape[4]), cond_fea=cond_fea, cond=cond,
                         cond_scale=cond_scale)

    # add by nhm
    @torch.no_grad()
    def ddim_sample(self, x_cond, shape, cond_fea, cond=None, cond_scale=1., clip_denoised=True):

        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(0., total_timesteps, steps=sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise = self.denoise_fn(
                img,
                time_cond,
                cond_fea=cond_fea,
                cond_frames=x_cond,
                cond=cond)
            x_start = self.predict_start_from_noise(img, t=time_cond, noise=pred_noise)

            if clip_denoised:
                s = 1.
                if self.use_dynamic_thres:
                    s = torch.quantile(
                        rearrange(x_start, 'b ... -> b (...)').abs(),
                        self.dynamic_thres_percentile,
                        dim=-1
                    )

                    s.clamp_(min=1.)
                    s = s.view(-1, *((1,) * (x_start.ndim - 1)))

                # clip by threshold, depending on whether static or dynamic
                x_start = x_start.clamp(-s, s) / s

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = torch.randn_like(img) if time_next > 0 else 0.

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        # img = unnormalize_to_zero_to_one(img)
        return img

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start_cond, x_start_pred, cond_fea, t, cond=None, noise=None, clip_denoised=True, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start_pred))
        x_noisy = self.q_sample(x_start=x_start_pred, t=t, noise=noise)

        pred_noise = self.denoise_fn.forward(x_noisy, t, cond_fea=cond_fea, cond_frames=x_start_cond, cond=cond, null_cond_prob=self.null_cond_prob, none_cond_mask=None, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, pred_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, pred_noise)
        else:
            raise NotImplementedError()

        pred_x0 = self.predict_start_from_noise(x_noisy, t, pred_noise)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(pred_x0, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (pred_x0.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            pred_x0 = pred_x0.clamp(-s, s) / s
        return loss, pred_x0

    def forward(self, x_cond, x_pred, cond_fea, *args, **kwargs):
        b, device, img_size, = x_cond.shape[0], x_cond.device, self.image_size
        # check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # x = normalize_img(x)
        return self.p_losses(x_cond, x_pred, cond_fea, t, cond=None, *args, **kwargs)

