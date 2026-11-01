# coding=utf-8

import torch
import torchvision
import numpy as np
import taichi as ti
import torch.nn as nn
from simulator import MPMSimulator
from gaussian_renderer import render
from utils.general_utils import get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
import os

def constraint(x, bound):
    # return x
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return y_scale * torch.tanh(x_scale * x) + (bound[0]+ y_scale)

def constraint_inv(y, bound):
    # return y
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return torch.arctanh((y - (bound[0] + y_scale)) / y_scale) / x_scale

@ti.data_oriented
class Estimator(torch.nn.Module):
    
    velocity_stage = 0
    physical_params_stage = 1

    def __init__(self, phys_args, dtype, gts: list, init_vol, 
                 surface_index=None, cuda_chunk_size=100, dynamic_scene=None, 
                 image_scale=1.0, pipeline=None, image_op=None):
        super(Estimator, self).__init__()
        self.pipeline = pipeline
        self.image_op = image_op
        self.image_scale = image_scale
        self.stage = ti.field(ti.int32, shape=())
        self.stage[None] = -1
        self.args = phys_args
        self.material = phys_args.material
        self.voxel_size = phys_args.voxel_size
        self.dtype = ti.f64 if dtype == 'float64' else ti.f32
        frame_dt = self.frame_dt = 1.0 / phys_args.fps
        dt = frame_dt / phys_args.mpm_iter_cnt
        gravity = phys_args.gravity
        self.img_loss = getattr(phys_args, 'img_loss', True)
        self.geo_loss = getattr(phys_args, 'geo_loss', True)
        self.w_img = torch.tensor(getattr(phys_args,"w_img", 0.0), device=init_vol.device)
        self.w_alp = torch.tensor(getattr(phys_args,"w_alp", 0.0), device=init_vol.device)
        self.w_geo = ti.field(ti.f32, shape=(), needs_grad=True)
        self.w_geo[None] = getattr(phys_args,"w_geo", 1.0)

        self.num_particles = ti.field(ti.i32, shape=())
        
        self.dx = ti.field(self.dtype, shape=())
        self.inv_dx = ti.field(self.dtype, shape=())
        self.frame_dt = frame_dt
        self.gts = gts
        self.max_f = len(gts)
        max_f = self.max_f if self.max_f > 0 else 1
        particles_count, _ = init_vol.shape
        print(f'obj particles count: {particles_count}')

        surface_count = self.get_gts_surface_count(gts)
        surface_count = surface_count if surface_count > 0 else 1
        self.num_particles[None] = particles_count
        self.device = init_vol.device
        self.init_vol = init_vol

        # max mpm surface particles count 
        self.sim_surface_particles_cnt = sim_surface_particles_cnt = self.get_surface_particles_cnt(surface_index)#surface_index.shape[0] 

        self.sim_surface_index = ti.field(dtype=ti.i32, shape=(max_f, sim_surface_particles_cnt))
        self.sim_surface_cnt = ti.field(dtype=ti.i32, shape=max_f) # mpm surface particles count per frame

        self.gt = ti.Vector.field(n=3, dtype=self.dtype, shape=(max_f, surface_count), needs_grad=True)

        self.match_indices_gt2sim = ti.field(ti.i32, shape=(max_f, surface_count))
        self.match_indices_sim2gt = ti.field(ti.i32, shape=(max_f, self.sim_surface_particles_cnt))

        # self.match_indices_trajectory = ti.field(ti.i32, shape=(sim_surface_particles_cnt))
        self.gt2sim_err_cnt = ti.field(ti.i32, shape=max_f)
        self.sim2gt_err_cnt = ti.field(ti.i32, shape=max_f)
        
        self.num_particles_surface = ti.field(ti.i32, shape=(max_f)) # gt surface particles count per frame
        self.num_particles[None] = particles_count
        # self.num_particles_surface[None] = surface_count

        self.load_gts(gts)
        self.load_all_sim_surfaces(surface_index, max_f)

        self.particle_rho = ti.field(dtype=self.dtype, needs_grad=True)
        self.loss = ti.field(self.dtype, shape=(), needs_grad=True)
        
        self.grid_particles_density = ti.field(self.dtype, needs_grad=True)
        grid_size = 4096
        offset = tuple(-grid_size // 2 for _ in range(3))
        grid_block_size = 128
        leaf_block_size = 4

        grid = self.grid_observe = ti.root.pointer(ti.ijk, grid_size // grid_block_size)
        block = grid.pointer(ti.ijk, grid_block_size // leaf_block_size)
        block.dense(ti.ijk, leaf_block_size).place(self.grid_particles_density, self.grid_particles_density.grad, offset=offset)

        particle_chunk_size = 2**14
        self.particle = ti.root.dynamic(ti.i, 2**30, particle_chunk_size)
        self.particle.place(self.particle_rho, self.particle_rho.grad)

        self.global_rho = nn.Parameter(torch.tensor(phys_args.rho, device=self.device).float())
        self.init_rhos = None
        self.init_yield_stress = None
        self.init_plastic_viscosity = None
        self.init_friction_alpha = None
        self.init_cohesion = None
        self.E = nn.Parameter(torch.tensor(getattr(phys_args, "init_E", 0.0), device=self.device))
        self.yield_stress = nn.Parameter(torch.tensor([getattr(phys_args, "init_yield_stress", 0.0)], device=self.device))
        self.plastic_viscosity = nn.Parameter(torch.tensor([getattr(phys_args, "init_plastic_viscosity", -1e6)], device=self.device))
        self.friction_alpha = nn.Parameter(torch.tensor([getattr(phys_args, "init_friction_alpha", 0.0)], device=self.device))
        
        self.cohesion = nn.Parameter(torch.tensor([getattr(phys_args, "init_cohesion", 0.0)], device=self.device))
        self.global_mu = nn.Parameter(torch.tensor([np.log10(getattr(phys_args, "mu", 1.0))], device=self.device).float())
        self.global_kappa = nn.Parameter(torch.tensor([np.log10(getattr(phys_args, "kappa", 1.0))], device=self.device).float())

        self.nu_bound = getattr(phys_args, "nu_bound", [-0.99, 0.5])
        self.nu = nn.Parameter(constraint_inv(torch.tensor(getattr(phys_args, "init_nu", 0.0), device=self.device), self.nu_bound))
        self.init_vel = nn.Parameter(torch.tensor(phys_args.init_vel, device=self.device))
        self.init_omega = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=self.device))
        self.simulator = MPMSimulator(dtype=self.dtype, dt=dt, frame_dt=frame_dt, n_particles=self.num_particles,
                                      material=phys_args.material, dx=self.dx, inv_dx=self.inv_dx, 
                                      particle_layout=self.particle, args=phys_args, gravity=gravity, 
                                      cuda_chunk_size=cuda_chunk_size)
        
        for collider_type, collider in phys_args.bc.items():
            if "ground" in collider_type:
                point, normal, bc_style = collider
                self.simulator.add_surface_collider(point, normal, bc_style)
            elif "cylinder" in collider_type:
                start, end, radius, bc_style = collider
                self.simulator.add_cylinder_collider(start, end, radius, bc_style)
            elif "mesh" in collider_type:
                npy = collider[0]
                meta = collider[1]
                print("[DBG] before register, analytic_collision len =", len(self.simulator.analytic_collision))
                self.simulator.add_mesh_collider_from_files(npy, meta)
                print("[DBG] after register, analytic_collision len =", len(self.simulator.analytic_collision))


        params = []
        for param_name, info in phys_args.params.items():
            if param_name == "Youngs modulus":
                params.append({'params':self.E, 'lr':info.get('init_lr', 0.1), 'name': param_name})
            elif param_name == "Poisson ratio":
                params.append({'params':self.nu, 'lr':info.get('init_lr', 0.025), 'name': param_name})
            elif param_name == "bulk modulus":
                params.append({'params':self.global_kappa, 'lr':info.get('init_lr', 0.1), 'name': param_name})
            elif param_name == "shear modulus":
                params.append({'params':self.global_mu, 'lr':info.get('init_lr', 0.1), 'name': param_name})
            elif param_name == "Yield stress":
                params.append({'params':self.yield_stress, 'lr':info.get('init_lr', 0.1), 'name': param_name})
            elif param_name == "plastic viscosity":
                params.append({'params':self.plastic_viscosity, 'lr':info.get('init_lr', 0.05), 'name': param_name})
            elif param_name == "friction angle":
                params.append({'params':self.friction_alpha, 'lr':info.get('init_lr', 1.0), 'name': param_name})
        # params.append({'params':self.init_omega, 'lr':info.get('init_lr', 1.5), 'name': "omega"})
        self.optimizer = torch.optim.Adam([*params],amsgrad=False)
        self.vel_optimizer = torch.optim.Adam([{'params': self.init_vel, 'lr': phys_args.vel_lr, 'name': 'velocity'}])
        self.lr_schedulers = {}
        for param_name, info in phys_args.params.items():
            if info.get('lr_decay', False):
                lr_init = info.get('init_lr', 0.1)
                lr_final = info.get('final_lr', 0.01)
                max_steps = info.get('max_steps', 60)
                self.lr_schedulers[param_name] = get_expon_lr_func(lr_init=lr_init, lr_final=lr_final, max_steps=max_steps, lr_delay_mult=0.01)
        self.config_id = phys_args.id
        self.pos_grad_seq = []
        self.image_loss = 0.0
        self.views = None
        self.scene = dynamic_scene

    def scene():
        def fget(self):
            return self._scene

        def fset(self, value):
            if value is None:
                return 
            self._scene = value
            views = self.scene.getTrainCameras(scale=self.image_scale)
            t_ls = torch.unique(torch.stack([view.fid for view in views if view.fid >= 0]))
            t_ls, _ = torch.sort(t_ls.cpu())
            all_views = []
            for t in t_ls:
                views_by_t = [v for v in views if torch.abs(v.fid.cpu() - t)<1e-7]
                all_views.append(views_by_t)

            self.views = all_views
        
        return locals()
    scene = property(**scene())

    def zero_grad(self):
        if self.stage[None] == self.velocity_stage:
            self.vel_optimizer.zero_grad()
        elif self.stage[None] == self.physical_params_stage:
            self.optimizer.zero_grad()
    
    def step(self, i):
        if self.stage[None] == self.velocity_stage:
            self.vel_optimizer.step()
        elif self.stage[None] == self.physical_params_stage:
            self.optimizer.step()
            self.update_learning_rate(i)

    def get_optimizer(self):
        if self.stage[None] == self.velocity_stage:
            return self.vel_optimizer
        elif self.stage[None] == self.physical_params_stage:
            return self.optimizer

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in self.lr_schedulers:
                f = self.lr_schedulers[param_group["name"]]
                lr = f(iteration)
                param_group['lr'] = lr

    def set_scene(self, scene):
        self.scene = scene

    def set_stage(self, stage):
        self.stage[None] = stage

    @ti.kernel
    def get_input_grad(self, position_grad: ti.types.ndarray(), 
                             velocity_grad: ti.types.ndarray(),rho_grad: ti.types.ndarray(),
                             mu_grad: ti.types.ndarray(), lam_grad: ti.types.ndarray()):
        for p in range(self.num_particles[None]):
            rho_grad[p] = self.particle_rho.grad[p]
            mu_grad[p] = self.simulator.mu.grad[p]
            lam_grad[p] = self.simulator.lam.grad[p]
            for d in ti.static(range(3)):
                velocity_grad[p, d] = self.simulator.v.grad[p, 0][d]
                position_grad[p, d] = self.simulator.x.grad[p, 0][d]


    def get_nu(self):
        return constraint(self.nu, self.nu_bound)

    def get_surface_particles_cnt(self, surfaces):
        if surfaces is None:
            return self.num_particles[None]
        elif type(surfaces) == type([]):
            cnt = 0
            for idx in range(len(surfaces)):
                surface_p_cnt, _ = surfaces[idx].shape
                if surface_p_cnt > cnt:
                    cnt = surface_p_cnt
            return cnt
        else:
            return surfaces.shape[0]


    @staticmethod
    def get_gts_surface_count(gts):
        count = 0
        for idx in range(len(gts)):
            surface_count, _ = gts[idx].shape
            if surface_count > count:
                count = surface_count
        return count

    def load_gts(self, gts):
        for idx in range(len(gts)):
            surface_count, _ = gts[idx].shape
            self.num_particles_surface[idx] = surface_count
            self.save_gt(idx, gts[idx], surface_count)

    def load_all_sim_surfaces(self, surfaces, max_f):
        if surfaces is None:
            surfaces = torch.arange(self.num_particles[None])
            # self.num_particles[None]
            
        surfaces = surfaces.data.cpu().numpy() 
        cnt = max_f
        if type(surfaces) == type([]):
            cnt = len(surfaces)
        for idx in range(cnt):
            surface = surfaces[idx] if type(surfaces) == type([]) else surfaces
            self.sim_surface_cnt[idx] = surface.shape[0]
            self.load_sim_surface(surface, surface.shape[0], idx)
            

    @ti.kernel
    def load_sim_surface(self, surface_index: ti.types.ndarray(), cnt: ti.int32, f: ti.int32):
        for i in range(cnt):
            self.sim_surface_index[f, i] = surface_index[i]

    def compute_velocities(self):
        cnt = self.init_vol.shape[0]
        velocities = self.init_vel.repeat(cnt).reshape(cnt, -1)
        centroid = self.init_vol.sum(dim=0) / self.init_vol.shape[0]
        omega = self.init_omega.repeat(cnt).reshape(cnt, -1)
        velocities += torch.cross(omega, self.init_vol - centroid)
        return velocities

    def initialize(self):
        torch.cuda.synchronize()
        ti.sync()
        self.pos_grad_seq.clear()
        self.image_loss = 0.0
        self.loss[None] = 0.0
        cnt = self.init_vol.shape[0]
        velocities = self.init_vel.repeat(cnt).reshape(cnt, -1)
        # velocities = self.compute_velocities()

        particles = self.init_vol
        self.dx[None], self.inv_dx[None] = self.voxel_size, 1.0 / self.voxel_size
        self.simulator.reset_dt()
    
        self.compute_particle_vol()
        self.init_rhos = self.global_rho.repeat(self.num_particles[None])
        particle_rho = self.init_rhos

        self.clear_grads()
        if self.material == MPMSimulator.elasticity:
            nu = self.get_nu()
            self.init_mu = (10**self.E) / (2. * (1. + nu)).repeat(cnt)
            self.init_lam = (10**self.E) * nu / ((1. + nu) * (1. - 2. * nu)).repeat(cnt)
        elif self.material == MPMSimulator.von_mises:
            if getattr(self.args, 'init_E', None) and getattr(self.args, 'init_nu', None):
                nu = self.get_nu()
                E = 10**self.E
                mu =  E / (2. * (1. + nu))
                lam = E * nu / ((1. + nu) * (1. - 2. * nu))
                self.init_mu = mu.repeat(cnt)
                self.init_lam = lam.repeat(cnt)
            elif getattr(self.args, 'kappa', None) and getattr(self.args, 'mu', None):
                self.init_mu = (10 ** self.global_mu).repeat(cnt)
                lam = 10 ** self.global_kappa - 2./3. * 10 ** self.global_mu
                self.init_lam = lam.repeat(cnt)
        elif self.material == MPMSimulator.viscous_fluid:
            self.init_mu = (10 ** self.global_mu).repeat(cnt)
            lam = 10 ** self.global_kappa - 2./3. * 10 ** self.global_mu
            self.init_lam = lam.repeat(cnt)
        elif self.material == MPMSimulator.drucker_prager:
            nu = self.get_nu()
            self.init_mu =  ((10**self.E) / (2. * (1. + nu))).repeat(cnt)
            self.init_lam = ((10**self.E) * nu / ((1. + nu) * (1. - 2. * nu))).repeat(cnt)

        yield_stress = 10 ** self.yield_stress
        eta = 10 ** self.plastic_viscosity
        sin_phi = torch.sin(self.friction_alpha / 180 * np.pi)
        friction_alpha = np.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        cohesion = self.cohesion
        self.init_pos = self.init_vol.clone().requires_grad_(True)
        self.init_velocities = velocities
        self.simulator.cached_states.clear()
        self.from_torch(particles.data.cpu().numpy(), velocities.data.cpu().numpy(), particle_rho.data.cpu().numpy(), self.init_mu.data.cpu().numpy(), self.init_lam.data.cpu().numpy())
        self.compute_particle_mass()
        self.simulator.yield_stress[None] = yield_stress.item()
        self.simulator.plastic_viscosity[None] = eta.item()
        self.simulator.friction_alpha[None] = friction_alpha.item()
        self.simulator.cohesion[None] = cohesion.item()
        self.simulator.cfl_satisfy[None] = True
        torch.cuda.empty_cache()

    @ti.kernel
    def compute_particle_vol(self):
        grid_vol = (self.dx[None] * 0.5) ** 3
        for p in range(self.num_particles[None]):
            self.simulator.p_vol[p] = grid_vol

    def clear_grads(self):
        self.particle_rho.grad.fill(0)
        self.simulator.clear_grads()

    @ti.kernel
    def from_torch(self, particles: ti.types.ndarray(), 
                         velocities: ti.types.ndarray(), 
                         particle_rho: ti.types.ndarray(), 
                         particle_mu: ti.types.ndarray(), 
                         particle_lam: ti.types.ndarray()):
        for p in range(self.num_particles[None]):
            self.particle_rho[p] = particle_rho[p]
            self.simulator.mu[p] = particle_mu[p]
            self.simulator.lam[p] = particle_lam[p]
            self.simulator.p_mass[p] = 0.0
            self.simulator.F[p, 0] = ti.Matrix.identity(self.dtype, 3)
            self.simulator.C[p, 0] = ti.Matrix.zero(self.dtype, 3, 3)
            for d in ti.static(range(3)):
                self.simulator.x[p, 0][d] = particles[p, d]
                self.simulator.v[p, 0][d] = velocities[p, d]

    def succeed(self):
        return self.simulator.cfl_satisfy[None]

    @ti.kernel
    def compute_particle_mass(self):
        for p in range(self.num_particles[None]):
            self.simulator.p_mass[p] = self.particle_rho[p] * self.simulator.p_vol[p]

    @ti.kernel
    def save_gt(self, f: ti.i32, gt: ti.types.ndarray(), count: ti.i32):
        for p in range(count):
            for d in ti.static(range(3)):
                self.gt[f, p][d] = gt[p, d]

    @ti.func
    def compute_distance(self, p1, p2):
        d = ti.math.distance(p1, p2)
        return d

    @ti.kernel
    def update_match_indices_gt2sim(self, f:ti.i32, local_index:ti.i32):
        err = 0.0
        self.gt2sim_err_cnt[f] = self.num_particles_surface[f]
        for i in range(self.num_particles_surface[f]):
            min_value, min_index = ti.math.inf, 0
            for j in range(self.sim_surface_cnt[f]):
                index_ = self.sim_surface_index[f, j]
                d = ti.math.distance(self.simulator.x[index_, local_index], self.gt[f, i])
                old_min_value = ti.atomic_min(min_value, d)
                if min_value != old_min_value:
                    min_index = index_
            err += min_value
            self.match_indices_gt2sim[f, i] = min_index
            if self.gt[f, i].y <= self.voxel_size:
                self.gt2sim_err_cnt[f] -= 1

    @ti.kernel
    def update_match_indices_sim2gt(self, f: ti.i32, local_index: ti.i32):
        self.sim2gt_err_cnt[f] = self.sim_surface_cnt[f]
        for i in range(self.sim_surface_cnt[f]):
            min_value, min_index = ti.math.inf, 0
            for j in range(self.num_particles_surface[f]):
                d = ti.math.distance(self.simulator.x[self.sim_surface_index[f,i], local_index], self.gt[f, j])
                old_min_value = ti.atomic_min(min_value, d)
                if min_value != old_min_value:
                    min_index = j
            self.match_indices_sim2gt[f, i] = min_index
            if not (self.gt[f, min_index].y > self.voxel_size and
                    self.simulator.x[self.sim_surface_index[f, i], local_index].y > self.voxel_size):
                self.sim2gt_err_cnt[f] -= 1

    @ti.kernel
    def compute_loss_gt2sim(self, f:ti.i32, local_index:ti.i32):
        '''
        compute loss: from gt to sim
        '''
        for i in range(self.num_particles_surface[f]):
            index_ = self.match_indices_gt2sim[f, i]
            d = self.compute_distance(self.simulator.x[index_, local_index], self.gt[f, i])

            if self.stage[None] == self.velocity_stage:
                self.loss[None] += d / self.num_particles_surface[f]
            else:
                # physical params stage
                self.loss[None] += d*self.w_geo[None] / self.gt2sim_err_cnt[f] if self.gt[f, i].y > self.voxel_size else 0.0


    @ti.kernel
    def compute_loss_sim2gt(self, f:ti.i32, local_index: ti.i32):
        '''
        compute loss: from sim to gt
        '''
        # cnt = 0
        # loss = 0.0
        for i in range(self.sim_surface_cnt[f]):
            index_ = self.match_indices_sim2gt[f, i]
            x = self.simulator.x[self.sim_surface_index[f, i], local_index]
            d = self.compute_distance(x, self.gt[f, index_])
            if self.stage[None] == self.velocity_stage:
                self.loss[None] += d / self.sim_surface_cnt[f]
            else:
                # physical params stage
                self.loss[None] += d*self.w_geo[None] / self.sim2gt_err_cnt[f] if self.gt[f, index_].y > self.voxel_size and x.y > self.voxel_size else 0.0

    def get_surface(self, f):
        surface = np.full((self.num_particles[None], 3), [0, 255, 0], dtype=np.uint8)

        @ti.kernel
        def extract(f:ti.i32, surface: ti.types.ndarray()):
            local_index = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size
            for i in range(self.sim_surface_cnt[f]):
                index = self.sim_surface_index[f, i]
                surface[index, 0] = ti.cast(255, ti.u8)
                surface[index, 1] = ti.cast(0, ti.u8)
                surface[index, 2] = ti.cast(0, ti.u8)
                # for d in ti.static(range(3)):
                # surface[i, d] = ti.cast(self.simulator.x[index, local_index][d], ti.f32)
        extract(f, surface)
        return surface

    def get_surface_vertics(self, f):
        surface = np.zeros([self.sim_surface_cnt[f], 3], dtype=np.float32)
        color = np.full((self.sim_surface_cnt[f], 3), [0, 255, 0], dtype=np.uint8)
        @ti.kernel
        def extract(f:ti.i32, surface: ti.types.ndarray(), color: ti.types.ndarray()):
            local_index = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size
            for i in range(self.sim_surface_cnt[f]):
                index = self.sim_surface_index[f, i]
                for d in ti.static(range(3)):
                    surface[i, d] = ti.cast(self.simulator.x[index, local_index][d], ti.f32)
            for i in range(self.sim_surface_cnt[f]):
                if surface[i, 1] <= self.voxel_size:
                    color[i, 0] = ti.cast(0, ti.u8)
                    color[i, 1] = ti.cast(0, ti.u8)
                    color[i, 2] = ti.cast(255, ti.u8)
        extract(f, surface, color)
        return surface, color

    def render_forward(self, f, xyz, backward=True):
        gaussians = self.scene.gaussians
        views = self.views[f]
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        d_xyz = xyz - gaussians.get_xyz
        loss_img = torch.tensor(0.0, device=self.device)
        loss_alp = torch.tensor(0.0, device=self.device)
        for view in views:
            results = render(view, gaussians, self.pipeline, background, d_xyz, 0.0, 0.0, False)
            image, alpha = results["render"], results["alpha"]
            gt_image = view.original_image.cuda()
            gt_alpha_mask = view.gt_alpha_mask
            # crop image loss
            mask = torch.logical_or(gt_image.sum(0) != 0, 
                                    image.sum(0) != 0)
            ids = torch.where(mask)
            h_min, h_max, w_min, w_max = ids[0].min(), ids[0].max(), ids[1].min(), ids[1].max()
            h_min, h_max = max(h_min-50, 0), min(h_max+50, image.shape[1])
            w_min, w_max = max(w_min-50, 0), min(w_max+50, image.shape[2])
            image = image[:, h_min:h_max, w_min:w_max]
            gt_image = gt_image[:, h_min:h_max, w_min:w_max]
            alpha = alpha[:, h_min:h_max, w_min:w_max]
            gt_alpha_mask = gt_alpha_mask[:, h_min:h_max, w_min:w_max]

            if self.w_img > 0.0:
                Ll1 = l1_loss(image, gt_image)
                loss_img += (1.0 - self.image_op.lambda_dssim) * Ll1 + self.image_op.lambda_dssim * (1.0 - ssim(image, gt_image))
            if self.w_alp > 0.0:
                L_alpha = l1_loss(alpha, gt_alpha_mask)
                loss_alp += L_alpha
        loss = (self.w_img * loss_img + self.w_alp * loss_alp) / len(views)
        if backward:
            loss.backward()
        with torch.no_grad():
            self.image_loss += loss.detach().cpu()
        self.pos_grad_seq.append(xyz.grad)


    def forward(self, f, img_backward=True):
        particle_pos = np.zeros([self.num_particles[None], 3], dtype=np.float32)

        if f > 0:
            self.simulator.advance(f-1)

        if not self.succeed():
            return particle_pos
        local_index = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size

        if len(self.gts) > 0 and (self.geo_loss or self.stage[None] == self.velocity_stage):
            self.update_match_indices_gt2sim(f, local_index)
            self.update_match_indices_sim2gt(f, local_index)
            self.compute_loss_gt2sim(f, local_index)
            self.compute_loss_sim2gt(f, local_index)

        self.simulator.get_x(f, particle_pos)
        particle_pos =torch.from_numpy(particle_pos).to(self.device).requires_grad_()
        if (f > 0 and self.stage[None] == self.physical_params_stage and self.img_loss) or\
           (f>0 and self.stage[None]==self.velocity_stage and len(self.gts)==0):
            self.render_forward(f, particle_pos, img_backward)
        return particle_pos
    
    @ti.kernel
    def set_pos_grad(self, f:ti.i32, dLdpo: ti.types.ndarray()):
        s = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size
        for p in range(self.num_particles[None]):
            for d in ti.static(range(3)):
                self.simulator.x.grad[p, s][d] += dLdpo[p, d]

    def backward(self, f):
        local_index = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size

        if self.stage[None] == self.physical_params_stage and f > 0 and self.img_loss or (len(self.gts) == 0):
            self.set_pos_grad(f, self.pos_grad_seq[f-1].data.cpu().numpy())
        if self.geo_loss or self.stage[None] == self.velocity_stage:
            self.compute_loss_sim2gt.grad(f, local_index)
            self.compute_loss_gt2sim.grad(f, local_index)
        if f > 0:
            self.simulator.advance_grad(f-1)
        else:
            self.compute_particle_mass.grad()
            dtype=np.float32
            velocity_grad = np.zeros([self.num_particles[None], 3], dtype=dtype)
            position_grad = np.zeros([self.num_particles[None], 3], dtype=dtype)
            rho_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            mu_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            lam_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            yield_stress_grad = np.zeros([1], dtype=dtype)
            viscosity_grad = np.zeros([1], dtype=dtype)
            friction_alpha_grad = np.zeros([1], dtype=dtype)
            cohesion_grad = np.zeros([1], dtype=dtype)
            self.get_input_grad(position_grad, velocity_grad, rho_grad, mu_grad, lam_grad)
            yield_stress_grad[0] = self.simulator.yield_stress.grad[None]
            friction_alpha_grad[0] = self.simulator.friction_alpha.grad[None]
            viscosity_grad[0] = self.simulator.plastic_viscosity.grad[None]
            cohesion_grad[0] = self.simulator.cohesion.grad[None]
            return torch.from_numpy(position_grad).to(self.device), \
                   torch.from_numpy(velocity_grad).to(self.device), \
                   torch.from_numpy(mu_grad).to(self.device), torch.from_numpy(lam_grad).to(self.device), \
                   torch.from_numpy(yield_stress_grad).to(self.device), torch.from_numpy(viscosity_grad).to(self.device), \
                   torch.from_numpy(friction_alpha_grad).to(self.device), torch.from_numpy(cohesion_grad).to(self.device),\
                   torch.from_numpy(rho_grad).to(self.device)
        