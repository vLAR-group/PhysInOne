
from simulator import MPMSimulator
from argparse import Namespace
import taichi as ti
import numpy as np
import torch

@ti.data_oriented
class Simulator:
    material_attr_names = ["E", 
                           "nu", 
                           "yield_stress", 
                           "plastic_viscosity", 
                           "mu",
                           "kappa", 
                           "friction_alpha"]
    def __init__(self, phys_args, vol, device='cuda'):
        self.device = device
        self.num_particles = ti.field(ti.i32, shape=())
        self.num_particles[None] = vol.shape[0]
        gravity = phys_args.gravity
        self.dx = ti.field(ti.f32, shape=())
        self.inv_dx = ti.field(ti.f32, shape=())
        self.dx[None] = phys_args.voxel_size
        self.inv_dx[None] = 1/self.dx[None]

        self.particle_rho = ti.field(dtype=ti.f32)
        particle = ti.root.dynamic(ti.i, 2**30, 2**14)
        particle.place(self.particle_rho)
        frame_dt = 1.0 / phys_args.fps
        dt = frame_dt / phys_args.mpm_iter_cnt
        self.sim = MPMSimulator(ti.f32, dt, frame_dt, 
                                particle, self.dx, 
                                self.inv_dx, 
                                self.num_particles, 
                                Namespace(**phys_args.mat_params),
                                gravity, 
                                phys_args.mat_params["material"],
                                cuda_chunk_size = 100)
        self.vol = vol
        self.phys_args = phys_args
        self.mat = phys_args.mat_params
        
        self.vel = torch.tensor(phys_args.vel, device=self.device)
        self.omega = torch.tensor(getattr(phys_args, "omega", [0.0, 0.0, 0.0]), device=self.device)
        self.rho = torch.tensor([phys_args.rho], device=self.device)

    def forward(self, f):
        particle_pos = np.zeros([self.sim.n_particles[None], 3], dtype=np.float32)
        # while True:
        if f > 0:
            self.sim.advance(f-1)
        if not self.succeed():
            print('NOT SUCCEED ERROR!!')
            return torch.from_numpy(particle_pos).to(self.device)
        self.sim.get_x(f, particle_pos)
        particle_pos =torch.from_numpy(particle_pos).to(self.device)
        return particle_pos

    def succeed(self):
        return self.sim.cfl_satisfy[None]

    @ti.kernel
    def from_torch(self, particles: ti.types.ndarray(), 
                         velocities: ti.types.ndarray(), 
                         particle_rho: ti.types.ndarray(), 
                         particle_mu: ti.types.ndarray(), 
                         particle_lam: ti.types.ndarray()):
        for p in range(self.num_particles[None]):
            self.particle_rho[p] = particle_rho[p]
            self.sim.mu[p] = particle_mu[p]
            self.sim.lam[p] = particle_lam[p]
            self.sim.p_mass[p] = 0.0
            self.sim.F[p, 0] = ti.Matrix.identity(ti.f32, 3)
            self.sim.C[p, 0] = ti.Matrix.zero(ti.f32, 3, 3)
            for d in ti.static(range(3)):
                self.sim.x[p, 0][d] = particles[p, d]
                self.sim.v[p, 0][d] = velocities[p, d]

    @ti.kernel
    def compute_particle_vol(self):
        grid_vol = (self.dx[None] * 0.5) ** 3
        for p in range(self.num_particles[None]):
            self.sim.p_vol[p] = grid_vol

    @ti.kernel
    def compute_particle_mass(self):
        for p in range(self.num_particles[None]):
            # self.simulator.p_mass[p] = 1.0
            self.sim.p_mass[p] = self.particle_rho[p] * self.sim.p_vol[p]

    # TODO: del 
    def load_estimation_params(self, estimation_params):
        # 0. material 
        # TODO: 1.need to alias the result format 2. process the non-newtonian case
        self.material = estimation_params.get("material", MPMSimulator.elasticity)
        self.sim.material = self.material

        # 1. vel
        self.vel = torch.tensor(estimation_params["v"], device=self.device)

        # 2. physical params
        for attr_name in self.material_attr_names:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

        report_msg = ""
        for attr_name, value in estimation_params.items():
            report_msg += f"{attr_name}: {value} "
            # TODO: Consider mpm simulator non-newtonian material
            if attr_name in self.material_attr_names:
                setattr(self, attr_name, torch.tensor([value], device=self.device))
            else:
                print(f'{attr_name} wrote.')
        print("Material info(load estimation): " + report_msg)

    def reload(self, phys_args=None):
        if phys_args:
            self.phys_args = phys_args
            self.mat = phys_args.mat_params
            # TODO load phys_args to self.sim
        
        self._load_particles_params()
        self._load_material_params()
        self._load_bc()

    def _load_particles_params(self):
        self.vel = torch.tensor(self.phys_args.vel, device=self.device)
        self.rho = torch.tensor([self.phys_args.rho], device=self.device)

    def _load_material_params(self):
        for attr_name in self.material_attr_names:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

        report_msg = ""
        for attr_name, value in self.mat.items():
            report_msg += f"{attr_name}: {value} "
            # TODO: Consider mpm simulator non-newtonian material
            if attr_name == "material":
                self.material = value
                self.sim.material = value
            else:
                setattr(self, attr_name, torch.tensor([value], device=self.device))
        print("Material info: " + report_msg)

    def _load_bc(self):
        self.sim.analytic_collision.clear()
        for collider_type, collider in self.phys_args.bc.items():
            if "ground" in collider_type:
                point, normal, bc_style = collider
                self.sim.add_surface_collider(point, normal, bc_style)
                # self.simulator.add_surface_collider([0, 0, 0], [0, 1, 0], MPMSimulator.surface_sticky)
            elif "cylinder" in collider_type:
                start, end, radius, bc_style = collider
                self.sim.add_cylinder_collider(start, end, radius, bc_style)
            elif "mesh" in collider_type:
                npy = collider[0]
                meta = collider[1]
                print("[DBG] before register, analytic_collision len =", len(self.sim.analytic_collision))
                self.sim.add_mesh_collider_from_files(npy, meta)
                print("[DBG] after register, analytic_collision len =", len(self.sim.analytic_collision))

    def compute_velocities(self):
        cnt = self.vol.shape[0]
        velocities = self.vel.repeat(cnt).reshape(cnt, -1)
        centroid = self.vol.sum(dim=0) / self.vol.shape[0]
        omega = self.omega.repeat(cnt).reshape(cnt, -1)
        velocities += torch.cross(omega, self.vol - centroid) 
        return velocities

    def initialize(self, phys_args=None):
        self.reload(phys_args)
        self.compute_particle_vol()
        cnt = self.vol.shape[0]
        velocities = self.vel.repeat(cnt).reshape(cnt, -1)
        # velocities = self.compute_velocities()

        if getattr(self, 'E', None) and getattr(self, 'nu', None):
            mu = self.E / (2. * (1. + self.nu))
            lam = self.E * self.nu / ((1. + self.nu) * (1. - 2. * self.nu))
        elif getattr(self, 'kappa', None) and getattr(self, 'mu', None):
            mu = self.mu
            lam = self.kappa - 2./3. * self.mu
        else:
            print('Error: material undefined! ')
        mu = mu.repeat(cnt)
        lam = lam.repeat(cnt)
        rho = self.rho.repeat(cnt)
        self.from_torch(self.vol.data.cpu().numpy(), 
                        velocities.data.cpu().numpy(), 
                        rho.data.cpu().numpy(), 
                        mu.data.cpu().numpy(), 
                        lam.data.cpu().numpy())
        
        self.compute_particle_mass()

        if getattr(self, "friction_alpha", None):
            sin_phi = torch.sin(self.friction_alpha / 180 * np.pi)
            friction_alpha = np.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
            self.sim.friction_alpha[None] = friction_alpha.item()
        
        if getattr(self, "yield_stress", None):
            self.sim.yield_stress[None] = self.yield_stress.item()

        if getattr(self, "plastic_viscosity", None):
            self.sim.plastic_viscosity[None] = self.plastic_viscosity.item()
        
        # TODO: Pacnerf have not this params
        # if getattr(self, "cohesion", None):
        #     pass

        self.sim.cfl_satisfy[None] = True