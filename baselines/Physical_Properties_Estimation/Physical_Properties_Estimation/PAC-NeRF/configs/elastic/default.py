from lib.engine import MPMSimulator

_base_ = '../default.py'

cfg = dict(
    dtype='float32',
    n_cameras = 13,
    n_frames = 90,
    xyz_min = [-0.5, 0.1, -0.5],
    xyz_max = [0.5, 1.0, 0.5],
    dx = 0.08,
    material = MPMSimulator.elasticity,
    pg_scale = [1000, 2000, 4000],
    cuda_chunk_size = 100,
    nerf_bs = 2**16,
    particle_chunk_size=2**10,
    taichi_cuda_memory=0.6,
    dt = 1/150/40,
    frame_dt=1/150,
    H = 1120,
    W = 1120,
    N_static = 6001,
    entropy_weight = 1e-4,
    volume_weight = 1e-3,
    # tv_weight = 1e-4,
    direct_nerf = False,
    E = 316228,
    nu = 0.25,
    rho = 1000,
    hit_frame = 37,
    physical_params = dict(global_E=1e-1, global_nu=1e-2),
    write_out = True,
    random_particles=True,
)

del MPMSimulator