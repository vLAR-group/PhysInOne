"""Shared fitting artifact and camera-independent disk trajectories for GIC."""
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np

from property_export import (load_render_cameras, parse_camera_ids, prepare_property_output,
                             compose_rgb_frame, write_camera_videos)

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def key_for(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def reinitialize(points, rotation, translation, center):
    """Angles/translation are Blender XYZ; center and points are already PAC XYZ."""
    rx, ry, rz = np.deg2rad(rotation)
    sx, sy, sz = np.sin([rx, ry, rz])
    cx, cy, cz = np.cos([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    R = C @ (Rz @ Ry @ Rx) @ C.T
    center = np.asarray(center)
    return ((points - center) @ R.T + center + C @ np.asarray(translation)).astype(np.float32)


class Trajectory:
    def __init__(self, root, metadata):
        self.key = key_for(metadata)
        self.path = Path(root) / self.key

    def frame(self, index):
        return np.load(self.path / f'{index:06d}.npy', allow_pickle=False)

    def ensure(self, frames, particle_count, factory, refresh=False):
        manifest = self.path / 'manifest.json'
        if not refresh and manifest.is_file():
            try:
                value = json.loads(manifest.read_text(encoding='utf-8'))
                if (value['key'] == self.key and value['frames'] >= frames
                        and value['particles'] == particle_count
                        and all((self.path / f'{i:06d}.npy').is_file() for i in range(frames))):
                    return False
            except (OSError, ValueError, KeyError):
                pass
        self.path.mkdir(parents=True, exist_ok=True)
        manifest.unlink(missing_ok=True)
        simulator = factory()
        simulator.initialize()
        for index in range(frames):
            positions = simulator.forward(index)
            if not simulator.succeed():
                raise RuntimeError(f'MPM failed at frame {index}; reduce physical dt')
            if hasattr(positions, 'detach'):
                positions = positions.detach().cpu().numpy()
            positions = np.asarray(positions, dtype=np.float32)
            if positions.shape != (particle_count, 3) or not np.isfinite(positions).all():
                raise ValueError(f'Invalid particle positions at frame {index}: {positions.shape}')
            temporary = self.path / f'{index:06d}.tmp'
            with temporary.open('wb') as stream:
                np.save(stream, positions, allow_pickle=False)
            temporary.replace(self.path / f'{index:06d}.npy')
        save_json(manifest, dict(key=self.key, frames=frames, particles=particle_count))
        return True


def fitting_key(cfg, static_path, prediction_path):
    gs = {key: value for key, value in cfg['gs'].items()
          if key not in ('source_path', 'model_path', 'config_path')}
    return key_for(dict(static=digest(static_path), prediction=digest(prediction_path),
                        train_metadata=digest(Path(cfg['dataset_root']) / 'train/all_data.json'),
                        gs=gs, physics=cfg['physics'], infer=cfg['infer'],
                        simulation_sampling={key: cfg['render'].get(key, cfg['physics'][key])
                                             for key in ('fps', 'mpm_iter_cnt')},
                        implementation=digest(ROOT / 'train_gs_fixed_pcd.py'),
                        camera_reader=digest(ROOT / 'scene/dataset_readers.py'),
                        calibration=digest(ROOT / 'utils/calibration.py')))


def load_ready_artifact(path, expected_key, model_path):
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            gaussian = Path(model_path) / value['gaussian_path']
            if value['key'] == expected_key and gaussian.is_file() and digest(gaussian) == value['gaussian_sha256']:
                return value
        except (OSError, KeyError, ValueError):
            pass
    return None


def learned_physics(cfg, prediction):
    value = copy.deepcopy(prediction)
    for key in ('fps', 'mpm_iter_cnt', 'gravity', 'bc', 'voxel_size', 'rho', 'density_grid_size'):
        if key in cfg['physics']:
            value[key] = cfg['physics'][key]
    if not isinstance(value.get('mat_params'), dict) or len(value.get('vel', [])) != 3:
        raise ValueError('Prediction must contain learned mat_params and vel')
    return value


def fitting_frame_indices(fit_frames, training_fps, simulation_fps):
    """Match sparse training timestamps to the shared simulation's dense grid."""
    indices = []
    for frame in range(fit_frames):
        position = frame * simulation_fps / training_fps
        index = round(position)
        if not math.isclose(position, index, rel_tol=0, abs_tol=1e-5):
            raise ValueError('Training timestamps must lie on the render FPS grid; '
                             'use a render FPS that is an integer multiple of training FPS')
        indices.append(index)
    return indices


def export_parameters(physics, output):
    """Map native prediction fields to the public material JSON schema."""
    params = physics['mat_params']
    material = params['material']
    if material == 10:
        name, fields = 'ELASTIC', {'E': 'E', 'nu': 'nu'}
    elif material == 11:
        name, fields = 'NEWTONIAN', {'mu': 'mu', 'kappa': 'kappa'}
    elif material == 12 and 'E' in params:
        name, fields = 'PLASTICINE', {'E': 'E', 'nu': 'nu', 'yield': 'yield_stress'}
    elif material == 12:
        name, fields = 'NON_NEWTONIAN', {
            'mu': 'mu', 'kappa': 'kappa', 'yield': 'yield_stress', 'visc': 'plastic_viscosity'}
    elif material == 13:
        # GIC stores the angle in degrees under the native name friction_alpha.
        name, fields = 'SAND', {'friction_angle': 'friction_alpha'}
    else:
        raise ValueError(f'Unsupported material {material}')
    material_payload = {name: {output_key: params[source_key]
                              for output_key, source_key in fields.items()}}
    save_json(Path(output) / 'materials_configs.json', material_payload)
    save_json(Path(output) / 'initvelocity.json', dict(zip(('vx', 'vy', 'vz'), physics['vel'])))


def render_shared(trajectory, frames, cameras, outputs, backgrounds, render_frame):
    """Read one particle frame and shade all cameras before advancing the sequence."""
    for directory in outputs.values():
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob('*.jpg'):
            if path.stem.isdigit() and path.is_file():
                path.unlink()
    for frame in range(frames):
        positions = trajectory.frame(frame)
        for camera_id, camera in cameras.items():
            rgb, transmittance = render_frame(positions, camera, frame)
            image = compose_rgb_frame(rgb, transmittance, backgrounds[camera_id])
            imageio.imwrite(outputs[camera_id] / f'{frame:04d}.jpg', image, quality=95)


def native_parameters(cfg, config_path):
    from argparse import ArgumentParser, Namespace
    from arguments import ModelParams, OptimizationParams, PipelineParams
    parser = ArgumentParser(add_help=False)
    model, optimization, pipeline = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    values = vars(parser.parse_args([]))
    values.update(cfg['gs'])
    values['source_path'] = str(Path(cfg['dataset_root']) / 'train')
    values['config_path'] = str(config_path)
    values['num_frame'] = cfg['infer']['fit_frames']
    args = Namespace(**values)
    return model.extract(args), optimization.extract(args), pipeline.extract(args)


def make_camera(entry, height, width, device):
    import torch
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    K = np.asarray(entry['intrinsic'])
    if K.shape != (3, 3):
        raise ValueError('Camera intrinsic must be 3x3; regenerate converted data')
    from utils.calibration import camera_extrinsics
    R, T = camera_extrinsics(entry['c2w'])
    fovx, fovy = 2 * np.arctan(width / (2 * K[0, 0])), 2 * np.arctan(height / (2 * K[1, 1]))
    world_view = torch.tensor(getWorld2View2(R, T), dtype=torch.float32, device=device).T
    projection = getProjectionMatrix(znear=.01, zfar=100., fovX=fovx, fovY=fovy).T.to(device)
    return SimpleNamespace(image_height=height, image_width=width, FoVx=fovx, FoVy=fovy,
                           world_view_transform=world_view, full_proj_transform=world_view @ projection,
                           camera_center=world_view.inverse()[3, :3])


def run(cfg, config_path, mode, args):
    import torch
    import taichi as ti
    import trimesh
    from argparse import Namespace
    from simulator import Simulator
    from utils.general_utils import safe_state
    from scene import GaussianModel
    from gaussian_renderer import render

    safe_state(cfg['gs'].get('quiet', False))
    model_path = Path(cfg['gs']['model_path'])
    static_path = model_path / 'mpm/static_0.ply'
    prediction_path = model_path / f"{cfg['physics']['id']}-pred.json"
    for path in (static_path, prediction_path):
        if not path.is_file():
            raise FileNotFoundError(f'{path}: complete train_dynamic.py first')
    prediction = json.loads(prediction_path.read_text(encoding='utf-8'))
    physics = learned_physics(cfg, prediction)
    training_fps = physics['fps']
    for key in ('fps', 'mpm_iter_cnt'):
        physics[key] = cfg['render'].get(key, physics[key])
    source_points = np.asarray(trimesh.load_mesh(static_path).vertices, dtype=np.float32)
    if source_points.ndim != 2 or source_points.shape[1] != 3 or not np.isfinite(source_points).all():
        raise ValueError('Invalid static particle cloud')
    artifact_path = model_path / 'inference/manifest.json'
    expected_key = fitting_key(cfg, static_path, prediction_path)
    artifact = load_ready_artifact(artifact_path, expected_key, model_path)
    if mode != 'infer' and artifact is None:
        raise RuntimeError('Inference Gaussians are missing or stale; run test.py --mode infer first')
    if mode != 'infer':
        data_dir = Path(cfg['dataset_root']) if mode == 'resimulate' else Path(cfg['dataset_root']) / 'test'
        render_cfg = dict(data_dir=str(data_dir), base_dir=str(model_path), H=cfg['render']['H'], W=cfg['render']['W'],
                          n_cameras=cfg['render']['all_cameras' if mode == 'resimulate' else 'test_cameras'])
        options = SimpleNamespace(output_dir=None, camera_mapping=None, camera_split=None)
        export_mode = 'reinit' if mode == 'resimulate' else 'novel'
        camera_metadata = load_render_cameras(render_cfg, options, export_mode)
        if set(camera_metadata[0]) != set(range(render_cfg['n_cameras'])):
            raise ValueError('Camera count differs from config; regenerate configs')
        for entry in camera_metadata[0].values():
            if np.asarray(entry['intrinsic']).shape != (3, 3):
                raise ValueError('Camera intrinsic must be 3x3; regenerate converted data')
        ids = parse_camera_ids(args.cam_id, render_cfg['n_cameras'])
        outputs, backgrounds = prepare_property_output(render_cfg, options, export_mode, ids, camera_metadata=camera_metadata)
    frames = args.num_frame or cfg['render']['n_frames']
    if mode == 'infer':
        fit_indices = fitting_frame_indices(cfg['infer']['fit_frames'], training_fps, physics['fps'])
        frames = max(frames, fit_indices[-1] + 1)
    transform = cfg['reinit'] if mode == 'resimulate' else None
    points = source_points if transform is None else reinitialize(source_points, **transform)
    vol = torch.as_tensor(points, dtype=torch.float32, device='cuda').contiguous()
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=.2)
    metadata = dict(static=digest(static_path), physics=physics, transform=transform,
                    simulator=digest(ROOT / 'simulator/simulator.py'), mpm=digest(ROOT / 'simulator/mpm_simulator.py'))
    trajectory = Trajectory(model_path / 'inference/trajectories', metadata)
    output = model_path / 'property_estimation'
    with torch.no_grad():
        trajectory.ensure(frames, len(points), lambda: Simulator(Namespace(**physics), vol), args.refresh_simulation)
    if mode == 'infer':
        if artifact is None or args.force_infer:
            from train_gs_fixed_pcd import train_gs_with_fixed_pcd
            dataset, opt, pipe = native_parameters(cfg, config_path)
            fit_frames = len(fit_indices)
            displacements = [torch.as_tensor(trajectory.frame(i) - source_points, device='cuda')
                             for i in fit_indices]
            scene = train_gs_with_fixed_pcd(vol, dataset, opt, pipe,
                cfg['gs']['test_iterations'], cfg['gs']['save_iterations'], displacements,
                training_fps, force_train=True, grid_size=physics['density_grid_size'])
            gaussian_path = model_path / f'point_cloud_fix_pcd/iteration_{opt.iterations}/point_cloud.ply'
            if not gaussian_path.is_file():
                raise FileNotFoundError(f'Fixed-pcd training did not save {gaussian_path}')
            artifact = dict(key=expected_key, gaussian_path=gaussian_path.relative_to(model_path).as_posix(),
                            gaussian_sha256=digest(gaussian_path), fit_frames=fit_frames)
            save_json(artifact_path, artifact)
            del scene, displacements
        export_parameters(physics, output)
        print(f'Inference artifact ready: {artifact_path}')
        return artifact_path

    gaussians = GaussianModel(cfg['gs'].get('sh_degree', 3))
    gaussians.load_ply(str(model_path / artifact['gaussian_path']))
    if tuple(gaussians.get_xyz.shape) != tuple(vol.shape):
        raise ValueError('Fixed Gaussians and simulator particles have different counts; rerun infer')
    if not torch.allclose(gaussians.get_xyz.detach(), torch.as_tensor(source_points, device='cuda'), atol=1e-5):
        raise ValueError('Fixed Gaussians do not match canonical particle order; rerun infer')
    cameras = {i: make_camera(camera_metadata[0][i], render_cfg['H'], render_cfg['W'], 'cuda') for i in ids}
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.ones(3, device='cuda')
    state = {}
    def render_frame(positions, camera, frame):
        if state.get('frame') != frame:
            state['frame'] = frame
            state['displacement'] = torch.as_tensor(positions, device='cuda') - gaussians.get_xyz.detach()
        results = render(camera, gaussians, pipe, background, state['displacement'], 0., 0., False)
        rgb = results['render'].detach().permute(1, 2, 0).cpu().numpy()
        transmittance = (1 - results['alpha'].detach()).permute(1, 2, 0).cpu().numpy()
        return rgb, transmittance
    with torch.no_grad():
        render_shared(trajectory, frames, cameras, outputs, backgrounds, render_frame)
    export_parameters(physics, output)
    if args.write_video:
        args.num_frame = frames
        write_camera_videos(outputs, args)
    return outputs
