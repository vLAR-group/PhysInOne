"""Camera-independent particle trajectories for inference and disk reuse."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import torch


PHYSICS_KEYS = (
    'dtype', 'dt', 'frame_dt', 'dx', 'material', 'gravity', 'BC',
    'use_complex_boundary', 'alpha_init', 'mask_thres', 'fast_color_thres',
    'particle_chunk_size', 'cuda_chunk_size', 'E', 'nu', 'mu', 'kappa',
    'lambda', 'lam', 'rho', 'yield_stress', 'plastic_viscosity',
    'plastic_pow', 'friction_angle', 'cohesion',
)


def _json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_metadata(cfg, checkpoint_path, transform=None):
    # Camera, image size, output format and data directory do not affect physics.
    assets = {}

    def find_assets(value):
        if isinstance(value, str) and os.path.isfile(value):
            path = Path(value).resolve()
            stat = path.stat()
            assets[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        elif isinstance(value, dict):
            for child in value.values():
                find_assets(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                find_assets(child)

    find_assets(cfg.get('BC', {}))
    lib_dir = Path(__file__).resolve().parent
    sources = ['trajectory_cache.py', 'pac_nerf.py',
               'engine/dynamic_observer.py', 'engine/mpm_simulator.py']
    return _json_value({
        'schema': 1,
        'checkpoint_sha256': _file_digest(checkpoint_path),
        'physics': {key: cfg[key] for key in PHYSICS_KEYS if key in cfg},
        'transform': transform,
        'particle_seed': 123,
        'boundary_assets': assets,
        'engine': {name: _file_digest(lib_dir / name) for name in sources},
    })


def parse_camera_ids(value, count):
    if count < 1:
        raise ValueError('The configured camera count must be positive')
    if value == 'all':
        return list(range(count))
    try:
        ids = [int(item.strip()) for item in value.split(',')]
    except ValueError as exc:
        raise ValueError("--cam-id must be 'all' or comma-separated camera IDs") from exc
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= count for i in ids):
        raise ValueError(f'Camera IDs must be unique and between 0 and {count - 1}')
    return ids


class ParticleTrajectory:
    """Store CPU particle positions, not large per-frame feature grids."""
    def __init__(self, root, metadata, mode='auto'):
        if mode not in ('auto', 'refresh', 'off'):
            raise ValueError(f'Unknown simulation cache mode: {mode}')
        self.metadata = metadata
        self.mode = mode
        encoded = json.dumps(metadata, sort_keys=True, allow_nan=False).encode('utf-8')
        self.key = hashlib.sha256(encoded).hexdigest()[:24]
        self.directory = Path(root) / self.key
        self._initial = None
        self._frames = []
        self._written = 0
        self._generation = None
        self.num_frames = 0

    def _reuse(self, num_frames):
        path = self.directory / 'manifest.json'
        if not path.exists():
            return False
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
            generation = manifest['generation']
            if Path(generation).name != generation or not generation.startswith('data-'):
                return False
            folder = self.directory / generation
            if manifest['metadata'] != self.metadata or manifest['num_frames'] < num_frames:
                return False
            if not (folder / 'initial.pt').is_file():
                return False
            if not all((folder / f'{i:06d}.pt').is_file() for i in range(num_frames)):
                return False
        except (OSError, KeyError, TypeError, ValueError):
            return False
        self._generation = folder
        self._initial = None
        self.num_frames = manifest['num_frames']
        print(f'[Simulation] reuse {self.num_frames} cached frames: {self.directory}')
        return True

    def prepare(self, num_frames, build):
        if num_frames < 1:
            raise ValueError('--num-frame must be positive')
        if self.mode == 'auto' and self._reuse(num_frames):
            return
        if self.mode == 'off':
            self._initial, self._frames, self._written = None, [], 0
            build(self)
            self._finish(num_frames)
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = self.directory / 'build.lock'
        try:
            handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(
                f'Another process is building {self.directory}. Wait for it to finish. '
                f'If it crashed, remove only its stale lock file: {lock}'
            ) from exc
        try:
            with os.fdopen(handle, 'w') as stream:
                stream.write(str(os.getpid()))
            # A preceding process may have finished between lookup and locking.
            if self.mode == 'auto' and self._reuse(num_frames):
                return
            self._generation = self.directory / ('data-' + uuid.uuid4().hex)
            self._generation.mkdir()
            self._initial, self._frames, self._written = None, [], 0
            print(f'[Simulation] simulate once for all requested cameras: {self.directory}')
            build(self)
            self._finish(num_frames)
            manifest = {'metadata': self.metadata, 'num_frames': self.num_frames,
                        'generation': self._generation.name}
            temporary = self.directory / ('manifest-' + uuid.uuid4().hex + '.tmp')
            temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
            os.replace(str(temporary), str(self.directory / 'manifest.json'))
        finally:
            lock.unlink()

    def save_initial(self, features, dx):
        features = features.detach().cpu().contiguous()
        if features.ndim != 2 or len(features) == 0 or not bool(torch.isfinite(features).all()):
            raise ValueError('Initial particle features must be a finite nonempty matrix')
        self._initial = {'features': features.clone(), 'dx': float(dx)}
        if self.mode != 'off':
            torch.save(self._initial, self._generation / 'initial.pt')

    def save_frame(self, index, positions):
        if self._initial is None or index != self._written:
            raise ValueError('Save initial features, then consecutive trajectory frames')
        positions = positions.detach().cpu().contiguous()
        if positions.shape != (len(self._initial['features']), 3):
            raise ValueError('Particle count/order must stay fixed across trajectory frames')
        if not bool(torch.isfinite(positions).all()):
            raise FloatingPointError(f'Nonfinite particle positions at frame={index}')
        if self.mode == 'off':
            self._frames.append(positions.clone())
        else:
            torch.save(positions, self._generation / f'{index:06d}.pt')
        self._written += 1

    def _finish(self, num_frames):
        if self._initial is None or self._written != num_frames:
            raise RuntimeError(f'Incomplete simulation: {self._written}/{num_frames} frames')
        self.num_frames = num_frames

    def initial(self):
        if self._initial is None:
            self._initial = torch.load(self._generation / 'initial.pt', map_location='cpu')
        return self._initial

    def frame(self, index):
        if index < 0 or index >= self.num_frames:
            raise IndexError(index)
        if self.mode == 'off':
            return self._frames[index]
        return torch.load(self._generation / f'{index:06d}.pt', map_location='cpu')
