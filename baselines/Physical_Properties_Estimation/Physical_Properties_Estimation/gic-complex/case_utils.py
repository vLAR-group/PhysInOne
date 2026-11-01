"""Case selection, metadata and deterministic batch-file writing (stdlib only)."""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath

MATERIALS = {
    'ElasticFall': 'elastic',
    'NewtonianFluidFall': 'newtonian',
    'NonNewtonianFluidFall': 'non_newtonian',
    'PlasticineFall': 'plasticine',
    'GranularFall': 'sand',
}


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def select_cases(rows, expected_count=30):
    if not isinstance(rows, list):
        raise ValueError('Supplement map must be a list of case records')
    selected, skipped, seen = [], Counter(), set()
    for row in rows:
        name = row['case_name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_]+', name):
            raise ValueError(f'Invalid case name: {name!r}')
        if name in seen:
            raise ValueError(f'Duplicate case: {name}')
        seen.add(name)
        if not isinstance(row.get('sdf'), list):
            raise ValueError(f'{name}: sdf must be a list (empty means no SDF)')
        if not row['sdf']:
            skipped['no_sdf'] += 1
        elif name.split('__', 1)[0] not in MATERIALS:
            skipped['multiple_or_unsupported_materials'] += 1
        else:
            if any(not isinstance(name, str) or not re.fullmatch(r'test_sdf_[A-Za-z0-9_]+', name) for name in row['sdf']):
                raise ValueError(f'{name}: invalid SDF reference')
            if len(set(row['sdf'])) != len(row['sdf']):
                raise ValueError(f'{name}: duplicate SDF references')
            if not isinstance(row.get('reinit'), dict) or len(row['reinit']) != 1:
                raise ValueError(f'{name}: expected one particle object for reinitialization')
            selected.append(row)
    if len(selected) != expected_count:
        raise ValueError(f'Expected {expected_count} SDF single-material cases, found {len(selected)}')
    counts = Counter(MATERIALS[row['case_name'].split('__', 1)[0]] for row in selected)
    if expected_count == 30 and counts != Counter({name: 6 for name in MATERIALS.values()}):
        raise ValueError(f'Expected six cases per material, got {dict(counts)}')
    return sorted(selected, key=lambda row: (MATERIALS[row['case_name'].split('__', 1)[0]], row['case_name'])), dict(skipped)


def vector3(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'{label} must contain three numbers')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError(f'{label} must contain finite numbers')
    return [float(v) for v in value]


def reinit_config(row):
    item = next(iter(row['reinit'].values()))
    # PACNeRF converts rotation and translation from Blender coordinates itself.
    # Its rotation center is consumed directly, so convert ONLY the center here.
    x, y, z = vector3(item['center_point'], 'center_point')
    return {
        'rotation': vector3(item['rotation'], 'rotation'),
        'translation': vector3(item['transformation'], 'transformation'),
        'center': [x, z, -y if y else 0.0],
    }


def metadata_resolution(intrinsic):
    if isinstance(intrinsic, list) and len(intrinsic) == 3:
        if all(isinstance(row, list) and len(row) == 3 for row in intrinsic):
            # create_dataset.py fixes the principal point at (W/2, H/2).
            width, height = 2 * intrinsic[0][2], 2 * intrinsic[1][2]
            legacy = False
        elif all(isinstance(value, (int, float)) for value in intrinsic):
            # Historical metadata used [horizontal_fov, width, height].
            _, width, height = intrinsic
            legacy = True
        else:
            raise ValueError('Unsupported intrinsic shape')
        if not all(math.isfinite(v) and v > 0 and abs(v - round(v)) < 1e-4 for v in (width, height)):
            raise ValueError(f'Cannot infer image size from intrinsic: {intrinsic}')
        return int(round(height)), int(round(width)), legacy
    raise ValueError('intrinsic must be a 3x3 matrix')


def dataset_metadata(folder, fallback, require_data=False):
    path = Path(folder) / 'all_data.json'
    if not path.is_file():
        if require_data:
            raise FileNotFoundError(path)
        return dict(fallback, source='assumed', metadata_path=path.as_posix(), warnings=[])
    frames, sizes, warnings = {}, set(), set()
    for entry in read_json(path):
        match = re.fullmatch(r'r_(\d+)_(-?\d+)', PurePosixPath(entry['file_path'].replace('\\', '/')).stem)
        if not match:
            raise ValueError(f'{path}: invalid frame path {entry["file_path"]!r}')
        cam, frame = map(int, match.groups())
        if frame < 0:
            continue
        per_camera = frames.setdefault(cam, {})
        if frame in per_camera:
            raise ValueError(f'{path}: duplicate frame {cam}/{frame}')
        time = float(entry.get('time', frame * fallback['frame_dt']))
        if not math.isfinite(time):
            raise ValueError(f'{path}: nonfinite timestamp')
        per_camera[frame] = time
        height, width, legacy = metadata_resolution(entry['intrinsic'])
        sizes.add((height, width))
        if legacy:
            warnings.add('Legacy intrinsic [FOV,W,H]; regenerate with create_dataset.py before training/rendering (a 3x3 matrix is required).')
        c2w = entry['c2w']
        if not isinstance(c2w, list) or len(c2w) != 3 or any(not isinstance(row, list) or len(row) != 4 for row in c2w):
            warnings.add('c2w is not 3x4; regenerate camera metadata before training/rendering.')
    if not frames or set(frames) != set(range(len(frames))):
        raise ValueError(f'{path}: camera indices must be contiguous from zero')
    if len(sizes) != 1:
        raise ValueError(f'{path}: camera resolutions differ')
    first = frames[0]
    count = len(first)
    if set(first) != set(range(count)):
        raise ValueError(f'{path}: frame indices must be contiguous from zero')
    for cam, values in frames.items():
        if set(values) != set(first) or any(not math.isclose(values[i], first[i], rel_tol=1e-6, abs_tol=1e-8) for i in first):
            raise ValueError(f'{path}: cameras have different frame indices/timestamps (camera {cam})')
    frame_dt = first[1] - first[0] if count > 1 else fallback['frame_dt']
    if frame_dt <= 0 or any(not math.isclose(first[i], i * frame_dt, rel_tol=1e-6, abs_tol=1e-8) for i in first):
        raise ValueError(f'{path}: expected uniform timestamps starting at zero')
    if warnings and require_data:
        raise ValueError(f'{path}: {"; ".join(sorted(warnings))}')
    height, width = next(iter(sizes))
    return dict(n_cameras=len(frames), n_frames=count, H=height, W=width, frame_dt=frame_dt,
                source='metadata', metadata_path=path.as_posix(), warnings=sorted(warnings))


def config_reference(path, start):
    # Windows cannot express a relative path between different drive letters.
    try:
        return Path(os.path.relpath(path, start)).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def write_plan(files, overwrite=False):
    # Validate the entire batch before writing any config.
    conflicts = [str(path) for path, text in files.items()
                 if path.exists() and path.read_text(encoding='utf-8') != text]
    if conflicts and not overwrite:
        raise FileExistsError(f'Changed files already exist; use --overwrite to update: {conflicts[:5]}')
    written = 0
    for path, text in files.items():
        if not path.exists() or path.read_bytes() != text.encode('utf-8'):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', encoding='utf-8', newline='\n') as stream:
                stream.write(text)
            written += 1
    return written


def sdf_assets(row, sdf_root, require_data=False):
    """Match each supplementary SDF to the requested test_sdf file pair."""
    boundaries, assets = {}, []
    for index, name in enumerate(row['sdf']):
        npy = Path(sdf_root) / f'{name}.npy'
        meta = Path(sdf_root) / f'{name}.json'
        if require_data:
            for path in (npy, meta):
                if not path.is_file():
                    raise FileNotFoundError(path)
        boundaries[f'mesh{index}'] = [npy.as_posix(), meta.as_posix()]
        assets.append(dict(name=name, npy_path=npy.as_posix(), meta_path=meta.as_posix(),
                           available=npy.is_file() and meta.is_file()))
    return boundaries, assets
