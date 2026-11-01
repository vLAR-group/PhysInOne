"""Generate SDF-boundary, single-material training/novel/resimulation configs.

Uses only Python's standard library; does not import MMCV, PyTorch or Taichi.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path, PurePosixPath
import pprint
import re

ROOT = Path(__file__).resolve().parent
MATERIALS = {
    'ElasticFall': 'elastic',
    'NewtonianFluidFall': 'newtonian',
    'NonNewtonianFluidFall': 'non_newtonian',
    'PlasticineFall': 'plasticine',
    'GranularFall': 'sand',
}
REFERENCE_HIT_FRAMES = {'test_sdf_elephanttoy': 17, 'test_sdf_kettle': 19}
CASE_OVERRIDES = {'GranularFall__bg853__ZBMM7p_trajectory': {'dx': 0.12}}


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


def config_text(base, values, comment):
    text = ('# Generated by generate_configs.py.\n# ' + comment + '\n'
            + '_base_ = ' + repr(base) + '\n\ncfg = ' + pprint.pformat(values, sort_dicts=False, width=110) + '\n')
    compile(text, '<generated config>', 'exec')
    return text


def config_reference(path, start):
    # Windows cannot express a relative path between different drive letters.
    try:
        return Path(os.path.relpath(path, start)).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def sdf_assets(row, sdf_root, require_data=False):
    """Resolve supplementary SDF names to the existing simulator file pair."""
    boundaries, assets = {}, []
    for index, name in enumerate(row['sdf']):
        suffix = name[len('test_sdf_'):]
        npy = Path(sdf_root) / f'test_sdf_{suffix}.npy'
        meta = Path(sdf_root) / f'test_sdf_{suffix}.json'
        if require_data:
            for path in (npy, meta):
                if not path.is_file():
                    raise FileNotFoundError(path)
        boundaries['mesh' if index == 0 else f'mesh_{index}'] = [npy.as_posix(), meta.as_posix()]
        assets.append(dict(name=name, npy_path=npy.as_posix(), meta_path=meta.as_posix(),
                           available=npy.is_file() and meta.is_file()))
    return boundaries, assets


def build_plan(args, rows):
    selected, skipped = select_cases(rows, args.expected_cases)
    output = Path(args.output_dir).resolve()
    files, records = {}, []
    for row in selected:
        case = row['case_name']
        material = MATERIALS[case.split('__', 1)[0]]
        template = ROOT / 'configs' / material / 'default.py'
        if not template.is_file():
            raise FileNotFoundError(template)
        boundaries, assets = sdf_assets(row, args.sdf_root, args.require_data)
        reference_hit = args.hit_frame if args.hit_frame is not None else REFERENCE_HIT_FRAMES.get(row['sdf'][0])
        if reference_hit is None:
            raise ValueError(f'{case}: no known hit_frame; specify --hit-frame at the original FPS')
        for gap in sorted(set(args.gap)):
            stem = f'{case}_gap{gap}'
            folder = Path(args.data_root) / stem
            fallback = dict(n_frames=math.ceil(args.source_frames / gap), frame_dt=gap / args.original_fps,
                            H=args.height, W=args.width)
            train = dataset_metadata(folder / 'train', dict(fallback, n_cameras=args.train_cameras), args.require_data)
            novel = dataset_metadata(folder / 'test', dict(fallback, n_cameras=args.test_cameras), args.require_data)
            if novel['n_frames'] > 1 and not math.isclose(train['frame_dt'], novel['frame_dt'], rel_tol=1e-6):
                raise ValueError(f'{case}: train/test frame intervals differ')
            if (train['H'], train['W']) != (novel['H'], novel['W']):
                raise ValueError(f'{case}: train/test resolutions differ; cannot render all cameras together')
            hit_frame = math.ceil(reference_hit / (train['frame_dt'] * args.original_fps) - 1e-9)
            if hit_frame < 2 or hit_frame + args.warmup_step > train['n_frames']:
                raise ValueError(f'{stem}: hit_frame={hit_frame}, warmup_step={args.warmup_step}, '
                                 f'valid frames={train["n_frames"]}; too few frames for the existing velocity/dynamic stages')
            max_dt = 1.0 / (args.original_fps * args.substeps_per_source_frame)
            substeps = max(1, math.ceil(train['frame_dt'] / max_dt - 1e-9))
            dt = train['frame_dt'] / substeps
            model_dir = (PurePosixPath(str(args.model_root).replace('\\', '/')) / material / stem).as_posix()
            directory = output / material
            paths = {mode: directory / (stem + suffix + '.py') for mode, suffix in
                     [('train', ''), ('novel', '_novel'), ('resimulate', '_resimulate')]}
            base = config_reference(template, directory)
            values = dict(data_dir=(folder / 'train').as_posix(), base_dir=model_dir,
                          n_cameras=train['n_cameras'], n_frames=train['n_frames'],
                          H=train['H'], W=train['W'], frame_dt=train['frame_dt'], dt=dt,
                          hit_frame=hit_frame, warmup_step=args.warmup_step, BC=boundaries)
            values.update(CASE_OVERRIDES.get(case, {}))
            files[paths['train']] = config_text(base, values, 'Training: nonnegative frame IDs only; SDF mesh boundaries are enabled.')
            render_frames = args.render_frames if args.render_frames is not None else args.source_frames
            render_frame_dt = 1.0 / args.original_fps
            novel_values = dict(data_dir=(folder / 'test').as_posix(), n_cameras=novel['n_cameras'],
                                n_frames=render_frames, frame_dt=render_frame_dt, H=novel['H'], W=novel['W'])
            files[paths['novel']] = config_text(paths['train'].name, novel_values,
                                               'Novel views: one simulation shared by all test cameras.')
            resimulate_values = dict(data_dir=folder.as_posix(),
                                     n_cameras=train['n_cameras'] + novel['n_cameras'],
                                     n_frames=render_frames, frame_dt=render_frame_dt, H=train['H'], W=train['W'])
            resimulate_values.update(reinit_config(row))
            files[paths['resimulate']] = config_text(paths['train'].name, resimulate_values,
                'Reinitialize particles; one simulation shared by all train + test cameras (13 + 2 = 15 by default).')
            relative = {mode: path.relative_to(output).as_posix() for mode, path in paths.items()}
            commands = {}
            for mode, path in paths.items():
                config_arg = config_reference(path, ROOT)
                cmd = ['python', 'train.py' if mode == 'train' else 'test.py', '--config', config_arg]
                if mode != 'train':
                    cmd += ['--render-kind', 'novel' if mode == 'novel' else 'reinit',
                            '--num-frame', str(render_frames), '--fps', f'{args.original_fps:g}', '--cam-id', 'all']
                commands[mode] = cmd
            records.append(dict(case_name=case, material=material, gap=gap, configs=relative,
                                base_dir=model_dir, train=train, novel=novel, sdf_assets=assets,
                                case_overrides=CASE_OVERRIDES.get(case, {}),
                                resimulate=dict(data_dir=folder.as_posix(),
                                                camera_splits=['train', 'test'],
                                                n_cameras=resimulate_values['n_cameras']),
                                train_frames=train['n_frames'], render_frames=render_frames,
                                render_fps=args.original_fps, render_frame_dt=render_frame_dt,
                                hit_frame=hit_frame, substeps=substeps, dt=dt,
                                reinit=reinit_config(row), commands=commands))
    manifest = dict(supp_map=str(args.supp_map).replace('\\', '/'), data_root=str(args.data_root).replace('\\', '/'),
                    sdf_root=str(args.sdf_root).replace('\\', '/'),
                    selected_cases=len(selected), per_material=dict(Counter(MATERIALS[r['case_name'].split('__', 1)[0]] for r in selected)),
                    skipped=skipped, gaps=sorted(set(args.gap)), config_count=len(files), records=records)
    files[output / 'manifest.json'] = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    return files, manifest


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


def config_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  python %(prog)s --supp-map /path/to/test_case_supp_map.json --data-root /path/to/baseline --sdf-root /path/to/boundary --model-root checkpoint --output-dir configs/generated --gap 1\nAll paths must be provided explicitly. Numerical options have defaults.')
    parser.add_argument('--supp-map', required=True, metavar='JSON',
                        help='Supplementary case map JSON file (required)')
    parser.add_argument('--data-root', required=True, metavar='DIR',
                        help='Converted datasets root containing <case>_gapN/train and test (required)')
    parser.add_argument('--sdf-root', required=True, metavar='DIR',
                        help='SDF root containing test_sdf_*.npy and test_sdf_*.json (required)')
    parser.add_argument('--model-root', required=True, metavar='DIR',
                        help='Model root; checkpoints go under <material>/<case>_gapN (required)')
    parser.add_argument('--output-dir', required=True, metavar='DIR',
                        help='Destination for generated configs and manifest.json (required)')
    parser.add_argument('--gap', nargs='+', type=int, default=[1],
                        help='One or more source-frame sampling intervals (default: %(default)s)')
    parser.add_argument('--original-fps', type=float, default=150.0,
                        help='Render FPS and source FPS for missing metadata (default: %(default)s)')
    parser.add_argument('--source-frames', type=int, default=90,
                        help='Default render length and source frame count before gap sampling (default: %(default)s)')
    parser.add_argument('--train-cameras', type=int, default=13,
                        help='Training camera count when metadata is absent (default: %(default)s)')
    parser.add_argument('--test-cameras', type=int, default=2,
                        help='Test camera count when metadata is absent (default: %(default)s)')
    parser.add_argument('--height', type=int, default=1120,
                        help='Fallback image height in pixels (default: %(default)s)')
    parser.add_argument('--width', type=int, default=1120,
                        help='Fallback image width in pixels (default: %(default)s)')
    parser.add_argument('--substeps-per-source-frame', type=int, default=40,
                        help='Limit physical dt to 1/(original_fps * this value) (default: %(default)s)')
    parser.add_argument('--hit-frame', type=int, default=None,
                        help='Override pre-contact frame count at original FPS; otherwise use case-specific values')
    parser.add_argument('--warmup-step', type=int, default=3,
                        help='Additional dynamic warmup frames (default: %(default)s)')
    parser.add_argument('--render-frames', type=int, default=None,
                        help='Number of novel/reinit frames; defaults to --source-frames (90), independent of --gap')
    parser.add_argument('--expected-cases', type=int, default=30,
                        help='Required number of selected single-material cases (default: %(default)s)')
    parser.add_argument('--require-data', action='store_true',
                        help='Require camera metadata, current calibration format and all SDF file pairs')
    parser.add_argument('--overwrite', action='store_true',
                        help='Update generated files whose contents differ (default: keep and report conflicts)')
    return parser

def main(argv=None):
    parser = config_parser()
    args = parser.parse_args(argv)
    for name in ('original_fps', 'source_frames', 'train_cameras', 'test_cameras', 'height', 'width',
                 'substeps_per_source_frame', 'expected_cases'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if any(g <= 0 for g in args.gap) or args.warmup_step < 0:
        parser.error('--gap must be positive and --warmup-step must be nonnegative')
    if (args.hit_frame is not None and args.hit_frame < 2) or (args.render_frames is not None and args.render_frames < 1):
        parser.error('--hit-frame must be at least 2 and --render-frames must be positive')
    files, manifest = build_plan(args, read_json(args.supp_map))
    written = write_plan(files, args.overwrite)
    print(f'Selected {manifest["selected_cases"]} cases: {manifest["per_material"]}')
    print(f'Skipped: {manifest["skipped"]}')
    assets = [asset for record in manifest['records'] for asset in record['sdf_assets']]
    print(f'SDF file pairs: {sum(asset["available"] for asset in assets)}/{len(assets)} available')
    found = sum(r[s]['source'] == 'metadata' for r in manifest['records'] for s in ('train', 'novel'))
    warnings = sum(bool(r[s]['warnings']) for r in manifest['records'] for s in ('train', 'novel'))
    total = len(manifest['records']) * 2
    print(f'Metadata: {found}/{total} splits found; {total-found} use fallback defaults; {warnings} need calibration regeneration')
    print(f'Generated {manifest["config_count"]} configs ({written} files written including manifests) in {Path(args.output_dir).resolve()}')
    print('Run commands and per-case metadata status: manifest.json')
    return manifest


if __name__ == '__main__':
    main()
