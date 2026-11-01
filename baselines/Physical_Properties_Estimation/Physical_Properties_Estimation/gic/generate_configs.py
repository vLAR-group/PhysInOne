"""Generate GIC train/infer/novel/resimulate JSON configs for 30 no-SDF cases."""
import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

from case_utils import (MATERIALS, read_json, select_cases, dataset_metadata,
                        reinit_config, config_reference, write_plan)

ROOT = Path(__file__).resolve().parent
MODES = ('train', 'infer', 'novel', 'resimulate')
LIGHT_SAND_CASES = {'GranularFall__bg852__MoyIIZ_trajectory', 'GranularFall__bg852__tDtaft_trajectory'}


def build_plan(args, rows):
    selected, skipped = select_cases(rows, args.expected_cases)
    output = Path(args.output_dir).resolve()
    files, records = {}, []
    for row in selected:
        case = row['case_name']
        material = MATERIALS[case.split('__', 1)[0]]
        template = read_json(ROOT / 'config/materials' / f'{material}.json')
        for gap in sorted(set(args.gap)):
            stem = f'{case}_gap{gap}'
            folder = Path(args.data_root) / stem
            fallback = dict(n_frames=math.ceil(args.source_frames / gap), frame_dt=gap / args.original_fps,
                            H=args.height, W=args.width)
            train = dataset_metadata(folder / 'train', dict(fallback, n_cameras=13), args.require_data)
            test = dataset_metadata(folder / 'test', dict(fallback, n_cameras=2), args.require_data)
            if (train['H'], train['W']) != (test['H'], test['W']):
                raise ValueError(f'{case}: train/test resolutions differ')
            if not math.isclose(train['frame_dt'], test['frame_dt'], rel_tol=1e-6):
                raise ValueError(f'{case}: train/test frame intervals differ')
            if train['n_frames'] < 2:
                raise ValueError(f'{case}: at least two training frames are required')
            fps = 1 / train['frame_dt']
            mpm_steps = max(1, math.ceil(train['frame_dt'] * args.original_fps * args.substeps_per_source_frame - 1e-9))
            model_dir = (Path(args.model_root) / material / stem).as_posix()
            physics = copy.deepcopy(template['physics'])
            physics.update(fps=fps, mpm_iter_cnt=mpm_steps, n_frames=train['n_frames'])
            physics['vel_estimation_frames'] = min(physics['vel_estimation_frames'], train['n_frames'])
            if case in LIGHT_SAND_CASES:
                physics.update(voxel_size=.025, density_grid_size=.16)
            render_frames = args.render_frames or args.source_frames
            paths, commands = {}, {}
            for mode in MODES:
                iterations = args.gs_iterations if mode == 'train' else args.infer_iterations
                gs = copy.deepcopy(template['gs'])
                saving = sorted(set([n for n in gs['save_iterations'] if n <= iterations] + [iterations]))
                testing = sorted(set([n for n in gs['test_iterations'] if n <= iterations]
                                     + list(range(10000, iterations + 1, 1000)) + [iterations]))
                gs.update(source_path=(folder / ('test' if mode == 'novel' else 'train')).as_posix(),
                          model_path=model_dir, iterations=iterations, iteration=args.gs_iterations,
                          save_iterations=saving, test_iterations=testing, num_frame=train['n_frames'],
                          reg_scale=True, reg_alpha=True)
                value = dict(stage=mode, case_name=case, material=material, gap=gap,
                    dataset_root=folder.as_posix(), data=copy.deepcopy(template['data']), gs=gs,
                    physics=copy.deepcopy(physics), infer=dict(fit_frames=min(args.infer_frames, train['n_frames'])),
                    render=dict(n_frames=render_frames, fps=args.original_fps,
                                mpm_iter_cnt=args.substeps_per_source_frame, H=train['H'], W=train['W'],
                                test_cameras=test['n_cameras'], all_cameras=train['n_cameras'] + test['n_cameras']),
                    reinit=reinit_config(row))
                path = output / material / f'{stem}_{mode}.json'
                files[path] = json.dumps(value, indent=2, allow_nan=False) + '\n'
                paths[mode] = path.relative_to(output).as_posix()
                commands[mode] = ['python', 'train_dynamic.py' if mode == 'train' else 'test.py',
                                  '--config_path' if mode == 'train' else '--config', config_reference(path, ROOT)]
                if mode != 'train':
                    commands[mode] += ['--mode', mode]
                    if mode != 'infer':
                        commands[mode] += ['--cam-id', 'all']
            records.append(dict(case_name=case, material=material, gap=gap, configs=paths, model_path=model_dir,
                                train=train, novel=test, render_frames=render_frames, render_fps=args.original_fps,
                                infer_frames=min(args.infer_frames, train['n_frames']), commands=commands))
    manifest = dict(supp_map=str(args.supp_map).replace('\\', '/'), data_root=str(args.data_root).replace('\\', '/'),
        model_root=str(args.model_root).replace('\\', '/'), selected_cases=len(selected), skipped=skipped,
        per_material=dict(Counter(MATERIALS[row['case_name'].split('__', 1)[0]] for row in selected)),
        gaps=sorted(set(args.gap)), config_count=len(files), records=records)
    files[output / 'manifest.json'] = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
    return files, manifest


def config_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python %(prog)s --supp-map /path/map.json --data-root /path/data '
               '--model-root output --output-dir config/generated --gap 1')
    parser.add_argument('--supp-map', required=True, help='Supplementary case map JSON')
    parser.add_argument('--data-root', required=True, help='Converted dataset root containing train/test splits')
    parser.add_argument('--model-root', required=True, help='Root for per-material/per-case trained models')
    parser.add_argument('--output-dir', required=True, help='Destination for configs and manifest.json')
    parser.add_argument('--gap', nargs='+', type=int, default=[1], help='Sampling gaps (default: 1)')
    parser.add_argument('--original-fps', type=float, default=150, help='Render FPS and source FPS for missing metadata (default: 150)')
    parser.add_argument('--source-frames', type=int, default=90, help='Source frame count before sampling (default: 90)')
    parser.add_argument('--height', type=int, default=1120, help='Fallback image height (default: 1120)')
    parser.add_argument('--width', type=int, default=1120, help='Fallback image width (default: 1120)')
    parser.add_argument('--substeps-per-source-frame', type=int, default=40, help='Maximum dt control (default: 40)')
    parser.add_argument('--gs-iterations', type=int, default=40000, help='Initial deformable GS iterations (default: 40000)')
    parser.add_argument('--infer-iterations', type=int, default=40000, help='One-time fixed-pcd GS iterations (default: 40000)')
    parser.add_argument('--infer-frames', type=int, default=5, help='Training frames for inference GS fitting (default: 5)')
    parser.add_argument('--render-frames', type=int, help='Render length; defaults to --source-frames (90), independent of --gap')
    parser.add_argument('--expected-cases', type=int, default=30, help='Expected selected case count (default: 30)')
    parser.add_argument('--require-data', action='store_true', help='Reject missing or incompatible camera metadata')
    parser.add_argument('--overwrite', action='store_true', help='Update changed generated files')
    return parser


def main(argv=None):
    parser = config_parser()
    args = parser.parse_args(argv)
    for key in ('original_fps', 'source_frames', 'height', 'width', 'substeps_per_source_frame',
                'gs_iterations', 'infer_iterations', 'infer_frames', 'expected_cases'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if any(gap <= 0 for gap in args.gap) or (args.render_frames is not None and args.render_frames < 1):
        parser.error('--gap and --render-frames must be positive')
    files, manifest = build_plan(args, read_json(args.supp_map))
    written = write_plan(files, args.overwrite)
    found = sum(record[split]['source'] == 'metadata' for record in manifest['records'] for split in ('train', 'novel'))
    legacy = sum(bool(record[split]['warnings']) for record in manifest['records'] for split in ('train', 'novel'))
    print(f'Selected {manifest["selected_cases"]} cases: {manifest["per_material"]}; skipped {manifest["skipped"]}')
    print(f'Metadata: {found}/{len(manifest["records"])*2} splits found; {legacy} need calibration regeneration')
    print(f'Generated {manifest["config_count"]} configs; {written} files written including manifest')
    return manifest


if __name__ == '__main__':
    main()
