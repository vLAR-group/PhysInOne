"""Fit inference Gaussians once, then render novel/reinitialized views."""
import argparse
import json
from pathlib import Path


def config_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', '-c', required=True, help='Generated infer/novel/resimulate JSON config')
    parser.add_argument('--mode', choices=('infer', 'novel', 'resimulate', 'reinit'),
                        help='Override the stage recorded in the config')
    parser.add_argument('--cam-id', default='all', help='all or comma-separated camera indices')
    parser.add_argument('--num-frame', type=int, help='Override render length')
    parser.add_argument('--force-infer', action='store_true', help='Refit inference Gaussians even when cached')
    parser.add_argument('--refresh-simulation', action='store_true', help='Regenerate the shared trajectory')
    parser.add_argument('--write-video', action='store_true', help='Encode video.mp4 per camera after JPEG export')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS (default: 30)')
    return parser


def main(argv=None):
    parser = config_parser()
    args = parser.parse_args(argv)
    if args.fps < 1 or (args.num_frame is not None and args.num_frame < 1):
        parser.error('--fps and --num-frame must be positive')
    path = Path(args.config).resolve()
    with path.open(encoding='utf-8') as stream:
        cfg = json.load(stream)
    mode = args.mode or cfg.get('stage')
    mode = 'resimulate' if mode == 'reinit' else mode
    if mode not in ('infer', 'novel', 'resimulate'):
        parser.error('Select --mode infer, novel or resimulate')
    from inference import run
    return run(cfg, path, mode, args)


if __name__ == '__main__':
    main()
