"""Camera metadata and streamed property_estimation RGB export."""
import json
import os
import warnings
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def read_camera_metadata(data_folder):
    with open(Path(data_folder) / "all_data.json", encoding="utf-8") as f:
        entries = json.load(f)
    cameras = {}
    backgrounds = {}
    for entry in entries:
        filename = Path(entry["file_path"].replace("\\", "/")).stem
        prefix, cam_id, frame_id = filename.rsplit("_", 2)
        cam_id, frame_id = int(cam_id), int(frame_id)
        if prefix != "r":
            continue
        if frame_id == -1:
            backgrounds[cam_id] = Path(data_folder) / entry["file_path"]
        if frame_id >= 0:
            cameras.setdefault(cam_id, entry)
    if not cameras:
        raise ValueError(f"No render cameras found in {data_folder}/all_data.json")
    return cameras, backgrounds


def load_camera_rays(data_folder, num_views, H, W, *, cameras=None):
    # Evaluation only needs camera calibration, not all ground-truth RGB frames.
    import torch
    from lib.nerf import get_rays_of_a_view

    if cameras is None:
        cameras, _ = read_camera_metadata(data_folder)
    expected = set(range(num_views))
    if set(cameras) != expected:
        raise ValueError(f"Expected camera indices {sorted(expected)}, got {sorted(cameras)}")
    all_rays = []
    for cam_id in range(num_views):
        entry = cameras[cam_id]
        all_rays.append(get_rays_of_a_view(
            H, W, torch.tensor(entry["intrinsic"]), torch.tensor(entry["c2w"])))
    return tuple(torch.stack([rays[i].reshape(-1, 3) for rays in all_rays])
                 for i in range(3))


def load_camera_mapping(data_folder, mapping_path=None, split=None, *, required=False):
    """Use the active dataset split: train/test indices are independent."""
    data_folder = Path(data_folder)
    split = split or (data_folder.name if data_folder.name in ("train", "test") else None)
    candidates = [Path(mapping_path)] if mapping_path else [
        data_folder / "camera_mapping.json",
        data_folder.parent / "camera_mapping.json",
        data_folder.parent / "train" / "camera_mapping.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with candidate.open(encoding="utf-8") as f:
            mapping = json.load(f)
        if "train" in mapping or "test" in mapping:
            if split is None:
                raise ValueError(f"Select --camera-split train/test for {candidate}")
            if split not in mapping:
                raise ValueError(f"No {split!r} camera mapping in {candidate}")
            mapping = mapping[split]
        for key, name in mapping.items():
            if not isinstance(name, str) or not name.startswith("CineCamera_") or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError(f"Invalid camera name for index {key}: {name!r}")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(f"Duplicate camera names in {candidate}")
        return {int(key): name for key, name in mapping.items()}
    if mapping_path:
        raise FileNotFoundError(mapping_path)
    if required:
        raise FileNotFoundError(
            f"Resimulation requires camera_mapping.json for {data_folder}; "
            "train/test local indices cannot identify the original cameras. "
            "Regenerate the dataset or supply --camera-mapping with both split mappings.")
    warnings.warn("camera_mapping.json not found; using original camera indices.")
    return None


def load_render_cameras(cfg, args, mode):
    """Merge reinit calibration by original name, never by split-local index."""
    if mode != 'reinit':
        cameras, backgrounds = read_camera_metadata(cfg['data_dir'])
        mapping = load_camera_mapping(cfg['data_dir'], args.camera_mapping, args.camera_split)
        return cameras, mapping, backgrounds

    if args.camera_split is not None:
        raise ValueError('Resimulation always loads both train and test; omit --camera-split')
    folder = Path(cfg['data_dir'])
    if folder.name in ('train', 'test'):
        folder = folder.parent
    by_name = {}
    for split in ('train', 'test'):
        split_folder = folder / split
        cameras, backgrounds = read_camera_metadata(split_folder)
        mapping = load_camera_mapping(split_folder, args.camera_mapping, split, required=True)
        if set(mapping) != set(cameras):
            raise ValueError(f'{split_folder}: camera mapping indices {sorted(mapping)} '
                             f'do not match metadata indices {sorted(cameras)}')
        for local_id, entry in cameras.items():
            name = mapping[local_id]
            if name in by_name:
                raise ValueError(f'Camera {name} appears in both train and test')
            by_name[name] = (entry, backgrounds.get(local_id))
    if len(by_name) != cfg['n_cameras']:
        raise ValueError(f'Resimulation found {len(by_name)} train+test cameras, '
                         f"but n_cameras={cfg['n_cameras']}; regenerate the resimulate config")

    def camera_order(name):
        suffix = name[len('CineCamera_'):]
        return (0, int(suffix), name) if suffix.isdigit() else (1, suffix, name)

    # Dense render IDs follow original camera order; output directories retain names.
    mapping = dict(enumerate(sorted(by_name, key=camera_order)))
    cameras = {index: by_name[name][0] for index, name in mapping.items()}
    backgrounds = {index: by_name[name][1] for index, name in mapping.items()
                   if by_name[name][1] is not None}
    return cameras, mapping, backgrounds


def camera_output_dir(output_dir, mode, cam_id, mapping):
    name = f"CineCamera_{cam_id}" if mapping is None else mapping[cam_id]
    return Path(output_dir) / mode / name / "rgb"


def property_output_dir(cfg, args):
    return args.output_dir or os.path.join(cfg["base_dir"], "property_estimation")


def compose_rgb_frame(rgb, bgmap, background=None):
    # rgb is already composited over white by DirectVoxGO (bg=1).
    # Replace that background using transmittance; do not multiply alpha twice.
    rgb = np.asarray(rgb)
    if background is not None:
        rgb = rgb + np.asarray(bgmap) * (np.asarray(background, dtype=np.float32) / 255.0 - 1.0)
    return np.rint(255.0 * np.clip(rgb, 0, 1)).astype(np.uint8)


def prepare_property_output(cfg, args, mode, camera_ids, *, camera_metadata=None):
    output_root = property_output_dir(cfg, args)
    if Path(output_root).suffix.lower() == '.json':
        raise ValueError('--output-dir/--params-output expects a directory, not a JSON file')
    if camera_metadata is None:
        camera_metadata = load_render_cameras(cfg, args, mode)
    _, mapping, paths = camera_metadata
    if mapping is not None and not set(camera_ids).issubset(mapping):
        raise ValueError(f'Missing camera mappings for {sorted(set(camera_ids) - set(mapping))}')
    outputs = {cam: camera_output_dir(output_root, mode, cam, mapping) for cam in camera_ids}
    backgrounds = {}
    for cam in camera_ids:
        background = None
        path = paths.get(cam)
        if not args.alpha_output and path is not None:
            if not path.is_file():
                raise FileNotFoundError(f'Missing camera background: {path}')
            background = np.asarray(imageio.imread(path))[..., :3]
            expected = (cfg['H'], cfg['W'], 3)
            if background.shape != expected:
                raise ValueError(f'Background {path} has shape {background.shape}, expected {expected}')
        backgrounds[cam] = background
    return outputs, backgrounds


def write_camera_videos(outputs, args):
    for directory in outputs.values():
        with imageio.get_writer(directory.parent / 'video.mp4', fps=args.fps, quality=8) as video:
            for frame_id in range(args.num_frame):
                video.append_data(imageio.imread(directory / f'{frame_id:04d}.jpg'))


def render_property_views(pnerf, cfg, args, camera_ids, rays, trajectory, outputs, backgrounds):
    for directory in outputs.values():
        directory.mkdir(parents=True, exist_ok=True)
        # A shorter rerun must not leave stale frames mixed into this sequence.
        for path in directory.glob('*.jpg'):
            if path.stem.isdigit() and path.is_file():
                path.unlink()

    def on_frame(frame_id, cam_id, rgb, bgmap):
        frame = compose_rgb_frame(rgb, bgmap, backgrounds[cam_id])
        imageio.imwrite(outputs[cam_id] / f'{frame_id:04d}.jpg', frame, quality=95)

    pnerf.render_sequence_multiview(
        args.num_frame, H=cfg['H'], W=cfg['W'], camera_ids=camera_ids,
        rays_o_all=rays[0], rays_d_all=rays[1], viewdirs_all=rays[2],
        trajectory=trajectory, on_frame=on_frame)
    if args.write_video:
        write_camera_videos(outputs, args)
    return outputs
