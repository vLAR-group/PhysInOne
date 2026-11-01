import argparse
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np


BLENDER_TO_NERF = np.asarray(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ORIGIN_CASE_TYPES = {"origin", "ori"}
VAL_JSON_SPLIT = "val-json"
TRAIN_ONLY_SPLIT = "train-only"
RANDOM_SPLIT = "random"


def natural_sort_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def compute_intrinsic(camera_angle_x, img_w, img_h):
    fx = 0.5 * img_w / np.tan(0.5 * camera_angle_x)
    fy = fx
    cx = img_w / 2
    cy = img_h / 2
    return [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]


def convert_c2w_to_nerf(transform_matrix):
    c2w = np.asarray(transform_matrix, dtype=np.float64)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, [0, 0, 0, 1]])
    if c2w.shape != (4, 4):
        raise ValueError(f"Unsupported transform_matrix shape: {c2w.shape}")
    return (BLENDER_TO_NERF @ c2w)[:3, :].tolist()


def parse_camera_index(camera_name):
    match = re.search(r"_(\d+)$", camera_name)
    if not match:
        raise ValueError(f"Cannot parse camera index from {camera_name}")
    return int(match.group(1))


def detect_test_cameras(val_json_path):
    if not val_json_path or not val_json_path.exists():
        return set()

    data = load_json(val_json_path)
    test_cameras = set()
    for frame in data.get("frames", []):
        camera_name = frame["file_path"].replace("\\", "/").split("/")[0]
        if camera_name == "CineCamera_Moving":
            continue
        test_cameras.add(parse_camera_index(camera_name))
    return test_cameras


def read_camera_list(path):
    with path.open("r", encoding="utf-8") as f:
        camera_names = (line.strip() for line in f)
        return [
            camera_name for camera_name in camera_names
            if camera_name and camera_name != "CineCamera_Moving"
        ]


def camera_name_from_json_path(path):
    return path.stem.replace("blender_", "", 1)


def discover_camera_list_from_json(dataset_path):
    camera_names = []
    for json_path in sorted(dataset_path.glob("blender_CineCamera_*.json"), key=natural_sort_key):
        camera_name = camera_name_from_json_path(json_path)
        if camera_name != "CineCamera_Moving":
            camera_names.append(camera_name)
    return camera_names


def iter_images(rgb_dir):
    if not rgb_dir.exists():
        return []
    return sorted(
        [path for path in rgb_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_sort_key,
    )


def has_rendered_rgb(dataset_path, camera_name):
    return bool(iter_images(Path(dataset_path) / camera_name / "rgb"))


def normalize_case_type(reinit_data):
    case_type = reinit_data.get("case_type", reinit_data.get("type", ""))
    return str(case_type).strip().lower()


def load_reinit_metadata(dataset_path):
    reinit_path = dataset_path / "reinit.json"
    if not reinit_path.exists():
        return {}
    return load_json(reinit_path)


def is_origin_dataset(dataset_path):
    if not (dataset_path / "reinit.json").exists():
        return True
    reinit_data = load_reinit_metadata(dataset_path)
    return normalize_case_type(reinit_data) in ORIGIN_CASE_TYPES


def has_camera_metadata(dataset_path):
    return any(Path(dataset_path).glob("blender_CineCamera_*.json"))


def is_dataset_folder(dataset_path):
    dataset_path = Path(dataset_path)
    return (dataset_path / "static_camera_list.txt").exists() or has_camera_metadata(dataset_path)


def get_particle_mesh_names(dataset_path):
    particle_materials_path = dataset_path / "particle_materials.json"
    if not particle_materials_path.exists():
        raise FileNotFoundError(f"Missing particle_materials.json: {particle_materials_path}")

    particle_materials = load_json(particle_materials_path)
    if isinstance(particle_materials, dict):
        return set(particle_materials.keys())
    if isinstance(particle_materials, list):
        return {
            str(item)
            for item in particle_materials
            if isinstance(item, str)
        }
    raise ValueError(f"Unsupported particle_materials format: {particle_materials_path}")


def get_stencil_value(object_info):
    components = object_info.get("components", [])
    for component in components:
        stencil_value = component.get("custom_stencil_value")
        if stencil_value is not None:
            return int(stencil_value)
    return None


def get_foreground_stencil_values(dataset_path):
    mesh_names = get_particle_mesh_names(dataset_path)
    recorder_stats_path = dataset_path / "recorder_stats.json"
    if not recorder_stats_path.exists():
        raise FileNotFoundError(f"Missing recorder_stats.json: {recorder_stats_path}")

    recorder_stats = load_json(recorder_stats_path)
    stencil_values = set()

    for group_name in ("solid_object", "interactable_object", "breakable_object"):
        for item in recorder_stats.get(group_name, []):
            if not isinstance(item, dict):
                continue
            for mesh_name, object_info in item.items():
                if mesh_name not in mesh_names or not isinstance(object_info, dict):
                    continue
                stencil_value = get_stencil_value(object_info)
                if stencil_value is not None:
                    stencil_values.add(stencil_value)

    if not stencil_values:
        raise ValueError(
            f"Cannot find custom_stencil_value for particle meshes "
            f"{sorted(mesh_names)} in {recorder_stats_path}"
        )
    return stencil_values


def write_mask_images(rgb_path, seg_path, mask_path, foreground_values):
    import cv2

    img = cv2.imread(str(rgb_path))
    if img is None:
        raise ValueError(f"Cannot read image: {rgb_path}")

    seg_mask = np.load(seg_path)["seg"]
    if seg_mask.ndim == 3:
        seg_mask = np.squeeze(seg_mask)
    if seg_mask.shape != img.shape[:2]:
        seg_mask = cv2.resize(
            seg_mask,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    foreground = np.isin(seg_mask, list(foreground_values))
    masked = img.copy()
    masked[~foreground] = 255
    cv2.imwrite(str(mask_path), masked)

    alpha_path = mask_path.with_name(mask_path.name.replace("m_", "a_", 1))
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = foreground.astype(np.uint8) * 255
    cv2.imwrite(str(alpha_path), rgba)


def copy_frame(src_path, dst_path, copy_images):
    if copy_images:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def build_frame_info(file_name, frame_idx, effective_fps, c2w, intrinsic):
    return {
        "file_path": f"./data/{file_name}",
        "time": frame_idx / effective_fps,
        "c2w": c2w,
        "intrinsic": intrinsic,
    }


def process_camera(
    dataset_path,
    camera_name,
    test_cameras,
    output_dirs,
    counters,
    gap,
    effective_fps,
    copy_images,
    write_masks,
    copy_background,
    foreground_values,
    allow_missing_images=False,
):
    if camera_name == "CineCamera_Moving":
        return

    camera_idx = parse_camera_index(camera_name)
    camera_json = dataset_path / f"blender_{camera_name}.json"
    if not camera_json.exists():
        print(f"Warning: missing camera json, skip {camera_name}: {camera_json}")
        return

    cam_data = load_json(camera_json)
    frame0 = cam_data["frames"][0]
    c2w = convert_c2w_to_nerf(frame0["transform_matrix"])
    intrinsic = compute_intrinsic(
        cam_data["camera_angle_x"],
        cam_data["img_w"],
        cam_data["img_h"],
    )

    image_paths = iter_images(dataset_path / camera_name / "rgb")
    if not image_paths:
        if not allow_missing_images:
            print(f"Warning: no rgb images, skip {camera_name}")
            return
        sampled_frames = cam_data.get("frames", [])[::gap]
        if not sampled_frames:
            print(f"Warning: no rgb images or json frames, skip {camera_name}")
            return
    else:
        sampled_frames = image_paths[::gap]

    split = "test" if camera_idx in test_cameras else "train"
    new_camera_idx = counters[split]
    counters[split] += 1

    data_dir = output_dirs[split] / "data"
    info_list = output_dirs[f"{split}_info"]

    if copy_background:
        bg_src = dataset_path / camera_name / "bg" / "bg_0001.jpg"
        if bg_src.exists():
            bg_name = f"r_{new_camera_idx}_-1.jpg"
            copy_frame(bg_src, data_dir / bg_name, copy_images)
            info_list.append(
                build_frame_info(bg_name, -1, effective_fps, c2w, intrinsic)
            )

    for new_frame_idx, frame_src in enumerate(sampled_frames):
        rgb_src = frame_src if isinstance(frame_src, Path) else None
        suffix = rgb_src.suffix.lower() if rgb_src else ".jpg"
        rgb_name = f"r_{new_camera_idx}_{new_frame_idx}{suffix}"
        if rgb_src:
            rgb_dst = data_dir / rgb_name
            copy_frame(rgb_src, rgb_dst, copy_images)

        if write_masks and rgb_src:
            seg_src = dataset_path / camera_name / "seg" / f"{rgb_src.stem}.npz"
            if seg_src.exists():
                mask_dst = data_dir / f"m_{new_camera_idx}_{new_frame_idx}.png"
                write_mask_images(rgb_src, seg_src, mask_dst, foreground_values)
            else:
                print(f"Warning: missing seg file: {seg_src}")

        info_list.append(
            build_frame_info(rgb_name, new_frame_idx, effective_fps, c2w, intrinsic)
        )

    return new_camera_idx


def create_dataset(
    dataset_path,
    output_root,
    dataset_name,
    gap,
    static_camera_list_path=None,
    val_json_path=None,
    original_fps=150,
    copy_images=True,
    write_masks=True,
    copy_background=True,
    foreground_stencil_values=None,
    test_split_mode=VAL_JSON_SPLIT,
    overwrite=False,
    test_camera_count=None,
    random_seed=42,
):
    dataset_path = Path(dataset_path)
    output_root = Path(output_root)
    static_camera_list_path = (
        Path(static_camera_list_path)
        if static_camera_list_path
        else dataset_path / "static_camera_list.txt"
    )
    val_json_path = Path(val_json_path or dataset_path / "transforms_val.json")
    output_path = output_root / f"{dataset_name}_gap{gap}"

    train_dir = output_path / "train"
    test_dir = output_path / "test"

    effective_fps = original_fps / gap
    train_cameras = []
    test_camera_names = []
    if test_split_mode == TRAIN_ONLY_SPLIT:
        camera_list = discover_camera_list_from_json(dataset_path)
        for camera_name in camera_list:
            if has_rendered_rgb(dataset_path, camera_name):
                train_cameras.append(camera_name)
            else:
                test_camera_names.append(camera_name)
        test_cameras = {parse_camera_index(camera_name) for camera_name in test_camera_names}
    else:
        camera_list = (
            read_camera_list(static_camera_list_path)
            if static_camera_list_path.exists()
            else discover_camera_list_from_json(dataset_path)
        )
        if test_split_mode == RANDOM_SPLIT:
            if test_camera_count is None or test_camera_count <= 0:
                raise ValueError("random mode requires a positive --test-camera-count")
            available_cameras = []
            for camera_name in sorted(set(camera_list), key=lambda name: natural_sort_key(Path(name))):
                camera_json = dataset_path / f"blender_{camera_name}.json"
                if camera_json.exists() and has_rendered_rgb(dataset_path, camera_name):
                    available_cameras.append(camera_name)
                else:
                    print(f"Warning: missing camera json or rgb images, exclude {camera_name} from random split")
            camera_list = available_cameras
            if test_camera_count >= len(camera_list):
                raise ValueError(
                    f"--test-camera-count must be smaller than the number of available "
                    f"static cameras ({len(camera_list)}), got {test_camera_count}; "
                    "at least one training camera is required"
                )
            selected_cameras = random.Random(random_seed).sample(camera_list, test_camera_count)
            test_cameras = {parse_camera_index(name) for name in selected_cameras}
        else:
            test_cameras = detect_test_cameras(val_json_path)
        for camera_name in camera_list:
            target = test_camera_names if parse_camera_index(camera_name) in test_cameras else train_cameras
            target.append(camera_name)
    if write_masks and foreground_stencil_values is None:
        foreground_stencil_values = get_foreground_stencil_values(dataset_path)
    foreground_stencil_values = {
        int(value) for value in (foreground_stencil_values or [])
    }

    if overwrite and output_path.exists():
        shutil.rmtree(output_path)
    (train_dir / "data").mkdir(parents=True, exist_ok=True)
    (test_dir / "data").mkdir(parents=True, exist_ok=True)

    output_dirs = {
        "train": train_dir,
        "test": test_dir,
        "train_info": [],
        "test_info": [],
    }
    counters = {"train": 0, "test": 0}
    camera_mapping = {"train": {}, "test": {}}

    for camera_name in camera_list:
        new_camera_idx = process_camera(
            dataset_path=dataset_path,
            camera_name=camera_name,
            test_cameras=test_cameras,
            output_dirs=output_dirs,
            counters=counters,
            gap=gap,
            effective_fps=effective_fps,
            copy_images=copy_images,
            write_masks=write_masks,
            copy_background=copy_background,
            foreground_values=foreground_stencil_values,
            allow_missing_images=(
                test_split_mode == TRAIN_ONLY_SPLIT
                and parse_camera_index(camera_name) in test_cameras
            ),
        )
        if new_camera_idx is not None:
            split = "test" if parse_camera_index(camera_name) in test_cameras else "train"
            camera_mapping[split][str(new_camera_idx)] = camera_name

    dump_json(train_dir / "all_data.json", output_dirs["train_info"])
    dump_json(test_dir / "all_data.json", output_dirs["test_info"])
    dump_json(output_path / "camera_mapping.json", camera_mapping)

    print(f"Done: {output_path}")
    print(f"Train cameras: {counters['train']}, test cameras: {counters['test']}")
    if test_split_mode in (TRAIN_ONLY_SPLIT, RANDOM_SPLIT):
        print(f"Train camera names: {train_cameras}")
        print(f"Test camera names: {test_camera_names}")
    if test_split_mode == RANDOM_SPLIT:
        print(f"Random seed: {random_seed}")
    print(
        "Frames: "
        f"train={len(output_dirs['train_info'])}, "
        f"test={len(output_dirs['test_info'])}, "
        f"total={len(output_dirs['train_info']) + len(output_dirs['test_info'])}"
    )
    print(f"Effective FPS: {effective_fps}")
    if write_masks:
        print(f"Foreground stencil values: {sorted(foreground_stencil_values)}")
    else:
        print("Mask generation: disabled")


def resolve_case_path(origin_path, case_name):
    origin_path = Path(origin_path)
    case_path = origin_path / case_name
    if case_path.exists():
        return case_path

    if not case_name.endswith("_trajectory"):
        case_path = origin_path / f"{case_name}_trajectory"
        if case_path.exists():
            return case_path

    raise FileNotFoundError(f"Cannot find case '{case_name}' under {origin_path}")


def validate_case_path(case_path):
    case_path = Path(case_path)
    if not is_dataset_folder(case_path):
        raise FileNotFoundError(
            f"Missing static_camera_list.txt or blender_CineCamera_*.json: {case_path}"
        )
    if not is_origin_dataset(case_path):
        reinit_data = load_reinit_metadata(case_path)
        case_type = normalize_case_type(reinit_data) or "unknown"
        paired_case = reinit_data.get("paired_case", "")
        paired_text = f", paired_case={paired_case}" if paired_case else ""
        raise ValueError(
            f"Case is not origin: {case_path.name} "
            f"(case_type={case_type}{paired_text})"
        )
    return case_path


def discover_datasets(origin_path=None, case_name=None, case_path=None):
    if case_path:
        case_path = validate_case_path(case_path)
        return [(case_path, case_path.name)]

    if not origin_path:
        raise ValueError("Either --origin-path or --case-path is required.")

    origin_path = Path(origin_path)
    if case_name:
        case_path = resolve_case_path(origin_path, case_name)
        case_path = validate_case_path(case_path)
        return [(case_path, case_path.name)]

    if is_dataset_folder(origin_path):
        return [(origin_path, origin_path.name)] if is_origin_dataset(origin_path) else []
    return [
        (path, path.name)
        for path in sorted(origin_path.iterdir())
        if path.is_dir()
        and is_dataset_folder(path)
        and is_origin_dataset(path)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create PAC-NeRF/GIC baseline datasets with configurable gap and paths."
    )
    parser.add_argument("--origin-path", help="Dataset folder or root folder containing datasets.")
    parser.add_argument("--output-path", required=True, help="Output root folder.")
    parser.add_argument("--case-path", help="Build only this explicit case folder path.")
    parser.add_argument(
        "--case-name",
        help="Build only this case under --origin-path. The _trajectory suffix is optional.",
    )
    parser.add_argument("--dataset-name", help="Output dataset name when --origin-path points to one dataset.")
    parser.add_argument("--gap", type=int, nargs="+", required=True, help="One or more frame sampling gaps.")
    parser.add_argument("--static-camera-list-path", help="Override static_camera_list.txt path for one dataset.")
    parser.add_argument("--val-json-path", help="Override transforms_val.json path for one dataset.")
    parser.add_argument(
        "--test-split-mode",
        choices=[VAL_JSON_SPLIT, TRAIN_ONLY_SPLIT, RANDOM_SPLIT],
        default=VAL_JSON_SPLIT,
        help=(
            "Choose test camera split logic. "
            "val-json uses transforms_val.json; train-only treats cameras without rendered rgb as test; "
            "random samples --test-camera-count static cameras with rendered rgb."
        ),
    )
    parser.add_argument(
        "--test-camera-count", type=int,
        help="Number of test cameras in random mode; must leave at least one training camera.",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random camera split seed (default: 42); the same cameras and seed reproduce the split.",
    )
    parser.add_argument("--original-fps", type=float, default=150.0)
    parser.add_argument(
        "--foreground-stencil-values",
        type=int,
        nargs="+",
        help="Override foreground seg values. By default they are read from particle_materials.json and recorder_stats.json.",
    )
    parser.add_argument("--no-copy-images", action="store_true", help="Only write all_data.json metadata.")
    parser.add_argument("--no-masks", action="store_true", help="Do not generate m_*.png and a_*.png files.")
    parser.add_argument("--no-background", action="store_true", help="Do not add r_*_-1 background frames.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output dataset before writing.")
    return parser.parse_args()


def main():
    args = parse_args()
    if any(gap <= 0 for gap in args.gap):
        raise ValueError("--gap must be positive")

    datasets = discover_datasets(args.origin_path, args.case_name, args.case_path)
    if not datasets:
        raise FileNotFoundError(
            f"No origin dataset with static_camera_list.txt found under {args.origin_path}"
        )

    if len(datasets) > 1 and (
        args.dataset_name
        or args.static_camera_list_path
        or args.val_json_path
    ):
        raise ValueError(
            "--dataset-name, --static-camera-list-path and --val-json-path "
            "are only valid when --origin-path points to one dataset."
        )

    for dataset_path, discovered_name in datasets:
        dataset_name = args.dataset_name or discovered_name
        for gap in args.gap:
            create_dataset(
                dataset_path=dataset_path,
                output_root=args.output_path,
                dataset_name=dataset_name,
                gap=gap,
                static_camera_list_path=args.static_camera_list_path,
                val_json_path=args.val_json_path,
                original_fps=args.original_fps,
                copy_images=not args.no_copy_images,
                write_masks=(not args.no_masks) and (not args.no_copy_images),
                copy_background=not args.no_background,
                foreground_stencil_values=args.foreground_stencil_values,
                test_split_mode=args.test_split_mode,
                overwrite=args.overwrite,
                test_camera_count=args.test_camera_count,
                random_seed=args.random_seed,
            )


if __name__ == "__main__":
    main()
