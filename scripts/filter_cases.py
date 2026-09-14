#!/usr/bin/env python3
"""
Filter PhysInOne cases from repo_assignment.txt.

Expected assignment file format:
    /Game/PhysInOne/Scenes/Train/DoublePhysics/CaseName.CaseName    physinone_part1

The script parses:
    split: Train / Val / Test
    activity_type: SinglePhysics / DoublePhysics / TriplePhysics
    phenomena: abbreviations before "__bg..."
    background: bgXXX
    hash: final six-character code
    part_id: e.g. physinone_part1
    hf_zip_path: expected zip path inside the shard repo, e.g.
        Train/DoublePhysics/AccelConcaveSpin_AccelSurfaceSpin__bg070__K5ER39_trajectory.zip
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional


DEFAULT_ASSIGNMENT_FILE = Path(__file__).resolve().parent / "download_lists" / "repo_assignment.txt"


SUPPORTED_PHENOMENA = {
    "MovingHitsFixed",
    "MovingHitsStationary",
    "MovingHitsMoving",
    "WindGravityBalance",
    "WindPushStationary",
    "WindPushSameDir",
    "WindPushOppDir",
    "WindDeflectMotion",
    "ObliqueProjectile",
    "VerticalFall",
    "RollDownSlope",
    "RollUpSlope",
    "MagnetAttract",
    "MagnetRepel",
    "UniformPanelSpin",
    "AccelPanelSpin",
    "UniformConcaveSpin",
    "AccelConcaveSpin",
    "UniformSurfaceSpin",
    "AccelSurfaceSpin",
    "FrictionStop",
    "SpringCompress",
    "SpringStretch",
    "ImpactFracture",
    "MirrorFragmentReflect",
    "ElasticCouple",
    "SpringboardRebound",
    "SeesawCenterPivot",
    "SeesawOffsetPivot",
    "BalloonFloat",
    "BalloonTether",
    "BalloonLift",
    "FixedPlanarRedirect",
    "FixedArrayRedirect",
    "FixedConcaveRedirect",
    "FixedConvexRedirect",
    "DynMirrorRedirect",
    "LaserBlock",
    "MirrorReflect",
    "CartMove",
    "RotTurnableInertia",
    "RotBoardInertia",
    "LinCarryInertia",
    "CatapultLaunch",
    "ChainSuspend",
    "SimplePendulum",
    "DoublePendulum",
    "CrankPush",
    "BlockWallCollapse",
    "StickSupportFail",
    "FloatOnLiquid",
    "DropInLiquid",
    "MovingObjDriveLiquid",
    "LiquidCarryMovingObj",
    "LiquidHitFixedObj",
    "LiquidTransfer",
    "LiquidMultiTransfers",
    "LiquidThroughGrid",
    "LiquidAcrossUneven",
    "LiquidRise",
    "LiquidAlongContours",
    "JetLiquid",
    "LiquidTension",
    "LiquidRefraction",
    "StickyToObjects",
    "StickyFromObjects",
    "ElasticFall",
    "PlasticineFall",
    "NewtonianFluidFall",
    "NonNewtonianFluidFall",
    "GranularFall",
}


SPLIT_ALIASES = {
    "train": "Train",
    "val": "Val",
    "validation": "Val",
    "test": "Test",
}

ACTIVITY_ALIASES = {
    "single": "SinglePhysics",
    "singlephysics": "SinglePhysics",
    "double": "DoublePhysics",
    "doublephysics": "DoublePhysics",
    "triple": "TriplePhysics",
    "triplephysics": "TriplePhysics",
}


@dataclass
class CaseRecord:
    case_id: str
    split: str
    activity_type: str
    phenomena: list[str]
    background: str
    hash: str
    ue_path: str
    part_id: str
    hf_zip_path: str


def normalize_split(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = value.strip().lower()
    if key not in SPLIT_ALIASES:
        raise ValueError(
            f"Unknown split '{value}'. Expected one of: train, val, test."
        )
    return SPLIT_ALIASES[key]


def normalize_activity(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = value.strip().lower()
    if key not in ACTIVITY_ALIASES:
        raise ValueError(
            f"Unknown activity_type '{value}'. Expected one of: single, double, triple."
        )
    return ACTIVITY_ALIASES[key]


def validate_phenomena(values: Optional[list[str]]) -> Optional[list[str]]:
    if not values:
        return values

    invalid = sorted(set(values) - SUPPORTED_PHENOMENA)
    if invalid:
        valid_preview = ", ".join(sorted(SUPPORTED_PHENOMENA))
        raise ValueError(
            "Unsupported phenomenon abbreviation(s): "
            + ", ".join(invalid)
            + "\nExpected one or more of the 71 supported abbreviations:\n"
            + valid_preview
        )
    return values


def parse_case_line(line: str, line_no: int) -> Optional[CaseRecord]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = re.split(r"\s+", line, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Line {line_no}: expected two columns: <ue_path> <part_id>")

    ue_path, part_id = parts[0], parts[1].strip()
    path_parts = ue_path.split("/")

    try:
        scenes_idx = path_parts.index("Scenes")
        split = path_parts[scenes_idx + 1]
        activity_type = path_parts[scenes_idx + 2]
        object_ref = path_parts[scenes_idx + 3]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Line {line_no}: cannot parse UE path: {ue_path}") from exc

    if split not in {"Train", "Val", "Test"}:
        raise ValueError(
            f"Line {line_no}: unsupported split '{split}' in assignment file. "
            "Expected Train, Val, or Test."
        )

    if activity_type not in {"SinglePhysics", "DoublePhysics", "TriplePhysics"}:
        raise ValueError(
            f"Line {line_no}: unsupported activity type '{activity_type}' in assignment file. "
            "Expected SinglePhysics, DoublePhysics, or TriplePhysics."
        )

    case_id = object_ref.split(".")[-1]

    if "__" not in case_id:
        raise ValueError(f"Line {line_no}: cannot parse case_id: {case_id}")

    segments = case_id.split("__")
    phenomenon_token = segments[0]
    phenomena = phenomenon_token.split("_") if phenomenon_token else []

    invalid_in_file = sorted(set(phenomena) - SUPPORTED_PHENOMENA)
    if invalid_in_file:
        raise ValueError(
            f"Line {line_no}: assignment file contains unsupported phenomenon "
            f"abbreviation(s): {', '.join(invalid_in_file)} in case_id {case_id}"
        )

    expected_count = {
        "SinglePhysics": 1,
        "DoublePhysics": 2,
        "TriplePhysics": 3,
    }[activity_type]
    if len(phenomena) != expected_count:
        raise ValueError(
            f"Line {line_no}: activity type {activity_type} expects "
            f"{expected_count} phenomenon abbreviation(s), but got {len(phenomena)} "
            f"in case_id {case_id}"
        )

    background = ""
    hash_code = ""
    for segment in segments[1:]:
        if re.fullmatch(r"bg\d+", segment):
            background = segment
        else:
            hash_code = segment

    if not background:
        raise ValueError(f"Line {line_no}: background segment not found in case_id: {case_id}")
    if not hash_code:
        raise ValueError(f"Line {line_no}: hash segment not found in case_id: {case_id}")

    hf_zip_path = f"{split}/{activity_type}/{case_id}_trajectory.zip"

    return CaseRecord(
        case_id=case_id,
        split=split,
        activity_type=activity_type,
        phenomena=phenomena,
        background=background,
        hash=hash_code,
        ue_path=ue_path,
        part_id=part_id,
        hf_zip_path=hf_zip_path,
    )


def load_assignment(path: Path) -> list[CaseRecord]:
    if not path.exists():
        raise FileNotFoundError(
            f"Assignment file not found: {path}\n"
            f"Put repo_assignment.txt at {DEFAULT_ASSIGNMENT_FILE} or pass --assignment_file."
        )

    records: list[CaseRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            record = parse_case_line(line, line_no)
            if record is not None:
                records.append(record)
    return records


def record_matches_phenomena(
    record: CaseRecord,
    requested: Iterable[str] | None,
    match_mode: Literal["contains", "exact"],
) -> bool:
    requested_list = list(requested or [])
    if not requested_list:
        return True

    requested_set = set(requested_list)
    record_set = set(record.phenomena)

    if match_mode == "contains":
        return requested_set.issubset(record_set)
    if match_mode == "exact":
        return requested_set == record_set

    raise ValueError(f"Unsupported match_mode: {match_mode}")


def filter_records(
    records: list[CaseRecord],
    split: Optional[str] = None,
    activity_type: Optional[str] = None,
    phenomena: Optional[list[str]] = None,
    match_mode: Literal["contains", "exact"] = "contains",
) -> list[CaseRecord]:
    split_norm = normalize_split(split)
    activity_norm = normalize_activity(activity_type)
    phenomena = validate_phenomena(phenomena)

    out = []
    for record in records:
        if split_norm is not None and record.split != split_norm:
            continue
        if activity_norm is not None and record.activity_type != activity_norm:
            continue
        if not record_matches_phenomena(record, phenomena, match_mode):
            continue
        out.append(record)
    return out


def build_stats(records: list[CaseRecord]) -> dict:
    stats: dict = {
        "num_cases": len(records),
        "by_split": {},
        "by_activity_type": {},
        "by_part": {},
    }
    for r in records:
        stats["by_split"][r.split] = stats["by_split"].get(r.split, 0) + 1
        stats["by_activity_type"][r.activity_type] = stats["by_activity_type"].get(r.activity_type, 0) + 1
        stats["by_part"][r.part_id] = stats["by_part"].get(r.part_id, 0) + 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter PhysInOne cases.")
    parser.add_argument(
        "--assignment_file",
        type=Path,
        default=DEFAULT_ASSIGNMENT_FILE,
        help=f"Path to repo_assignment.txt. Default: {DEFAULT_ASSIGNMENT_FILE}",
    )
    parser.add_argument("--split", type=str, default=None, help="train, val, or test.")
    parser.add_argument(
        "--activity_type",
        type=str,
        default=None,
        help="single, double, or triple.",
    )
    parser.add_argument(
        "--phenomena",
        nargs="*",
        default=None,
        help="One or more phenomenon abbreviations, e.g. AccelConcaveSpin FrictionStop.",
    )
    parser.add_argument(
        "--match_mode",
        choices=["contains", "exact"],
        default="contains",
        help=(
            "contains: selected cases contain all requested phenomena; "
            "exact: selected cases have exactly the requested phenomena. "
            "Order is ignored in both modes."
        ),
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Global number of cases to sample after filtering. Default: keep all matches.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --num is specified. Default: 42.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("selected_cases.json"),
        help="Output JSON path. Default: selected_cases.json",
    )
    parser.add_argument(
        "--show_stats",
        action="store_true",
        help="Print dataset/filtering statistics.",
    )

    args = parser.parse_args()

    records = load_assignment(args.assignment_file)
    matches = filter_records(
        records,
        split=args.split,
        activity_type=args.activity_type,
        phenomena=args.phenomena,
        match_mode=args.match_mode,
    )

    if args.num is not None:
        if args.num < 0:
            raise ValueError("--num must be non-negative.")
        rng = random.Random(args.seed)
        if args.num < len(matches):
            matches = rng.sample(matches, args.num)

    result = {
        "filters": {
            "split": args.split,
            "activity_type": args.activity_type,
            "phenomena": args.phenomena or [],
            "match_mode": args.match_mode,
            "num": args.num,
            "seed": args.seed,
        },
        "stats": build_stats(matches),
        "cases": [asdict(r) for r in matches],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Matched cases: {len(matches)}")
    print(f"Saved selection to: {args.output}")
    if args.show_stats:
        print(json.dumps(result["stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
