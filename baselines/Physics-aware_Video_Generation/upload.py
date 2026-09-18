#!/usr/bin/env python3
"""
Upload the CONTENTS of a verified local submission folder to a subfolder in an
existing Hugging Face repository. The local submission folder itself is not
added to the remote path.

Usage:
    python upload.py LOCAL_SUBMISSION_FOLDER \
        --repo REPO_ID \
        --remote_folder REMOTE_PATH \
        --token [HF_TOKEN]

Examples:
    # Use a token saved by `hf auth login`, or the HF_TOKEN environment variable:
    python upload.py ./submission \
        --repo username/dataset-name \
        --remote_folder submissions/team-name \
        --token

    # Pass a token explicitly (less secure because it may remain in shell history):
    python upload.py ./submission \
        --repo username/dataset-name \
        --remote_folder submissions/team-name \
        --token hf_xxx

Dependency:
    pip install -U "huggingface_hub[hf_xet]"
"""

import argparse
import os
import sys
from pathlib import Path, PurePosixPath


def normalize_remote_folder(raw_path: str) -> str:
    """Validate and normalize a relative Hugging Face repository folder."""
    normalized = raw_path.strip().replace("\\", "/").strip("/")
    if not normalized or normalized == ".":
        raise argparse.ArgumentTypeError(
            "--remote_folder must name a repository subfolder, not the root"
        )

    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            "--remote_folder must be a safe relative path without '..'"
        )

    return path.as_posix()


def readable_size(byte_count: int) -> str:
    """Return a human-readable byte count."""
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def folder_stats(folder: Path) -> tuple[int, int]:
    """Count files and bytes recursively, excluding Git metadata."""
    file_count = 0
    byte_count = 0

    for root, directory_names, file_names in os.walk(folder):
        directory_names[:] = [name for name in directory_names if name != ".git"]
        root_path = Path(root)

        for file_name in file_names:
            file_path = root_path / file_name
            if not file_path.is_file():
                continue
            file_count += 1
            try:
                byte_count += file_path.stat().st_size
            except OSError:
                # The uploader will report an authoritative error if this file
                # disappears or becomes unreadable during the upload.
                pass

    return file_count, byte_count


def repository_folder_url(
    repo_id: str,
    repo_type: str,
    revision: str,
    remote_folder: str,
) -> str:
    """Build the web URL for the uploaded repository folder."""
    type_prefix = {
        "model": "",
        "dataset": "datasets/",
        "space": "spaces/",
    }[repo_type]
    return (
        f"https://huggingface.co/{type_prefix}{repo_id}/tree/"
        f"{revision}/{remote_folder}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload the contents of a local submission folder to a specific "
            "folder in a Hugging Face repository."
        )
    )
    parser.add_argument(
        "local_submission_folder",
        type=Path,
        help="Local submission folder whose contents will be uploaded",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Hugging Face repository ID, such as username/repository",
    )
    parser.add_argument(
        "--remote_folder",
        "--remote-folder",
        dest="remote_folder",
        required=True,
        type=normalize_remote_folder,
        help="Destination subfolder inside the repository",
    )
    parser.add_argument(
        "--token",
        nargs="?",
        const=True,
        default=None,
        metavar="HF_TOKEN",
        help=(
            "Hugging Face write token. If used without a value, use HF_TOKEN "
            "or the token saved by `hf auth login`"
        ),
    )
    parser.add_argument(
        "--repo-type",
        choices=("dataset", "model", "space"),
        default="dataset",
        help="Hugging Face repository type (default: dataset)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch or revision (default: main)",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Upload only matching paths; this option may be repeated",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Ignore matching paths; this option may be repeated",
    )
    parser.add_argument(
        "--high-performance",
        action="store_true",
        help="Use maximum upload bandwidth and CPU through hf_xet",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the upload plan without uploading",
    )
    return parser.parse_args()


def resolve_token(token_argument: str | bool | None) -> str | bool | None:
    """Resolve explicit, environment, or locally saved authentication."""
    if isinstance(token_argument, str):
        return token_argument

    environment_token = os.environ.get("HF_TOKEN")
    if environment_token:
        return environment_token

    # True asks huggingface_hub to require the locally saved login token.
    if token_argument is True:
        return True

    # None lets huggingface_hub use its default cached-token behavior.
    return None


def main() -> int:
    args = parse_args()
    local_folder = args.local_submission_folder.expanduser().resolve()

    if not local_folder.is_dir():
        print(f"Error: local folder does not exist: {local_folder}", file=sys.stderr)
        return 1

    file_count, byte_count = folder_stats(local_folder)
    if file_count == 0:
        print(f"Error: local folder contains no files: {local_folder}", file=sys.stderr)
        return 1

    print("Hugging Face upload plan")
    print(f"  Source       : {local_folder}")
    print(f"  Files        : {file_count} ({readable_size(byte_count)})")
    print(f"  Repository   : {args.repo_type} / {args.repo}")
    print(f"  Revision     : {args.revision}")
    print(f"  Remote folder: {args.remote_folder}/")
    print(
        "  Layout       : upload source contents only; "
        f"'{local_folder.name}/' will not be added"
    )

    if args.allow:
        print(f"  Include      : {args.allow}")
    if args.ignore:
        print(f"  Ignore       : {args.ignore}")

    if args.dry_run:
        print("\nDry run complete. No files were uploaded.")
        return 0

    if args.high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            '\nError: huggingface_hub is not installed. Run:\n'
            '  pip install -U "huggingface_hub[hf_xet]"',
            file=sys.stderr,
        )
        return 2

    token = resolve_token(args.token)
    api = HfApi(token=token)

    print("\nChecking authentication and repository access...")
    try:
        user = api.whoami(token=token)
        username = (
            user.get("name", "authenticated user")
            if isinstance(user, dict)
            else "authenticated user"
        )
        api.repo_info(
            repo_id=args.repo,
            repo_type=args.repo_type,
            revision=args.revision,
            token=token,
        )
        print(f"Authenticated as: {username}")
    except Exception as error:
        print(
            f"Error: authentication or repository lookup failed: {error}",
            file=sys.stderr,
        )
        print(
            "Provide a write-capable token after --token, export HF_TOKEN, "
            "or run `hf auth login`.",
            file=sys.stderr,
        )
        return 1

    commit_message = args.commit_message or (
        f"Upload submission contents to {args.remote_folder}"
    )

    print("Uploading... Hugging Face will display transfer progress below.")
    try:
        commit_info = api.upload_folder(
            folder_path=local_folder,
            path_in_repo=args.remote_folder,
            repo_id=args.repo,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=commit_message,
            allow_patterns=args.allow,
            ignore_patterns=args.ignore,
            token=token,
        )
    except KeyboardInterrupt:
        print("\nUpload interrupted. Run the same command again to resume.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"\nUpload failed: {error}", file=sys.stderr)
        print("Run the same command again to resume completed chunks.", file=sys.stderr)
        return 1

    commit_url = getattr(commit_info, "commit_url", None)
    print("\nUpload complete.")
    print(
        "Destination: "
        + repository_folder_url(
            args.repo,
            args.repo_type,
            args.revision,
            args.remote_folder,
        )
    )
    if commit_url:
        print(f"Commit     : {commit_url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
