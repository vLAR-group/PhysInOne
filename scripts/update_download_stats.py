#!/usr/bin/env python3
"""Build public download badges from all 17 Hugging Face repository counters.

No authentication or third-party packages. A failed/incomplete poll never
replaces the previous published snapshot. README files are not modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

REPOSITORIES = ["vLAR/PhysInOne"] + [f"PhysInOneP{i:02d}/PhysInOneP{i:02d}" for i in range(1, 17)]


def fetch(repo):
    url = f"https://huggingface.co/api/datasets/{repo}?expand=downloads&expand=downloadsAllTime"
    for attempt in range(3):
        try:
            request = Request(url, headers={"Accept": "application/json", "User-Agent": "PhysInOne-download-stats/1.0"})
            with urlopen(request, timeout=30) as response:
                data = json.load(response)
            row = {"repo": repo, "all_time": data.get("downloadsAllTime"), "last_30_days": data.get("downloads")}
            validate_row(row)
            return row
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def validate_row(row):
    for key in ("all_time", "last_30_days"):
        if type(row.get(key)) is not int or row[key] < 0:
            raise ValueError(f"Invalid {key} for {row.get('repo')}")


def build(rows, updated_at):
    if len(rows) != len(REPOSITORIES) or {r["repo"] for r in rows} != set(REPOSITORIES):
        raise ValueError("Expected exactly one result for each of the 17 repositories")
    for row in rows:
        validate_row(row)
    rows = sorted(rows, key=lambda r: REPOSITORIES.index(r["repo"]))
    total = sum(r["all_time"] for r in rows)
    recent = sum(r["last_30_days"] for r in rows)
    snapshot = {"updated_at": updated_at, "repository_count": 17,
                "counting_note": "Sum of HF repository download events; not unique users or complete-dataset downloads.",
                "all_time": total, "last_30_days": recent, "repositories": rows}
    def badge(label, message, color="FFD21E"):
        return {"schemaVersion": 1, "label": label, "message": message, "color": color}
    return {"stats.json": snapshot, "downloads.json": badge("All repo downloads", f"{total:,}"),
            "updated.json": badge("Stats updated (UTC)", updated_at[:16].replace("T", " "), "grey")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, REPOSITORIES))
    data = build(rows, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in data.items():
        path = args.output_dir / filename
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    print(f"Validated 17/17 repositories: {data['stats.json']['all_time']:,} cumulative downloads")


if __name__ == "__main__":
    main()
