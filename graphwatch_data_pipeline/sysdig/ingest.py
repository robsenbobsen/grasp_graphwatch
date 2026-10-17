#!/usr/bin/env python3

"""Incremental ("live") sysdig ingestion: append new capture files to a database.

Each call picks up the *.scap* files in --scap-dir that were not ingested before and are
complete (unchanged for --min-file-age seconds, so the file sysdig is still writing is
skipped), converts and parses them, and APPENDS them to the tables of --db-name:

- nodes already in the DB keep their index_id, new nodes continue the numbering;
- process and open-fd state is carried over between calls, so events of a process started
  in an earlier file are still attributed to it;
- optionally (--run-inference), GRASP inference runs on exactly the ingested time range.

State per database lives in output/state/<db_name>/ (ingested files, process/fd state).
--reset drops the tables and the state and ingests the directory from scratch.

    python graphwatch_data_pipeline/sysdig/ingest.py \
        --scap-dir graphwatch_data_pipeline/sysdig/input/live --db-name sysdig_live
"""

import argparse
import fcntl
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
sys.path.insert(0, str(BASE_DIR))

import get_json_from_scap  # noqa: E402
import parse_edges_and_nodes_into_pidsmaker_structure as tables  # noqa: E402
import parse_json_into_edges_and_nodes as parser_  # noqa: E402
import upload_data_into_postgres as upload  # noqa: E402

STATE_ROOT = BASE_DIR / "output" / "state"
RUNS_ROOT = BASE_DIR / "output" / "runs"
TIME_FMT = "%Y-%m-%d %H:%M:%S"


def _natural_key(name: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def find_new_files(scap_dir: Path, ledger: dict, min_file_age: float) -> list[Path]:
    """Complete scap files not ingested yet, oldest first (sysdig rotation order)."""
    now = time.time()
    new = []
    for path in get_json_from_scap.find_scap_files(scap_dir):
        if not path.is_file():
            continue
        fp = _fingerprint(path)
        seen = ledger.get(path.name, {})
        if seen.get("size") == fp["size"] and seen.get("mtime_ns") == fp["mtime_ns"]:
            continue  # already ingested, unchanged
        if now - fp["mtime_ns"] / 1e9 < min_file_age:
            print(f"[*] Skipping {path.name}: still being written (younger than {min_file_age}s)")
            continue
        new.append(path)
    return sorted(new, key=lambda p: (p.stat().st_mtime_ns, _natural_key(p.name)))


def ns_to_time(ns: int) -> str:
    """Sysdig timestamps are UTC; GRASP's sysdig dataset reads times as UTC as well."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime(TIME_FMT)


def ingest(
    scap_dir: Path,
    db_name: str,
    min_file_age: float = 60,
    max_files: int | None = None,
    reset: bool = False,
    keep_work_dir: bool = False,
) -> dict:
    state_dir = STATE_ROOT / db_name
    state_dir.mkdir(parents=True, exist_ok=True)
    ledger_file = state_dir / "ingested_files.json"
    process_state_file = state_dir / "process_state.json"

    lock = (state_dir / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"[!] Another ingestion into '{db_name}' is running") from None

    upload.ensure_database(db_name)
    if reset:
        print(f"[*] Reset: dropping tables of '{db_name}' and its ingestion state")
        upload.drop_tables(db_name)
        ledger_file.unlink(missing_ok=True)
        process_state_file.unlink(missing_ok=True)

    ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else {}
    if ledger and not upload.has_events(db_name):
        # e.g. `docker compose down -v`, DROP DATABASE or DROP TABLE: the state survives (it
        # is a file in the repo) but its data is gone. Nothing can be duplicated, so start over.
        print(
            f"[!] The ingestion state lists {len(ledger)} ingested file(s), but '{db_name}' "
            "has no events (database or volume was deleted?). Discarding the stale state and "
            "ingesting from scratch."
        )
        ledger = {}
        ledger_file.unlink(missing_ok=True)
        process_state_file.unlink(missing_ok=True)
    files = find_new_files(Path(scap_dir), ledger, min_file_age)
    if max_files:
        files = files[:max_files]
    if not files:
        print("[*] No new complete capture files")
        return {"db_name": db_name, "new_files": [], "events": 0}
    print(f"[*] Ingesting {len(files)} new file(s): {', '.join(p.name for p in files)}")

    work_dir = RUNS_ROOT / f"{db_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    json_dir, edges_dir, tables_dir = work_dir / "json", work_dir / "edges_and_nodes", work_dir / "tables"
    # Parse with a copy of the state; it only replaces the real one once the upload succeeded.
    work_state = work_dir / "process_state.json"
    work_dir.mkdir(parents=True)
    if process_state_file.exists():
        shutil.copy2(process_state_file, work_state)

    try:
        get_json_from_scap.convert_scap_files(files, json_dir)
        parser_.parse_events(json_dir, edges_dir, state_file=work_state)
        existing_index, next_index, next_event_id = upload.read_existing_index(db_name)
        summary = tables.build_tables(
            edges_dir, tables_dir, existing_index, next_index, next_event_id
        )
        upload.upload_tables(tables_dir, db_name, append=True)
    except BaseException:
        print(f"[!] Ingestion failed; nothing was committed. Work files kept in {work_dir}")
        raise

    shutil.move(work_state, process_state_file)
    for path in files:
        ledger[path.name] = {**_fingerprint(path), "ingested_at": datetime.now(timezone.utc).isoformat()}
    tmp = ledger_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2))
    tmp.replace(ledger_file)
    if not keep_work_dir:
        shutil.rmtree(work_dir)

    result = {"db_name": db_name, "new_files": [p.name for p in files], **summary}
    if summary.get("min_timestamp_ns") is not None:
        result["start"] = ns_to_time(summary["min_timestamp_ns"])
        # +1 s so the last event (sub-second precision) is inside the range
        result["end"] = ns_to_time(summary["max_timestamp_ns"] + 1_000_000_000)
    print(f"[*] Ingested: {json.dumps(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally ingest sysdig captures")
    parser.add_argument("--scap-dir", type=Path, required=True, help="Directory sysdig writes to")
    parser.add_argument("--db-name", required=True, help="Database to append to (created if missing)")
    parser.add_argument(
        "--min-file-age", type=float, default=60,
        help="Only ingest files unchanged for this many seconds (default 60)",
    )  # fmt: skip
    parser.add_argument("--max-files", type=int, help="At most this many files per call")
    parser.add_argument("--reset", action="store_true", help="Drop tables + state, start over")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep intermediate files")
    parser.add_argument(
        "--run-inference", action="store_true", help="Run GRASP inference on the ingested range"
    )
    parser.add_argument("--model-id", help="Model for inference (default: latest for --db-name)")
    parser.add_argument("--ground-truth-dir", help="Optional ground truth for the inference")
    parser.add_argument("--run-id", help="Inference output name")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.db_name):
        parser.error("--db-name may only contain letters, digits and _")

    result = ingest(
        args.scap_dir, args.db_name, args.min_file_age, args.max_files, args.reset,
        args.keep_work_dir,
    )  # fmt: skip

    if args.run_inference:
        if not result["events"]:
            print("[*] No new events, skipping inference")
            return
        sys.path.insert(0, str(PROJECT_ROOT))
        from grasp import stages

        # Graphs are rebuilt: a window may have been cached before all its events arrived.
        out_dir = stages.infer(
            inference_start=result["start"],
            inference_end=result["end"],
            db_name=args.db_name,
            model_id=args.model_id,
            ground_truth_dir=args.ground_truth_dir,
            run_id=args.run_id,
            force_reload=True,
        )
        print(f"RUN_DIR={out_dir}")


if __name__ == "__main__":
    main()
