import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd

SYSDIG_DIR = Path(__file__).resolve().parents[1] / "graphwatch_data_pipeline" / "sysdig"
sys.path.insert(0, str(SYSDIG_DIR))

import ingest  # noqa: E402
import parse_edges_and_nodes_into_pidsmaker_structure as tables  # noqa: E402
import parse_json_into_edges_and_nodes as parser_  # noqa: E402


def _touch(path: Path, age_seconds: float, content: bytes = b"x") -> Path:
    path.write_bytes(content)
    mtime = time.time() - age_seconds
    os.utime(path, (mtime, mtime))
    return path


def test_find_new_files_skips_ingested_and_young_files(tmp_path):
    old = _touch(tmp_path / "cap.scap2", 300)
    _touch(tmp_path / "cap.scap10", 200)
    _touch(tmp_path / "cap.scap11", 5)  # still being written
    ledger = {"cap.scap2": {**ingest._fingerprint(old)}}

    new = ingest.find_new_files(tmp_path, ledger, min_file_age=60)
    assert [p.name for p in new] == ["cap.scap10"]


def test_find_new_files_reingests_rewritten_file_in_mtime_order(tmp_path):
    # sysdig -W reuses file names: a known name with new content counts as new
    first = _touch(tmp_path / "cap.scap0", 300)
    ledger = {"cap.scap0": ingest._fingerprint(first)}
    _touch(tmp_path / "cap.scap1", 200)
    _touch(tmp_path / "cap.scap0", 100, b"rewritten")

    new = ingest.find_new_files(tmp_path, ledger, min_file_age=60)
    assert [p.name for p in new] == ["cap.scap1", "cap.scap0"]


def test_state_round_trip_drops_exited_processes(tmp_path):
    state = tmp_path / "state.json"
    processes = {1: {"tid": 1, "exe": "/bin/a", "exited": None}, 2: {"tid": 2, "exited": "true"}}
    fd_table = {1: {3: ("file", "/etc/x")}, 2: {4: ("file", "/etc/y")}}
    parser_.save_state(state, processes, fd_table)

    loaded_processes, loaded_fds = parser_.load_state(state)
    assert list(loaded_processes) == [1]
    assert loaded_fds == {1: {3: ("file", "/etc/x")}}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_tables_continues_existing_index(tmp_path):
    src = tmp_path / "edges_and_nodes"
    src.mkdir()
    _write_csv(src / "nodes.csv", [
        {"id": "proc:1", "type": "process", "name": "/bin/a"},
        {"id": "file:/etc/x", "type": "file", "name": "/etc/x"},
        {"id": "file:/etc/new", "type": "file", "name": "/etc/new"},
    ])  # fmt: skip
    _write_csv(src / "edges.csv", [
        {"src": "proc:1", "dst": "file:/etc/x", "action": "open", "timestamp": "1700000000000000000",
         "evt_num": "1", "source_file": "f"},
        {"src": "proc:1", "dst": "file:/etc/new", "action": "write", "timestamp": "1700000001000000000",
         "evt_num": "2", "source_file": "f"},
    ])  # fmt: skip
    out = tmp_path / "tables"

    summary = tables.build_tables(
        src, out, existing_index={"proc:1": 7, "file:/etc/x": 3}, next_index=10, first_event_id=100
    )

    files = pd.read_csv(out / "file_node_table.csv")
    assert files["node_uuid"].tolist() == ["file:/etc/new"]  # only new nodes are written
    assert files["index_id"].tolist() == [10]
    assert pd.read_csv(out / "subject_node_table.csv").empty
    events = pd.read_csv(out / "event_table" / "event_table_batch_00000.csv")
    assert events["src_index_id"].tolist() == [7, 7]
    assert events["dst_index_id"].tolist() == [3, 10]
    assert events["_id"].tolist() == [100, 101]
    assert summary["new_nodes"] == 1
    assert (summary["min_timestamp_ns"], summary["max_timestamp_ns"]) == (
        1700000000000000000,
        1700000001000000000,
    )


def test_ns_to_time_is_utc():
    assert ingest.ns_to_time(1_700_000_000_000_000_000) == "2023-11-14 22:13:20"
