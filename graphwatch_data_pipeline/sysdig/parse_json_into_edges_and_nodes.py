#!/usr/bin/env python3

import argparse
import csv
import html
import json
import re
import shutil
import sqlite3
import tempfile
from functools import cache
from pathlib import Path

# ------------------------------
# CONFIG
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "output" / "json"
OUTPUT_DIR = BASE_DIR / "output" / "edges_and_nodes"

NODES_FILENAME = "nodes.csv"
EDGES_FILENAME = "edges.csv"
PROGRESS_INTERVAL = 1_000_000
JSON_CHUNK_SIZE = 1024 * 1024

# ------------------------------
# Regex helpers
# ------------------------------

FD_RE = re.compile(r"fd=(?P<fd>-?\d+)\((?P<tag><[^>]+>)(?P<value>[^)]*)\)")
NAME_RE = re.compile(r"\bname=(?P<name>(\"[^\"]+\"|'[^']+'|\S+))")
TUPLE_RE = re.compile(r"\btuple=(?P<tuple>\S+)")
EXE_RE = re.compile(r"\bexe=(?P<exe>\S+)")
COMM_RE = re.compile(r"\bcomm=(?P<comm>\S+)")
SIZE_RE = re.compile(r"\bsize=(?P<size>\d+)")
RES_RE = re.compile(r"\bres=(?P<res>-?\d+)")
FD_NUM_RE = re.compile(r"\bfd=(?P<fd>-?\d+)")
PROCESS_NUMERIC_FIELD_TEMPLATE = r"\b{key}=(?P<value>-?\d+)(?:\([^)]*\))?"

# Negative res codes that don't indicate a real failure (e.g. non-blocking
# connect() reporting EINPROGRESS, or a non-blocking read/write reporting
# EAGAIN/EWOULDBLOCK).
NON_FATAL_RES_CODES = {-11, -115}

def clear_directory(directory):
    """Delete all existing entries in directory, leaving it empty."""
    for entry in directory.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


# ------------------------------
# Node helpers
# ------------------------------


def proc_node(tid):
    return f"proc:{tid}"


def file_node(path):
    return f"file:{path}"


def socket_node(addr):
    return f"socket:{addr}"


# ------------------------------
# General helpers
# ------------------------------


def normalize_event(event):
    """Unescape (twice) HTML entities in string fields, in place.

    Only strings containing "&" can hold an entity; for all others html.unescape is a no-op,
    so they are skipped (most fields of most events).
    """
    for key, value in event.items():
        if isinstance(value, str) and "&" in value:
            event[key] = html.unescape(html.unescape(value))
    return event


def clean_name(value):
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
        value = value[1:-1]
    return value


@cache
def numeric_field_pattern(key):
    return re.compile(PROCESS_NUMERIC_FIELD_TEMPLATE.format(key=re.escape(key)))


def extract_numeric_field(info, key):
    match = numeric_field_pattern(key).search(info)
    return int(match.group("value")) if match else None


def extract_exe(info):
    match = EXE_RE.search(info)
    return match.group("exe") if match else None


def extract_comm(info):
    match = COMM_RE.search(info)
    return match.group("comm") if match else None


def extract_size(info):
    match = SIZE_RE.search(info)
    return int(match.group("size")) if match else None


def event_succeeded(info):
    """Best-effort check that a syscall event actually succeeded.

    Sysdig reports failures as a negative res=-<errno> (e.g. connect(),
    read()/write()) or, for open()/openat(), as a negative fd. Some negative
    res codes (EAGAIN/EWOULDBLOCK, EINPROGRESS) are expected on non-blocking
    fds and are not treated as failures.
    """
    res_match = RES_RE.search(info)
    if res_match:
        res = int(res_match.group("res"))
        if res < 0 and res not in NON_FATAL_RES_CODES:
            return False

    fd_match = FD_NUM_RE.search(info)
    if fd_match and int(fd_match.group("fd")) < 0:
        return False

    return True


def extract_process_identity(event):
    info = event.get("evt.info", "")
    tid = event.get("thread.tid")
    if tid is None:
        tid = extract_numeric_field(info, "tid")
    if tid is not None:
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            tid = None
    # Most events (read/write/...) carry none of these fields; the substring checks skip
    # the regexes, which could not match without them.
    return {
        "tid": tid,
        "pid": extract_numeric_field(info, "pid") if "pid=" in info else None,
        "ptid": extract_numeric_field(info, "ptid") if "ptid=" in info else None,
        "exe": extract_exe(info) if "exe=" in info else None,
        "comm": extract_comm(info) if "comm=" in info else None,
        "proc_name": event.get("proc.name"),
    }


# ------------------------------
# Resource parsing
# ------------------------------


def extract_fd_resource(info):
    match = FD_RE.search(info)
    if not match:
        return None, None, None
    fd = int(match.group("fd"))
    tag = match.group("tag")
    value = match.group("value").strip()
    fd_tag = tag.strip("<>")
    if fd_tag in ("f", "d"):
        return fd, "file", value
    if "->" in value:
        return fd, "socket", value
    return fd, "fd", value


def extract_name_resource(info):
    match = NAME_RE.search(info)
    if not match:
        return None, None
    name = clean_name(match.group("name"))
    return ("file", name) if name else (None, None)


def extract_tuple_resource(info):
    match = TUPLE_RE.search(info)
    if not match:
        return None, None
    return "socket", match.group("tuple")


def extract_resource(info):
    fd, resource_type, resource_value = extract_fd_resource(info)
    if fd is not None:
        return fd, resource_type, resource_value
    resource_type, resource_value = extract_name_resource(info)
    if resource_type:
        return None, resource_type, resource_value
    resource_type, resource_value = extract_tuple_resource(info)
    if resource_type:
        return None, resource_type, resource_value
    return None, None, None


# ------------------------------
# Streaming event input
# ------------------------------


def discover_input_files(input_dir):
    files = {
        path for pattern in ("*.json", "*.jsonl", "*.ndjson") for path in input_dir.glob(pattern)
    }
    return sorted(files)


def iter_json_values(file_path, chunk_size=JSON_CHUNK_SIZE):
    """Stream a JSON array, one JSON object, or adjacent JSON/JSONL values."""
    decoder = json.JSONDecoder()
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        buffer = ""
        pos = 0
        eof = False
        array_mode = None
        array_finished = False

        while True:
            if not eof and (len(buffer) - pos < chunk_size // 2):
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer = buffer[pos:] + chunk
                    pos = 0
                else:
                    eof = True

            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1

            if array_mode is None:
                if pos >= len(buffer):
                    if eof:
                        return
                    continue
                array_mode = buffer[pos] == "["
                if array_mode:
                    pos += 1

            while pos < len(buffer) and (
                buffer[pos].isspace() or (array_mode and buffer[pos] == ",")
            ):
                pos += 1

            if array_mode and pos < len(buffer) and buffer[pos] == "]":
                pos += 1
                array_finished = True
                return

            if pos >= len(buffer):
                if eof:
                    if array_mode and not array_finished:
                        raise ValueError(f"Unterminated JSON array in {file_path}")
                    return
                continue

            try:
                value, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(
                        f"Invalid JSON in {file_path} near character {pos}: {exc}"
                    ) from exc
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                    continue
                eof = True
                continue

            pos = end
            yield value


def iter_events_from_file(file_path):
    """Stream valid events in exactly the order stored in the source file."""
    item_number = 0
    try:
        for value in iter_json_values(file_path):
            item_number += 1
            if not isinstance(value, dict) or "evt.outputtime" not in value:
                continue
            event = normalize_event(value)
            event["_source_file"] = file_path.name
            event["_line_no"] = item_number
            yield event
    except OSError as exc:
        raise RuntimeError(f"Could not read {file_path}: {exc}") from exc


def iter_all_events(input_dir):
    """
    Stream files in deterministic filename order and preserve the event order
    found inside each file. No sorting or temporary sorted runs are used.
    """
    for file_path in discover_input_files(input_dir):
        print(f"[*] Reading: {file_path.name}")
        yield from iter_events_from_file(file_path)


# ------------------------------
# Node and process helpers
# ------------------------------


def add_node(nodes, node_id, node_type, name, extra=None):
    if node_id not in nodes:
        nodes[node_id] = {"id": node_id, "type": node_type, "name": name}
    elif name:
        nodes[node_id]["name"] = name
    if extra:
        for key, value in extra.items():
            if value is not None:
                nodes[node_id][key] = value


def ensure_process(processes, tid):
    if tid not in processes:
        processes[tid] = {
            "tid": tid,
            "pid": None,
            "ptid": None,
            "exe": None,
            "comm": None,
            "proc_name": None,
            "first_seen": None,
            "last_seen": None,
            "exited": None,
            "exit_timestamp": None,
        }
    return processes[tid]


def update_process_from_event(processes, event):
    identity = extract_process_identity(event)
    tid = identity["tid"]
    if tid is None:
        return None
    proc = ensure_process(processes, tid)
    timestamp = event.get("evt.outputtime")
    if proc["first_seen"] is None:
        proc["first_seen"] = timestamp
    proc["last_seen"] = timestamp
    for key in ("pid", "ptid", "exe", "comm", "proc_name"):
        value = identity.get(key)
        if value is not None:
            proc[key] = value
    return tid


# ------------------------------
# Single pass: nodes, process metadata and edge candidates
# ------------------------------
#
# Edges may only be written for "resolved" processes (an executable is known), which is
# only certain after the last event. Instead of reading all events a second time, every
# event is processed once: node/process bookkeeping and fd tracking as before, and each edge
# is stored as a candidate together with the process tid(s) it needs. After the last event
# the candidates are filtered by the resolved processes (keeping their order), which gives
# exactly the edges of the former two-pass version.


PROCESS_EVENT_TYPES = frozenset(
    ("clone", "fork", "vfork", "execve", "execveat", "exit", "exit_group")
)


def update_nodes_and_processes(event, tid, nodes, processes, resource, succeeded):
    """Node/process bookkeeping of one event (formerly pass 1).

    resource / succeeded: extract_resource() and event_succeeded() of the event's evt.info,
    computed once per event by the caller.
    """
    evt_type = event.get("evt.type")
    evt_info = event.get("evt.info", "")
    timestamp = event.get("evt.outputtime")

    if evt_type in ("clone", "fork", "vfork"):
        child_tid = extract_numeric_field(evt_info, "res")
        if child_tid is not None and child_tid > 0:
            ensure_process(processes, child_tid)
        return

    if evt_type in ("execve", "execveat"):
        executable = extract_exe(evt_info)
        if executable and succeeded:
            ensure_process(processes, tid)["exe"] = executable
            add_node(nodes, file_node(executable), "file", executable)
        return

    if evt_type in ("exit", "exit_group"):
        process = ensure_process(processes, tid)
        process["exited"] = "true"
        process["exit_timestamp"] = timestamp
        return

    if succeeded:
        _, resource_type, resource_value = resource
        if resource_type == "file" and resource_value:
            add_node(nodes, file_node(resource_value), "file", resource_value)
        elif resource_type == "socket" and resource_value:
            add_node(nodes, socket_node(resource_value), "socket", resource_value)


def add_process_nodes(nodes, processes):
    """Process nodes for every process with a known executable; returns their tids."""
    resolved_process_tids = set()
    for tid, process in processes.items():
        executable = process.get("exe")
        if not executable:
            continue
        add_node(
            nodes,
            proc_node(tid),
            "process",
            executable,
            extra={
                "tid": tid,
                "pid": process.get("pid"),
                "ptid": process.get("ptid"),
                "comm": process.get("comm"),
                "proc_name": process.get("proc_name"),
                "first_seen": process.get("first_seen"),
                "last_seen": process.get("last_seen"),
                "exited": process.get("exited"),
                "exit_timestamp": process.get("exit_timestamp"),
            },
        )
        resolved_process_tids.add(tid)
    return resolved_process_tids


EDGE_FIELDNAMES = [
    "src",
    "dst",
    "action",
    "timestamp",
    "evt_num",
    "syscall",
    "direction",
    "proc_name",
    "thread_tid",
    "resource",
    "size",
    "source_file",
]


def make_edge(src, dst, action, timestamp, event, resource_value=None):
    size = extract_size(event.get("evt.info", ""))
    return {
        "src": src,
        "dst": dst,
        "action": action,
        "timestamp": timestamp,
        "evt_num": event.get("evt.num"),
        "syscall": event.get("evt.type"),
        "direction": event.get("evt.dir"),
        "proc_name": event.get("proc.name"),
        "thread_tid": event.get("thread.tid"),
        "resource": resource_value or "",
        "size": size if size is not None else "",
        "source_file": event.get("_source_file", ""),
    }


def edge_candidates(event, tid, fd_table, resource, succeeded):
    """(edge, required tids) of one event, updating fd_table (formerly pass 2).

    Same rules as before, except that the "process is resolved" check is left to the
    caller: an edge is kept only if all its required tids turn out to be resolved.
    """
    evt_type = event.get("evt.type")
    evt_info = event.get("evt.info", "")
    direction = event.get("evt.dir")
    timestamp = event.get("evt.outputtime")

    fd_table.setdefault(tid, {})
    process_id = proc_node(tid)

    if evt_type in ("clone", "fork", "vfork"):
        child_tid = extract_numeric_field(evt_info, "res")
        if child_tid is not None and child_tid > 0:
            fd_table.setdefault(child_tid, dict(fd_table.get(tid, {})))
            edge = make_edge(
                proc_node(tid), proc_node(child_tid), evt_type, timestamp, event, str(child_tid)
            )
            return [(edge, (tid, child_tid))]
        return []

    if evt_type in ("execve", "execveat"):
        executable = extract_exe(evt_info)
        if executable and succeeded:
            edge = make_edge(process_id, file_node(executable), "exec", timestamp, event, executable)
            return [(edge, (tid,))]
        return []

    if evt_type in ("exit", "exit_group"):
        return []

    fd, resource_type, resource_value = resource
    if fd is not None and resource_type and resource_value:
        fd_table[tid][fd] = (resource_type, resource_value)
    if fd is not None and (not resource_type or not resource_value):
        stored = fd_table[tid].get(fd)
        if stored:
            resource_type, resource_value = stored

    edge_spec = None
    if resource_type == "file" and resource_value:
        resource_id = file_node(resource_value)
        if evt_type == "read" and direction == "<":
            edge_spec = (resource_id, process_id, "read")
        elif evt_type == "write" and direction == "<":
            edge_spec = (process_id, resource_id, "write")
        elif evt_type in ("open", "openat") and direction == "<":
            edge_spec = (process_id, resource_id, "open")
        elif evt_type == "close" and direction == "<":
            edge_spec = (process_id, resource_id, "close")
    elif resource_type == "socket" and resource_value:
        resource_id = socket_node(resource_value)
        if evt_type == "recvfrom" and direction == "<":
            edge_spec = (resource_id, process_id, "recv")
        elif evt_type == "sendto" and direction == "<":
            edge_spec = (process_id, resource_id, "send")
        elif evt_type == "read" and direction == "<":
            edge_spec = (resource_id, process_id, "recv")
        elif evt_type == "write" and direction == "<":
            edge_spec = (process_id, resource_id, "send")
        elif evt_type == "connect" and direction == "<":
            edge_spec = (process_id, resource_id, "connect")
        elif evt_type in ("accept", "accept4") and direction == "<":
            edge_spec = (resource_id, process_id, "accept")
        elif evt_type == "close" and direction == "<":
            edge_spec = (process_id, resource_id, "close")

    candidates = []
    if edge_spec and succeeded:
        edge = make_edge(edge_spec[0], edge_spec[1], edge_spec[2], timestamp, event, resource_value)
        candidates.append((edge, (tid,)))

    if evt_type == "close" and fd is not None:
        fd_table[tid].pop(fd, None)

    return candidates


def open_dedup_database(directory):
    with tempfile.NamedTemporaryFile(
        prefix="sysdig_edge_dedup_",
        suffix=".sqlite3",
        dir=directory,
        delete=False,
    ) as temp:
        path = Path(temp.name)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("CREATE TABLE seen (edge_key TEXT PRIMARY KEY) WITHOUT ROWID")
    return connection, path


def is_new_edge(dedup_connection, edge):
    key = json.dumps(
        [edge["src"], edge["dst"], edge["action"], edge["timestamp"], edge["evt_num"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    cursor = dedup_connection.execute("INSERT OR IGNORE INTO seen(edge_key) VALUES (?)", (key,))
    return cursor.rowcount != 0


def process_events(input_dir, edges_file, processes=None, fd_table=None):
    """Read all events once; write edges_file and return (nodes, resolved tids, events, edges).

    processes / fd_table: state carried over from earlier batches (updated in place).
    Duplicate edges are dropped at their first occurrence; since an edge's key contains
    its processes, a duplicate is kept or filtered exactly like its first occurrence.
    """
    processes = {} if processes is None else processes
    fd_table = {} if fd_table is None else fd_table
    nodes = {}
    event_count = 0
    candidate_count = 0
    edges_file = Path(edges_file)
    candidates_file = edges_file.with_name(edges_file.name + ".candidates")
    dedup_connection, dedup_path = open_dedup_database(edges_file.parent)

    try:
        with candidates_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for event in iter_all_events(input_dir):
                event_count += 1
                tid = update_process_from_event(processes, event)
                if tid is None:
                    continue
                evt_info = event.get("evt.info", "")
                succeeded = event_succeeded(evt_info)
                resource = (
                    None if event.get("evt.type") in PROCESS_EVENT_TYPES else extract_resource(evt_info)
                )
                update_nodes_and_processes(event, tid, nodes, processes, resource, succeeded)
                for edge, required in edge_candidates(event, tid, fd_table, resource, succeeded):
                    if is_new_edge(dedup_connection, edge):
                        required_2 = required[1] if len(required) > 1 else ""
                        writer.writerow(
                            [required[0], required_2, *(edge[f] for f in EDGE_FIELDNAMES)]
                        )
                        candidate_count += 1

                if event_count % PROGRESS_INTERVAL == 0:
                    dedup_connection.commit()
                    print(
                        f"[*] Events: {event_count:,}, nodes: {len(nodes):,}, "
                        f"processes: {len(processes):,}, edge candidates: {candidate_count:,}"
                    )
        dedup_connection.commit()
    finally:
        dedup_connection.close()
        try:
            dedup_path.unlink()
        except OSError:
            pass

    resolved_process_tids = add_process_nodes(nodes, processes)
    resolved = {str(tid) for tid in resolved_process_tids}

    edge_count = 0
    try:
        with (
            candidates_file.open(newline="", encoding="utf-8") as src,
            edges_file.open("w", newline="", encoding="utf-8") as dst,
        ):
            writer = csv.writer(dst)
            writer.writerow(EDGE_FIELDNAMES)
            for row in csv.reader(src):
                if row[0] in resolved and (row[1] == "" or row[1] in resolved):
                    writer.writerow(row[2:])
                    edge_count += 1
    finally:
        candidates_file.unlink(missing_ok=True)

    return nodes, resolved_process_tids, event_count, edge_count


# ------------------------------
# Save nodes
# ------------------------------


def save_nodes(nodes, nodes_file):
    fieldnames = [
        "id",
        "type",
        "name",
        "tid",
        "pid",
        "ptid",
        "comm",
        "proc_name",
        "first_seen",
        "last_seen",
        "exited",
        "exit_timestamp",
    ]
    with Path(nodes_file).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(nodes.values())


# ------------------------------
# State carried between incremental batches
# ------------------------------


def load_state(state_file):
    """Processes and open fds of earlier batches (empty if there is no state yet)."""
    state_file = Path(state_file)
    if not state_file.exists():
        return {}, {}
    state = json.loads(state_file.read_text(encoding="utf-8"))
    processes = {int(tid): proc for tid, proc in state.get("processes", {}).items()}
    fd_table = {
        int(tid): {int(fd): tuple(resource) for fd, resource in fds.items()}
        for tid, fds in state.get("fd_table", {}).items()
    }
    return processes, fd_table


def save_state(state_file, processes, fd_table):
    """Keep only processes that are still alive; their fds stay resolvable next batch."""
    alive = {tid: proc for tid, proc in processes.items() if proc.get("exited") != "true"}
    state = {
        "processes": {str(tid): proc for tid, proc in alive.items()},
        "fd_table": {
            str(tid): {str(fd): list(resource) for fd, resource in fds.items()}
            for tid, fds in fd_table.items()
            if tid in alive and fds
        },
    }
    state_file = Path(state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(state_file)


def parse_events(input_dir, output_dir, state_file=None):
    """Write nodes.csv and edges.csv for all JSON files in input_dir.

    With state_file, process/fd state from earlier batches is used and updated, so an
    incremental batch resolves processes and fds that were started in an earlier one.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    nodes_file, edges_file = output_dir / NODES_FILENAME, output_dir / EDGES_FILENAME
    print(f"[*] Input directory:  {input_dir}")
    print(f"[*] Output directory: {output_dir}")
    files = discover_input_files(input_dir)
    if not files:
        raise SystemExit(f"[!] No .json, .jsonl, or .ndjson files found in {input_dir}")

    print(f"[*] Input files: {len(files)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Clearing {output_dir} before parsing...")
    clear_directory(output_dir)

    processes, fd_table = load_state(state_file) if state_file else ({}, {})

    print("[*] Single pass: nodes, process metadata and edges")
    nodes, resolved_process_tids, event_count, edge_count = process_events(
        input_dir, edges_file, processes, fd_table
    )
    print(f"[*] Events:              {event_count:,}")
    print(f"[*] Resolved processes:  {len(resolved_process_tids):,}")
    print(f"[*] Collected nodes:     {len(nodes):,}")

    save_nodes(nodes, nodes_file)
    print(f"[*] Nodes written: {len(nodes):,} -> {nodes_file}")
    print(f"[*] Edges written: {edge_count:,} -> {edges_file}")
    print("[*] Provenance graph exported")

    if state_file:
        save_state(state_file, processes, fd_table)
    return {"events": event_count, "nodes": len(nodes), "edges": edge_count}


# ------------------------------
# Main
# ------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parse sysdig JSON events into nodes/edges CSVs")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="JSON input directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="CSV output directory")
    parser.add_argument(
        "--state-file", type=Path, help="Process/fd state to continue from (incremental mode)"
    )
    args = parser.parse_args(argv)
    parse_events(args.input_dir, args.output_dir, args.state_file)


if __name__ == "__main__":
    main()
