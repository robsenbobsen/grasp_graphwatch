#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# ------------------------------
# CONFIG
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent

SCAP_DIR = BASE_DIR / "input" / "sysdig_scaps"


# Output directory
OUTPUT_DIR = BASE_DIR / "output" / "json"


def clear_directory(directory):
    """Delete all existing entries in directory, leaving it empty."""
    for entry in directory.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def scap_to_json(scap_file, output_file):
    """
    Converts a single sysdig .scap file to JSON lines using sysdig CLI
    """

    print(f"[*] Processing {scap_file} -> {output_file}")

    cmd = ["sysdig", "-r", str(scap_file), "-j"]

    with open(output_file, "w") as outfile:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Drain stderr on a separate thread so a chatty sysdig process can't
        # fill the stderr pipe buffer and deadlock while we're only reading stdout.
        stderr_lines: list[str] = []

        def drain_stderr():
            for line in proc.stderr:  # type: ignore
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()

        for line in proc.stdout:  # type: ignore
            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)
                outfile.write(json.dumps(event) + "\n")
            except json.JSONDecodeError:
                continue

        proc.wait()
        stderr_thread.join()

        if proc.returncode != 0:
            raise RuntimeError(f"sysdig failed for {scap_file}:\n{''.join(stderr_lines)}")


def find_scap_files(scap_dir):
    return sorted(Path(scap_dir).glob("*.scap*"))


def default_workers():
    """Each conversion uses ~2 cores (sysdig + this process): use half the cores, max 16."""
    return max(1, min(16, (os.cpu_count() or 2) // 2))


def convert_scap_files(scap_files, output_dir, workers=None):
    """Convert the given scap files into <output_dir>/<name>.json (cleared first).

    Files are independent, so they are converted in parallel processes (workers, default
    default_workers()); each output file is the same as with a sequential conversion.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Clearing {output_dir} before conversion...")
    clear_directory(output_dir)

    jobs = [(scap_file, output_dir / f"{Path(scap_file).name}.json") for scap_file in scap_files]
    workers = min(workers or default_workers(), len(jobs))
    if workers <= 1:
        for scap_file, output_file in jobs:
            scap_to_json(scap_file, output_file)
    else:
        print(f"[*] Converting {len(jobs)} files with {workers} parallel workers")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(scap_to_json, *job) for job in jobs]:
                future.result()  # re-raises a failed conversion

    print(f"[*] Done. Output files written to {output_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert sysdig .scap files to JSON lines")
    parser.add_argument("--scap-dir", type=Path, default=SCAP_DIR, help="Directory with *.scap* files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="JSON output directory")
    parser.add_argument(
        "--workers", type=int, help=f"Parallel conversions (default {default_workers()})"
    )
    args = parser.parse_args(argv)

    print("[*] Searching for scap files...")
    scap_files = find_scap_files(args.scap_dir)
    if not scap_files:
        raise FileNotFoundError(f"No scap* files found in {args.scap_dir}")
    print(f"[*] Found {len(scap_files)} files")

    convert_scap_files(scap_files, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
