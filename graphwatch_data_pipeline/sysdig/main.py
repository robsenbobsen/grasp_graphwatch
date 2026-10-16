#!/usr/bin/env python3

"""Batch pipeline: convert all scap files of a directory and REPLACE the DB tables.

For continuous ("live") ingestion that only appends new capture files, use ingest.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def run_script(script_name: str, *args: str) -> None:
    subprocess.check_call([sys.executable, str(BASE_DIR / script_name), *args])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sysdig scap -> Postgres batch pipeline")
    parser.add_argument(
        "--scap-dir",
        type=Path,
        default=BASE_DIR / "input" / "sysdig_scaps",
        help="Directory with *.scap* files",
    )
    parser.add_argument("--db-name", help="Target database (default: env db_name or 'sysdig')")
    args = parser.parse_args()

    run_script("get_json_from_scap.py", "--scap-dir", str(args.scap_dir))
    run_script("parse_json_into_edges_and_nodes.py")
    run_script("parse_edges_and_nodes_into_pidsmaker_structure.py")
    run_script("upload_data_into_postgres.py", *(["--db-name", args.db_name] if args.db_name else []))


if __name__ == "__main__":
    main()
