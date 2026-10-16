#!/usr/bin/env python3

import argparse
import csv
import os
from collections.abc import Iterable
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

# ------------------ CONFIG ------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PARSED_DIR = BASE_DIR / "output" / "tables"

TABLES = (
    "event_table",
    "file_node_table",
    "subject_node_table",
    "netflow_node_table",
)
NODE_TABLES = TABLES[1:]


def conn_args(db_name: str | None = None) -> dict:
    """psycopg2 connection params from the db_* env vars (db_name overrides env)."""
    return {
        "host": os.environ["db_host"],
        "port": os.environ["db_port"],
        "dbname": db_name or os.getenv("db_name", "sysdig"),
        "user": os.environ["db_user"],
        "password": os.environ["db_password"],
    }

# ------------------ HELPERS ------------------


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def get_event_table_files(tables_dir: Path) -> list[Path]:
    return sorted((tables_dir / "event_table").glob("*.csv"))


def drop_table(cur, table_name: str):
    cur.execute(f"DROP TABLE IF EXISTS {quote_identifier(table_name)};")


def build_create_table_sql(table_name: str, columns: Iterable[str]) -> str:
    col_defs_list = []

    for col in columns:
        if table_name == "event_table" and col == "timestamp_rec":
            col_type = "BIGINT"
        elif (
            (
                table_name in {"file_node_table", "subject_node_table", "netflow_node_table"}
                and col == "index_id"
            )
            or table_name == "event_table"
            and (col == "src_index_id" or col == "dst_index_id")
        ):
            col_type = "INT"
        else:
            col_type = "TEXT"

        col_defs_list.append(f"{quote_identifier(col)} {col_type}")

    col_defs = ", ".join(col_defs_list)
    return f"CREATE TABLE IF NOT EXISTS {quote_identifier(table_name)} ({col_defs});"


def read_csv_header(file_path: Path) -> list:
    with file_path.open(newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def create_table_if_needed(cur, table_name: str, file_path: Path):
    columns = read_csv_header(file_path)

    if not columns:
        raise ValueError(f"CSV file is empty: {file_path}")

    sql = build_create_table_sql(table_name, columns)
    cur.execute(sql)


def copy_csv_into_table(cur, table_name: str, file_path: Path):
    with file_path.open("r") as f:
        cur.copy_expert(
            f"""
            COPY {quote_identifier(table_name)}
            FROM STDIN WITH CSV HEADER
            """,
            f,
        )


def upload_file(cur, table_name: str, file_path: Path):
    print(f"\n➡️ Processing {file_path}")

    create_table_if_needed(cur, table_name, file_path)

    copy_csv_into_table(cur, table_name, file_path)

    print(f"Loaded into table: {table_name}")


def ensure_database(db_name: str) -> None:
    """Create db_name on the server if it does not exist yet."""
    args = {**conn_args(db_name), "dbname": "postgres"}
    conn = psycopg2.connect(**args)
    conn.autocommit = True  # CREATE DATABASE cannot run in a transaction
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            if cur.fetchone() is None:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
                print(f"Created database {db_name}")
    finally:
        conn.close()


def _existing_tables(cur) -> set[str]:
    cur.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )
    return {row[0] for row in cur.fetchall()}


def read_existing_index(db_name: str) -> tuple[dict[str, int], int, int]:
    """Node uuid -> index_id already in the DB, the next free index_id and the next event _id."""
    conn = psycopg2.connect(**conn_args(db_name))
    try:
        with conn.cursor() as cur:
            tables = _existing_tables(cur)
            index: dict[str, int] = {}
            for table in NODE_TABLES:
                if table in tables:
                    cur.execute(f"SELECT node_uuid, index_id FROM {quote_identifier(table)}")
                    index.update({uuid: int(idx) for uuid, idx in cur.fetchall()})
            next_event_id = 0
            if "event_table" in tables:
                cur.execute('SELECT COALESCE(MAX("_id"::BIGINT) + 1, 0) FROM event_table')
                next_event_id = int(cur.fetchone()[0])
    finally:
        conn.close()
    next_index = max(index.values()) + 1 if index else 0
    return index, next_index, next_event_id


def drop_tables(db_name: str) -> None:
    conn = psycopg2.connect(**conn_args(db_name))
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for table in TABLES:
                drop_table(cur, table)
    finally:
        conn.close()


def upload_tables(tables_dir: Path = PARSED_DIR, db_name: str | None = None, append: bool = False):
    """Load the table CSVs into the DB; replaces the tables unless append is set."""
    tables_dir = Path(tables_dir)
    args = conn_args(db_name)
    print("🔌 Connecting to database...")

    conn = psycopg2.connect(**args)  # type: ignore
    print(f"Connected to database: {args['dbname']} at {args['host']}:{args['port']} as {args['user']}")
    conn.autocommit = append is False  # replace: as before; append: one transaction

    try:
        with conn.cursor() as cur:
            # ---- Single table CSVs ----
            for table_name in NODE_TABLES:
                file_path = tables_dir / f"{table_name}.csv"

                if not file_path.exists():
                    print(f"Missing: {file_path}, skipping")
                    continue

                if not append:
                    drop_table(cur, table_name)
                upload_file(cur, table_name, file_path)

            # ---- Event table (multiple CSVs) ----
            print(f"\n Loading event_table from: {tables_dir / 'event_table'}")

            event_files = get_event_table_files(tables_dir)

            if not event_files:
                print("No event files found")
            else:
                if not append:
                    drop_table(cur, "event_table")
                for file_path in event_files:
                    upload_file(cur, "event_table", file_path)

        if append:
            conn.commit()
        print("\n DATA SUCCESSFULLY LOADED")

    except BaseException:
        if append:
            conn.rollback()
        raise
    finally:
        conn.close()
        print("🔌 Connection closed")


# ------------------ MAIN ------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Upload sysdig table CSVs into Postgres")
    parser.add_argument("--tables-dir", type=Path, default=PARSED_DIR, help="Table CSV directory")
    parser.add_argument("--db-name", help="Target database (default: env db_name or 'sysdig')")
    parser.add_argument("--append", action="store_true", help="Append instead of replacing tables")
    args = parser.parse_args(argv)
    if args.db_name:
        ensure_database(args.db_name)
    upload_tables(args.tables_dir, args.db_name, args.append)


if __name__ == "__main__":
    main()
