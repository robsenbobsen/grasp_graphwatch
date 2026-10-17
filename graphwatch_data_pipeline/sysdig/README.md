# Sysdig Data Pipeline

This directory contains the Sysdig data pipeline for Graphwatch.

## Requirements

- `grasp` installed
- PostgreSQL set up and running
- A configured `.env` file
- Input data placed in `input/sysdig_scaps`

## Setup

1. Follow the PostgreSQL setup instructions in `../db_setup/README.md`.
2. Create and configure the `.env` file with the correct PostgreSQL connection details.
```bash
cp .env_example .env
```
3. Ensure the input directory exists and place the data in `input/sysdig_scaps`.
4. Make all files executable:

```bash
chmod +x ./*.py
```

## Run

### Batch: one directory, replace the tables

After placing the input data in `input/sysdig_scaps`, run:

```bash
source .env
./main.py
```

`--scap-dir` reads another directory and `--db-name` picks the target database, e.g.
`./main.py --scap-dir input/experiment3 --db-name sysdig_experiment3`. The tables of that
database are **replaced**.

### Live: append new capture files

`ingest.py` only processes capture files it has not ingested before, and **appends** them to
the database (created if missing):

```bash
./ingest.py --scap-dir input/live --db-name sysdig_live
```

Call it repeatedly (cron, a loop, or the REST API's `POST /jobs/sysdig-ingest`) while sysdig
writes rotating capture files (e.g. `sysdig -w input/live/live.scap -C 100`).

- **Files still being written are skipped.** A file is only ingested once it is unchanged for
  `--min-file-age` seconds (default 60).
- **Files are taken in write order.** Order is by modification time, so rotation with `-W`
  (reused file names) works; a rewritten file counts as new.
- **IDs continue.** Known nodes keep their `index_id`, new nodes and events continue the
  numbering.
- **Process state carries over.** Process and open-file state carries over between calls, so
  events of a process started in an earlier file are still attributed to it. Events of a
  process whose executable only shows up in a *later* file can't be attributed yet and are
  dropped (in tests ~0.7% of events).
- **State is kept per database.** The ingested files and the process state are stored in
  `output/state/<db_name>/`. `--reset` drops the tables and that state and starts over. If
  the database was deleted (e.g. `docker compose down -v`) but the state still lists files,
  the state is discarded automatically and everything is ingested again.
- **Optional inference.** `--run-inference` (optionally with `--model-id` and
  `--ground-truth-dir`) runs GRASP inference on exactly the ingested time range.

All scripts take their directories as arguments (`--help`), with the old paths as defaults.

## Notes

- Make sure `grasp` is installed before running.
- Verify your `.env` settings match your PostgreSQL configuration.
