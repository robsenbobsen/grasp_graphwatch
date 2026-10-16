# Sysdig live ingestion: what was built and how to check it

Status: implemented on branch `rest-api` (not committed yet), tested on 2026-10-16.

## Goal

Make the sysdig pipeline usable from the REST API, with the capture directory as an argument,
so it works "live". Sysdig keeps writing rotating capture files. Each call ingests only the new
files into Postgres and can directly run inference on the newly ingested time range.

## What changed

| File | Change |
| --- | --- |
| `graphwatch_data_pipeline/sysdig/ingest.py` | **new**: incremental ingestion (details below) |
| `graphwatch_data_pipeline/sysdig/main.py` | batch pipeline gets `--scap-dir` and `--db-name` (still replaces the tables) |
| `get_json_from_scap.py`, `parse_json_into_edges_and_nodes.py`, `parse_edges_and_nodes_into_pidsmaker_structure.py`, `upload_data_into_postgres.py` | directories and DB as arguments (`--help`), old paths as defaults; process/fd state load/save; index continuation; append mode; DB auto-create |
| `detection_api/app.py` | **new endpoint** `POST /jobs/sysdig-ingest` |
| `Dockerfile` | sysdig 0.41.4 (official release binary) instead of Debian's 0.29, which could not read the captures ("wrong fd type 21") |
| `tests/test_sysdig_ingest.py`, `tests/test_detection_api.py` | tests for file selection, state, index continuation, endpoint validation |
| docs | `graphwatch_data_pipeline/sysdig/README.md`, `README_setup_docker_compose.md`, `example_walkthrough.md` (section 8) |

## How ingest.py works

1. **Picks the files.** It looks for `*.scap*` files in `--scap-dir` that were not ingested
   before. A file counts as new if its name, size or modification time differs, so rotation
   with `-W` (which reuses file names) works. It skips files changed in the last
   `--min-file-age` seconds (default 60), because sysdig is still writing them. Files are
   processed oldest first.
2. **Converts and parses them.** This is the same code as the batch pipeline. Process state
   (which thread ID runs which executable) and open-fd state are loaded from the previous call.
3. **Reads the existing node IDs from the DB.** Known nodes keep their `index_id`. New nodes
   continue the numbering, and so does the event `_id`.
4. **Appends in one transaction.** It inserts only the new nodes and all new events. If
   anything fails, nothing is committed and the state does not advance.
5. **Saves the new state.** That is `output/state/<db_name>/ingested_files.json` and
   `process_state.json`.
6. **Optionally runs inference.** With `--run-inference` it runs GRASP inference on
   `[first event, last event + 1 s]` of the batch (UTC), with graphs rebuilt
   (`force_reload`).

`--reset` drops the tables and the state and starts over. A lock file prevents two ingestions
into the same DB at the same time.

## Test results (local test DB, 3 × 200k events cut from `sysdig copy.scap000-002`)

- **Incremental appends:** file 1 added 3,370 new nodes and 9,139 events. File 2 reused
  2,580 known nodes and added 1,317 new ones. A repeat call found nothing to do. A file
  younger than `--min-file-age` was skipped.
- **Compared with ingesting both files at once:** identical node set (4,687), all `node_uuid`
  and `index_id` unique, event `_id` unique, no edges pointing to missing nodes.
- **Expected difference:** live ingestion has 136 of 18,655 events (0.7%) fewer. These
  belong to a process whose executable only appears in the *later* file. A one-pass batch
  can attribute its earlier events; live ingestion can't know it yet when the first file
  arrives. The process node exists, just without those early events.
- **API:** `POST /jobs/sysdig-ingest` with `run_inference: true` ingested file 3 and
  classified 09:05:45–09:06:02 in one job, reporting 15 anomalies. A non-sysdig `db_name` is
  rejected (422).
- **Automated tests:** 26 passing.

## How to check it yourself

```bash
# 1. a folder sysdig writes to (inside the repo, so the container sees it)
mkdir -p graphwatch_data_pipeline/sysdig/input/live
sudo sysdig -w graphwatch_data_pipeline/sysdig/input/live/live.scap -C 100   # rotates every 100 MB

# 2. ingest (repeat this, e.g. every few minutes)
curl -s -X POST localhost:8000/jobs/sysdig-ingest -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"db_name": "sysdig_live", "scap_dir": "graphwatch_data_pipeline/sysdig/input/live"}' | jq

# 3. what happened: end of the job log
curl -s "localhost:8000/jobs/<id>/log?tail=30" -H "X-API-Key: $KEY"
cat graphwatch_data_pipeline/sysdig/output/state/sysdig_live/ingested_files.json

# 4. train once on some ingested data (sysdig needs an experiment config), then add
#    "run_inference": true to the ingest call
```

Without the API: `python graphwatch_data_pipeline/sysdig/ingest.py --scap-dir ... --db-name ...`.

## Limitations / open points

- **Events before the executable is known are dropped** (see the test results).
- **Each call's inference covers only that batch's time range.** A model window
  (`context_size`) that spans two batches is classified per batch, not as a whole.
- **Old states are not migrated.** A DB filled by the batch `main.py` has no ingestion state.
  The first `ingest.py` call treats all files in the folder as new, so use `--reset` or a
  fresh `db_name`.
- **The default CLI paths still point to the old shared folder.** Running the batch
  `main.py` (default paths) still clears `output/json`, `output/edges_and_nodes` and
  `output/tables` as before. `ingest.py` uses its own work folder per call
  (`output/runs/...`), which is deleted after success.
- **sysdig 0.41 can't write filtered captures.** `sysdig -r ... -w ... <filter>` exits with an
  error and a truncated file. Cutting with `-n <count>` works.
