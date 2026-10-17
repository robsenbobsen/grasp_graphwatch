# Example walkthrough with sysdig data (CinITo experiment 2, coffeebot)

This walkthrough runs the whole pipeline on real sysdig captures:
`graphwatch_data_pipeline/sysdig/backup_data/cinito/experiment2/coffeebot/`.

It covers these variations:

| | Static | Live |
| --- | --- | --- |
| **Ingest** | all capture files at once (A1) or the batch CLI (A2) | file by file, as sysdig would write them (B) |
| **Train + infer** | in one job: `/jobs/pipeline` (C1) | |
| **Pretrained model** | train once, infer later: `/jobs/train` + `/jobs/inference` (C2) | each new file is classified on arrival (B) |
| **Explore** | Streamlit workbench (D) | Streamlit workbench (D) |

Setup (GPU, `.env.docker`, API key, build) is in [example_walkthrough.md](example_walkthrough.md),
steps 1–3. This walkthrough stores the sysdig data in the **bundled PostgreSQL of the
docker-compose stack** (`doc-comp_postgres`, on kenny), not on the remote DARPA server.

### Use the bundled PostgreSQL

`.env.docker` holds **one** database server for all API jobs. While it points at the bundled
Postgres, the DARPA databases on `141.71.31.196` are not reachable through the API. See
"Switching back" below.

1. **Host port.** The bundled Postgres is reachable on the host at port 18888 (default). To use
   another one, set `POSTGRES_PORT=<port>` in `.env` (variables of `docker-compose.yml` itself).

2. **Point the API at the bundled server.** In `.env.docker`, set the DB block. Comment out the
   remote values; each key must appear only once:

   ```
   db_user=grasp
   db_password=grasp
   db_host=doc-comp_postgres
   db_port=5432
   ```

   `db_port` is the port inside the Docker network (5432), not the host port 18888.

3. **Start the stack with the database profile.** Recreate the containers so the API reads the
   new settings:

   ```bash
   docker compose --profile db up -d --force-recreate
   docker compose --profile db ps                      # doc-comp_postgres healthy, API healthy
   docker compose exec doc-comp_detection_api sh -c 'echo $db_user@$db_host:$db_port'   # grasp@doc-comp_postgres:5432
   ```

**Which databases exist** is configured in `.env.docker`:

```
POSTGRES_DATABASES=sysdig_coffeebot_static,sysdig_coffeebot_live
```

On every `docker compose --profile db up`, the one-off service `doc-comp_postgres_init` creates
the listed databases that don't exist yet, then exits (`docker logs doc-comp_postgres_init`
shows `created:` / `exists:`). Existing databases are never changed, and the API only starts
once this has succeeded. The databases are empty until the first ingestion creates their
tables; until then the `event_table` query below reports `relation "event_table" does not
exist`. Ingestion would also create a missing database by itself; the list just makes them
exist from the start, e.g. for `psql` or for restoring dumps.

The data is kept in the Docker volume `doc-comp_postgres_data` and survives restarts. To look
inside:

```bash
docker compose exec doc-comp_postgres psql -U grasp -l                                     # databases
docker compose exec doc-comp_postgres psql -U grasp -d sysdig_coffeebot_static \
  -c "SELECT count(*), to_timestamp(min(timestamp_rec)/1e9), to_timestamp(max(timestamp_rec)/1e9) FROM event_table;"
```

Use `--profile db` with every `docker compose up`/`down` from now on, so the database is
included.

**Switching back to the DARPA server:** restore the remote `db_*` lines in `.env.docker`, then
`docker compose up -d --force-recreate doc-comp_detection_api`. The sysdig data stays in the
volume for next time.

### Shell variables used below

Everything below assumes the stack is running and:

```bash
cd /home/buchta/grasp_graphwatch/grasp_graphwatch
KEY=$(grep ^GRASP_API_KEY= .env.docker | cut -d= -f2)
API=localhost:8000
H=(-H "X-API-Key: $KEY" -H "Content-Type: application/json")
```

The `curl ... "${H[@]}"` calls below use this header list.

## 0. The data

```
graphwatch_data_pipeline/sysdig/backup_data/cinito/experiment2/coffeebot/
  sysdig.scap00 … sysdig.scap09     10 files, 9.2 GB, written in this order
```

| File | Last event (UTC) |
| --- | --- |
| `scap00` | 09-08 19:11 |
| `scap01` | 09-08 21:04 |
| `scap02` | 09-08 22:57 |
| `scap03` | 09-09 00:50 |
| `scap04` | 09-09 02:44 |
| `scap05` | 09-09 04:37 |
| `scap06` | 09-09 06:24 |
| `scap07` | 09-09 07:49 |
| `scap08` | 09-09 09:35 |
| `scap09` | 09-09 10:37 |

- **Time range:** 2026-09-08 17:17:57 → 2026-09-09 ~10:39, **UTC**. `ls -l` shows local time
  (CEST = UTC+2). GRASP reads sysdig times as UTC, so all times below are UTC.
- **Settings:** `grasp/experiments/sysdig_cinito/experiment_sysdig_cinito_coffeebot_2.yaml`
  (30-minute windows; `_1.yaml` uses 15 minutes).
- **No ground truth:** there is none for this data, so all inference runs below are without
  evaluation. They report anomalies only.
- **Size:** about 10 million events per 1 GB file. Ingestion converts each file to about 1–2 GB
  of temporary JSON, which is deleted after the job. Expect a few minutes per file; this is an
  estimate from a small test, not measured on these files.

Use a separate database per variant. The ingestion state is kept per `db_name`, and the
database is created on first use in the bundled Postgres.

---

## A. Static ingestion: all files at once

### A1. Through the API (appends; recommended)

```bash
curl -s -X POST $API/jobs/sysdig-ingest "${H[@]}" -d '{
  "db_name": "sysdig_coffeebot_static",
  "scap_dir": "graphwatch_data_pipeline/sysdig/backup_data/cinito/experiment2/coffeebot"
}' | jq
```

This ingests all 10 files in one job. To keep each job (and its temporary disk use) smaller,
add `"max_files": 2` and send the same call 5 times. They queue up, and each one takes the
next 2 files. Check progress as in [example_walkthrough.md](example_walkthrough.md) step 5:

```bash
JOB=<id>
tail -f data/api/jobs/$JOB.log                       # "Ingesting 10 new file(s) ..."
curl -s $API/jobs/$JOB "${H[@]}" | jq '{status, error}'
cat graphwatch_data_pipeline/sysdig/output/state/sysdig_coffeebot_static/ingested_files.json
```

### A2. With the batch CLI (replaces the tables)

```bash
docker compose --profile pipeline run --rm doc-comp_sysdig_pipeline \
  python graphwatch_data_pipeline/sysdig/main.py \
  --scap-dir graphwatch_data_pipeline/sysdig/backup_data/cinito/experiment2/coffeebot \
  --db-name sysdig_coffeebot_batch
```

This is the old flow. It uses the shared `output/` folders, replaces all tables of the database,
and writes no ingestion state. Don't mix it with `ingest.py` on the same `db_name`.

---

## B. Live ingestion: one file at a time

This simulates sysdig writing the files one after another. Instead of copying 9 GB, link the
files one by one into a "live" folder. Relative links work on the host and in the container.

```bash
LIVE=graphwatch_data_pipeline/sysdig/input/coffeebot_live
mkdir -p $LIVE
add() { ln -s ../../backup_data/cinito/experiment2/coffeebot/sysdig.scap0$1 $LIVE/; }
```

**B1. Fill the history and ingest it (files 00–06, until 06:24 UTC):**

```bash
for i in 0 1 2 3 4 5 6; do add $i; done
curl -s -X POST $API/jobs/sysdig-ingest "${H[@]}" -d '{
  "db_name": "sysdig_coffeebot_live", "scap_dir": "'$LIVE'"
}' | jq
```

**B2. Train a model on that history** (pretrained model for the live phase):

```bash
curl -s -X POST $API/jobs/train "${H[@]}" -d '{
  "db_name": "sysdig_coffeebot_live",
  "train_start": "2026-09-08 17:00:00", "train_end": "2026-09-09 06:29:59",
  "experiment_config": "grasp/experiments/sysdig_cinito/experiment_sysdig_cinito_coffeebot_2.yaml"
}' | jq
```

Wait until this job has `succeeded`. The inference calls below use the latest model of
`sysdig_coffeebot_live`, chosen when you send them.

**B3. Live: each new file is ingested and classified at once.** Repeat for 7, 8 and 9. In real
operation, a cron job would call the endpoint every few minutes.

```bash
add 7
curl -s -X POST $API/jobs/sysdig-ingest "${H[@]}" -d '{
  "db_name": "sysdig_coffeebot_live", "scap_dir": "'$LIVE'", "run_inference": true
}' | jq
```

- The job log shows the newly ingested file and its time range (e.g. `"start": "2026-09-09 06:24:…"`),
  then the inference on exactly that range.
- `/jobs/<id>/report` and `/jobs/<id>/anomalies` show the result.
- A call without a new file ends with `No new complete capture files` and skips inference.

Real live operation: let sysdig write into the folder directly, e.g.
`sudo sysdig -w $LIVE/live.scap -C 1000` (a new file every 1000 MB). The file currently being
written is skipped automatically until it hasn't changed for 60 seconds.

**Starting over:** add `"reset": true` to one ingest call. It drops the tables and the ingestion
state of that `db_name`.

---

## C. Training and inference on the static database

### C1. Whole pipeline in one job

```bash
curl -s -X POST $API/jobs/pipeline "${H[@]}" -d '{
  "db_name": "sysdig_coffeebot_static",
  "train_start": "2026-09-08 17:00:00", "train_end": "2026-09-09 08:59:59",
  "inference_start": "2026-09-09 09:00:00", "inference_end": "2026-09-09 10:59:59",
  "experiment_config": "grasp/experiments/sysdig_cinito/experiment_sysdig_cinito_coffeebot_2.yaml"
}' | jq
```

### C2. Pretrained: train once, infer whenever

```bash
# train only
curl -s -X POST $API/jobs/train "${H[@]}" -d '{
  "db_name": "sysdig_coffeebot_static",
  "train_start": "2026-09-08 17:00:00", "train_end": "2026-09-09 08:59:59",
  "experiment_config": "grasp/experiments/sysdig_cinito/experiment_sysdig_cinito_coffeebot_2.yaml"
}' | jq

# later, any range, as often as you like (latest model of the DB)
curl -s -X POST $API/jobs/inference "${H[@]}" -d '{
  "db_name": "sysdig_coffeebot_static",
  "inference_start": "2026-09-09 09:00:00", "inference_end": "2026-09-09 10:59:59"
}' | jq
```

**One model across databases.** A model trained on one sysdig DB can classify another, e.g. the
static model on the live DB. Pass its `model_id` (from `GET /models`):

```bash
curl -s "$API/models?db_name=sysdig_coffeebot_static" "${H[@]}" | jq '.[] | {model_id, status, train_start, train_end}'
curl -s -X POST $API/jobs/inference "${H[@]}" -d '{
  "db_name": "sysdig_coffeebot_live", "model_id": "<model_id>",
  "inference_start": "2026-09-09 07:00:00", "inference_end": "2026-09-09 07:59:59"
}' | jq
```

This also works for live ingestion: `"run_inference": true, "model_id": "<model_id>"` in the
sysdig-ingest call.

### Results

```bash
curl -s $API/jobs/$JOB/report "${H[@]}" | jq '.detection_summary, .time_window_breakdown[:5]'
curl -s "$API/jobs/$JOB/anomalies?limit=20" "${H[@]}" | jq '.total, .items[] | {observed_cmd, predicted_cmd, window}'
ls data/models/ data/inference/
```

---

## D. Explore in the Streamlit workbench

Open http://localhost:8501.

1. **Create report**, source **Inference runs**. The dropdown lists every inference run from
   B3, C1 and C2: run id, `sysdig`, time range, anomalies, `report=yes/no`.
2. Pick a run and set **Max anomalies** to e.g. `20` for a first look. Without a limit, it takes
   about 2 seconds per anomaly. Press **Run analysis** and keep the tab open until "Wrote report
   to …" appears.
3. **Overview**, then *Choose report* in the sidebar: all analyzed anomalies, which label their
   neighborhood resembles more (prediction vs. observed command), ablation, confidences.
4. **Context View**: one anomaly at a time. The anomalous process (TARGET) is shown next to its
   closest training samples of the predicted and the observed command, with graphs and event
   diffs. Here you decide whether it is a false positive.

The report is stored in `pids_runs/<run_id>_pygdist_pool5_max20/`. For the live variant, each
ingested file gives its own inference run, so you can create a report per file.

---

## Cleanup

```bash
docker compose --profile db down                         # stop (data stays in the volume)
rm -r graphwatch_data_pipeline/sysdig/input/coffeebot_live                 # links only
rm -r graphwatch_data_pipeline/sysdig/output/state/sysdig_coffeebot_live   # live ingestion state
```

Drop a single database in the bundled Postgres:

```bash
docker compose exec doc-comp_postgres psql -U grasp -d postgres -c "DROP DATABASE sysdig_coffeebot_live;"
```

To delete **all** data of the bundled Postgres, run `docker compose --profile db down -v`, which
removes the volume `doc-comp_postgres_data`.

The ingestion state (`graphwatch_data_pipeline/sysdig/output/state/<db_name>/`) is a file in the
repository, so it survives both. You don't need to delete it by hand: when the state lists
ingested files but the database has no events, the next ingest call prints `[!] ... Discarding
the stale state` and ingests everything again. To start over while the data is still there,
send one ingest call with `"reset": true`.
