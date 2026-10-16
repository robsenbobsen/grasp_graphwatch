# Example walkthrough: detection API with an existing database

This walkthrough starts the detection REST API with Docker Compose against an existing
Postgres server that already holds the data (here the DARPA `cadets_e3` database). It then
trains a model and runs inference through the API. The bundled `doc-comp_postgres` is not used.
The full reference is [README_setup_docker_compose.md](README_setup_docker_compose.md).

All commands run from the repository root.

## 1. One-time prerequisite: GPU access for Docker

```bash
nvidia-smi                                            # host sees the GPUs
sudo ./scripts/install_nvidia_container_toolkit.sh    # skip if already installed
```

## 2. Configure

```bash
cp .env.docker.example .env.docker
```

In `.env.docker`, fill in the existing `db_user`/`db_password`/`db_host`/`db_port` lines with
your Postgres server's values, the same as in your `.env`. Set each key exactly once: with
duplicate keys the last one wins. `db_name` is not needed, because each API request names the
database:

```
db_user=postgres
db_password=<password>
db_host=141.71.31.196
db_port=8899
```

Set an API key. Clients send it as the `X-API-Key` header:

```bash
sed -i "s/^GRASP_API_KEY=.*/GRASP_API_KEY=$(openssl rand -hex 32)/" .env.docker
```

Let the containers write files as your user (otherwise they use UID/GID 1000):

```bash
printf 'HOST_UID=%s\nHOST_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
```

## 3. Build and start

```bash
docker compose build          # first time ~10 min (CUDA wheels)
docker compose up -d          # doc-comp_detection_api (:8000) + doc-comp_workbench (:8501)
docker compose ps             # detection API should be "healthy"
docker compose exec doc-comp_detection_api sh -c 'echo $db_user@$db_host:$db_port'   # your DB server
```

PostgreSQL is not started here on purpose: the bundled `doc-comp_postgres` only runs with
`--profile db`. After changing `.env.docker`, apply it with
`docker compose up -d --force-recreate doc-comp_detection_api`.

API docs: http://localhost:8000/docs

## 4. Train and infer in one job

```bash
KEY=$(grep ^GRASP_API_KEY= .env.docker | cut -d= -f2)
API=localhost:8000

curl -s -X POST $API/jobs/pipeline -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d '{
  "db_name": "cadets_e3",
  "train_start": "2018-04-02 00:00:00", "train_end": "2018-04-05 23:59:59",
  "inference_start": "2018-04-06 00:00:00", "inference_end": "2018-04-13 23:59:59",
  "ground_truth_dir": "ground_truth/cadets_e3"
}' | jq
```

`ground_truth_dir` is optional. Without it you get predictions and anomalies only, with no
evaluation. Copy the `id` from the response:

```bash
JOB=<id>
curl -s $API/jobs/$JOB -H "X-API-Key: $KEY" | jq '{status, error, model_id}'
curl -s "$API/jobs/$JOB/log?tail=20" -H "X-API-Key: $KEY"
```

The status goes `queued` → `running` → `succeeded`. The first run builds all graphs from the
database, which takes a while; later runs reuse them.

## 5. Check that it is running (and where)

**Job status:** all jobs and the one that is currently `running`:

```bash
curl -s $API/jobs -H "X-API-Key: $KEY" | jq '.[] | {id, type, status, started_at, finished_at}'
```

**Live log of the job.** The log is a plain file in your checkout, because the repository is
mounted into the container:

```bash
tail -f data/api/jobs/$JOB.log          # Ctrl+C to stop following
```

It shows the current step (`$ ... train.py ...`, then `$ ... infer.py ...`) and progress
lines such as `Progress 12/48` while graphs are built.

**Processes inside the container:** the API runs each step as its own Python process:

```bash
docker top doc-comp_detection_api        # uvicorn + the running train.py / infer.py
```

**GPU usage:** training/inference shows up as a python process with GPU memory:

```bash
nvidia-smi                               # or: watch -n 2 nvidia-smi
```

**Container logs.** These are the API's own requests and errors, not the job output:

```bash
docker compose logs -f doc-comp_detection_api
```

**Output appearing on disk:**

```bash
ls data/graphs/cadets_e3/processed | wc -l     # cached graphs, grows while graphs are built
ls data/models/ data/inference/                # model bundle, then inference results
```

If a job is `failed`, `error` in the job status says which step failed, and the end of the log
shows why (`curl -s "$API/jobs/$JOB/log?tail=50" -H "X-API-Key: $KEY"`).

## 6. Results

```bash
curl -s $API/jobs/$JOB/report -H "X-API-Key: $KEY" | jq '.detection_summary, .evaluation_hits.by_group'
curl -s "$API/jobs/$JOB/anomalies?limit=10" -H "X-API-Key: $KEY" | jq
```

Files on disk:

| Path | Content |
| --- | --- |
| `data/models/<model_id>/` | trained model (location encoder, command mapping, classifier, settings) |
| `data/inference/<run_id>/` | `report.json`, `anomalies.json`, `detailed_report.json`, ... |
| `data/graphs/cadets_e3/` | cached graphs per time window |
| `data/api/jobs/` | job state and logs |

## 7. Inference only, with the trained model

This uses the latest model trained on `db_name`, so wait until the training job has succeeded:

```bash
curl -s -X POST $API/jobs/inference -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d '{
  "db_name": "cadets_e3",
  "inference_start": "2018-04-12 00:00:00", "inference_end": "2018-04-12 23:59:59"
}' | jq
```

Add `"model_id": "<id>"` to use a specific model. To list the trained models:

```bash
curl -s "$API/models?db_name=cadets_e3" -H "X-API-Key: $KEY" | jq '.[] | {model_id, status, train_start, train_end}'
```

## 8. Explore anomalies in the workbench

Open http://localhost:8501, page **Create report**, source **Inference runs**. The dropdown
lists every inference run, with time range, number of anomalies, ground truth, and whether a
report exists already. Pick one and press **Run analysis**. A small *Max anomalies* (e.g. 20)
gives a quick first look. The report is written to `pids_runs/<run_id>_...`. Then open it on
**Overview** (all anomalies) or **Context View** (one anomaly against its closest training
samples) via *Choose report* in the sidebar.

## 9. Live sysdig ingestion

Let sysdig write rotating capture files into a folder inside the repository, e.g.
`graphwatch_data_pipeline/sysdig/input/live/` (`sudo sysdig -w .../live.scap -C 100` starts a
new file every 100 MB). Then call this periodically, e.g. every few minutes. Each call
appends only the new, complete files to `sysdig_live`:

```bash
curl -s -X POST $API/jobs/sysdig-ingest -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d '{
  "db_name": "sysdig_live",
  "scap_dir": "graphwatch_data_pipeline/sysdig/input/live"
}' | jq
```

To check what a call did, look at the end of its job log (`Ingesting N new file(s)`,
`... nodes already known, ... new nodes`, or `No new complete capture files`).

Once some data is in, train a model on it. sysdig needs an experiment config for the
hyperparameters:

```bash
curl -s -X POST $API/jobs/train -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d '{
  "db_name": "sysdig_live",
  "train_start": "2026-09-09 08:00:00", "train_end": "2026-09-09 12:00:00",
  "experiment_config": "grasp/experiments/sysdig_spade/experiment_sysdig_1.yaml"
}' | jq
```

Sysdig times are UTC. From then on, add `"run_inference": true` to the ingest call: each new
batch is classified right after it is ingested, and `/jobs/<id>/report` and
`/jobs/<id>/anomalies` show the result.

## 10. Other useful calls and stopping

```bash
curl -s $API/jobs -H "X-API-Key: $KEY" | jq '.[] | {id, type, status}'   # all jobs
curl -s -X POST $API/jobs/$JOB/cancel -H "X-API-Key: $KEY" | jq .status  # cancel a job
docker compose down                                                      # stop everything
```

`POST /jobs/train` only trains. It takes the same fields as `/jobs/pipeline` without the
inference range.
