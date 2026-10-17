# Setup with Docker Compose (GPU)

This guide runs the whole GRASP repository in Docker: PostgreSQL, the detection REST API
(training and inference), the contextualization workbench and the sysdig data pipeline. Everything uses the `doc-comp`
prefix (project `doc-comp`) and runs on NVIDIA GPUs.

| Service | Purpose |
| --- | --- |
| `doc-comp_postgres` | optional PostgreSQL 17 (profile `db`), volume `doc-comp_postgres_data`, host port `18888` |
| `doc-comp_detection_api` | REST API on http://localhost:8000 that queues training/inference jobs |
| `doc-comp_grasp` | runs `main.py` (full pipeline) manually (profile `cli`) |
| `doc-comp_workbench` | contextualization workbench on http://localhost:8501 |
| `doc-comp_console` | GRASP GraphWatch console on http://localhost:8502: interactive front end for the whole API (profile `console`) |
| `doc-comp_sysdig_pipeline` | sysdig `.scap` → Postgres pipeline (profile `pipeline`, on demand) |

All detection services share the image `doc-comp_grasp:latest`, the network `doc-comp_net`, and
mount the repository at `/detection_service`. Results (`data/`, `pids_runs/`) therefore land directly in
your checkout and are owned by your user.

## 1. Prerequisites

- Docker Engine with the Compose plugin (`docker compose version`)
- An NVIDIA GPU and driver supporting CUDA 12.9 (check with `nvidia-smi`)
- The NVIDIA Container Toolkit (next step)

## 2. Install the NVIDIA Container Toolkit

Docker needs this toolkit to give containers access to the GPUs. On Debian/Ubuntu:

```bash
sudo ./scripts/install_nvidia_container_toolkit.sh
```

The script adds NVIDIA's apt repository, installs `nvidia-container-toolkit`, registers
the runtime with Docker (`nvidia-ctk runtime configure --runtime=docker`) and restarts
Docker. For other distributions see the
[official install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Check it works. This should list your GPUs:

```bash
docker run --rm --gpus all doc-comp_grasp:latest nvidia-smi
```

(This needs the image from step 4. Any CUDA image works too, e.g. `ubuntu` with `--gpus all`.)

## 3. Configure

```bash
cp .env.docker.example .env.docker
```

Set `db_user`, `db_password`, `db_host` and `db_port` to the Postgres server holding your data
(e.g. the remote server). The database itself is chosen per request (`db_name`). To use the
bundled, initially empty `doc-comp_postgres` instead, use `db_user=grasp`, `db_password=grasp`,
`db_host=doc-comp_postgres`, `db_port=5432` and add `--profile db` to `docker compose up`.
`POSTGRES_DATABASES` in `.env.docker` lists the databases that should exist in it; missing
ones are created on every `--profile db up` (see step 5).

Set `GRASP_API_KEY` in `.env.docker` (e.g. `openssl rand -hex 32`); the API refuses to start
without it, and clients send it as the `X-API-Key` header.

Variables that compose itself reads (set them in your shell or in `.env`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `POSTGRES_PORT` | `18888` | host port of `doc-comp_postgres` |
| `WORKBENCH_PORT` | `8501` | host port of the workbench |
| `API_PORT` | `8000` | host port of the detection API |
| `GRASP_EXPERIMENT_CONFIG` | `.../cadets_e3_default/experiment_cadets_e3.yaml` | experiment run by `doc-comp_grasp` |
| `DUMPS_DIR` | `./db_setup/dumps` | directory with PIDSMaker `.dump` files, mounted at `/dumps` |
| `HOST_UID` / `HOST_GID` | `1000` | user the detection service containers run as |
| `TORCH_VARIANT` | `cu129` | PyTorch/PyG wheel build; `cpu` for a CPU-only image |

**Port already in use:** set another `POSTGRES_PORT` in `.env`, or use an existing database
instead of the bundled one (see [External database](#external-database)).

## 4. Build

```bash
docker compose build
```

The image is `python:3.12.7-slim` with torch 2.8.0 (CUDA 12.9), PyTorch Geometric,
`requirements.txt`, `grasp_contextualization/requirements.txt` and the `sysdig` CLI.

## 5. Start the database and load data

```bash
docker compose --profile db up -d
docker logs doc-comp_postgres_init          # created: / exists: per database
```

The one-off service `doc-comp_postgres_init` creates every database listed in
`POSTGRES_DATABASES` (`.env.docker`, comma-separated) that does not exist yet, then exits;
existing databases are never changed. The other services start once it has succeeded. A
database added to the list later is created on the next `--profile db up`.

### DARPA datasets (PIDSMaker dumps)

Download the dumps as described in the [db_setup README](db_setup/README.md), put the
`.dump` files in `db_setup/dumps/`, list the target databases (e.g. `cadets_e3`) in
`POSTGRES_DATABASES`, run `docker compose --profile db up -d`, then restore each one:

```bash
docker compose exec doc-comp_postgres pg_restore -U grasp -d cadets_e3 /dumps/cadets_e3.dump
```

Optional, for faster graph construction:

```bash
docker compose exec doc-comp_postgres psql -U grasp -d cadets_e3 \
  -c "CREATE INDEX time_index ON event_table (timestamp_rec);"
```

### Sysdig captures

Batch (replaces the tables): put `.scap` files in `graphwatch_data_pipeline/sysdig/input/sysdig_scaps/`, then:

```bash
docker compose --profile pipeline run --rm doc-comp_sysdig_pipeline
```

Live (appends only new capture files; see the
[sysdig README](graphwatch_data_pipeline/sysdig/README.md)): use the API endpoint
`POST /jobs/sysdig-ingest` (below), or run it directly:

```bash
docker compose --profile cli run --rm doc-comp_grasp \
  python graphwatch_data_pipeline/sysdig/ingest.py --scap-dir graphwatch_data_pipeline/sysdig/input/live --db-name sysdig_live
```

## 6. Detection API

```bash
docker compose up -d            # doc-comp_detection_api, doc-comp_workbench (DB from .env.docker)
```

Interactive docs: http://localhost:8000/docs. All endpoints except `/health` need the API key:

```bash
KEY=$(grep ^GRASP_API_KEY= .env.docker | cut -d= -f2)
```

**Train and infer in one job.** The model is trained on the training range, then the
inference range is classified with it. `ground_truth_dir` is optional; with it the result
also contains the evaluation against the ground truth.

```bash
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" localhost:8000/jobs/pipeline -d '{
  "db_name": "cadets_e3",
  "train_start": "2018-04-02 00:00:00", "train_end": "2018-04-05 23:59:59",
  "inference_start": "2018-04-06 00:00:00", "inference_end": "2018-04-13 23:59:59",
  "ground_truth_dir": "ground_truth/cadets_e3"
}'
```

**Inference only, with a pretrained model.** Uses the latest model trained on `db_name`
(or pass `"model_id"`):

```bash
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" localhost:8000/jobs/inference -d '{
  "db_name": "cadets_e3",
  "inference_start": "2018-04-12 00:00:00", "inference_end": "2018-04-12 23:59:59"
}'
```

**Live sysdig ingestion.** Appends the capture files in `scap_dir` that were not ingested before
(and are complete) to `db_name`; with `run_inference` the newly ingested time range is classified
with the latest model of `db_name` (or `model_id`). Call it periodically:

```bash
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" localhost:8000/jobs/sysdig-ingest -d '{
  "db_name": "sysdig_live",
  "scap_dir": "graphwatch_data_pipeline/sysdig/input/live",
  "run_inference": true
}'
```

Optional: `min_file_age_seconds` (default 60), `max_files`, `model_id`, `ground_truth_dir`,
`reset` (drop tables and ingestion state first). sysdig models need an `experiment_config`
when training, e.g. `grasp/experiments/sysdig_spade/experiment_sysdig_1.yaml`.

`POST /jobs/train` only trains (same fields as `/jobs/pipeline` without the inference range).
Optional fields: `experiment_config` (hyperparameters YAML in the repo; default: the dataset's
`*_default` experiment) and `force_reload` (rebuild cached graphs from the DB).

Each call returns a job (`"status": "queued"`) with its `id`. Jobs run one at a time:

| Endpoint | Returns |
| --- | --- |
| `GET /jobs`, `GET /jobs/{id}` | job status: `queued`, `running`, `succeeded`, `failed`, `cancelled` |
| `GET /jobs/{id}/log?tail=200` | job output |
| `POST /jobs/{id}/cancel` | cancels a queued or running job |
| `GET /jobs/{id}/report` | summary: anomalies per window and, with ground truth, detections |
| `GET /jobs/{id}/anomalies?offset=0&limit=1000` | flagged nodes (uuid, observed vs. predicted command, window) |
| `GET /jobs/{id}/detailed-report` | full detailed report |
| `GET /models?db_name=...`, `GET /models/{id}` | trained models with their training range and metrics |

`db_name` is the database on the configured Postgres server; its dataset type is derived from
the name (`cadets_e3`, `theia_e5`, `sysdig_<anything>`, ...). Times use `YYYY-MM-DD HH:MM:SS`
in the dataset's time zone, as in the experiment configs.

### Where results are stored

| Path | Content |
| --- | --- |
| `data/models/<model_id>/` | trained model: location encoder, command mapping, classifier, clusters, settings |
| `data/inference/<run_id>/` | `report.json`, `anomalies.json`, `detailed_report.json`, classification, evaluation |
| `data/graphs/<db_name>/` | cached raw graphs per time window (shared by all models) |
| `data/api/jobs/` | job state and logs |

### Without the API

The same steps are available on the command line:

```bash
docker compose --profile cli run --rm doc-comp_grasp python train.py --db-name cadets_e3 \
  --train-start "2018-04-02 00:00:00" --train-end "2018-04-05 23:59:59"
docker compose --profile cli run --rm doc-comp_grasp python infer.py --db-name cadets_e3 \
  --inference-start "2018-04-06 00:00:00" --inference-end "2018-04-13 23:59:59" \
  --ground-truth-dir ground_truth/cadets_e3
docker compose --profile cli run --rm doc-comp_grasp     # full pipeline (main.py)
```

## 7. Workbench

```bash
docker compose up -d doc-comp_workbench
```

Open http://localhost:8501. See the
[grasp_contextualization README](grasp_contextualization/README.md) for the workflow.

## 8. GRASP GraphWatch console (interactive)

```bash
docker compose --profile console up -d      # add --profile db too if you use the bundled Postgres
```

Open http://localhost:8502. The console uses only the REST API, so it also serves as an
interactive API test. Its pages:

- **Home**: API status, job and model counts, links to the API docs and the workbench.
- **Run**: forms for all job types (pipeline, train, inference, sysdig ingest), with examples to
  prefill (DARPA cadets_e3, sysdig coffeebot). Dropdowns list models, experiment configs, ground
  truth and capture folders. Each form shows the equivalent `curl` command.
- **Jobs**: all jobs, status, live log (auto-refresh), cancel.
- **Results**: detection summary, anomalies per time window, ground-truth evaluation, anomaly
  list, download, and **Open in workbench**, which opens the workbench with that run
  preselected (`http://localhost:8501/?run=<run_id>`).
- **Models**: trained models with training range and metrics.

The API key is taken from `.env.docker` (`GRASP_API_KEY`) and can be changed in the sidebar
(*Connection*). Links use the host you opened the console with. Set `CONSOLE_PORT` in `.env` for
another port, and `API_PUBLIC_URL` / `WORKBENCH_PUBLIC_URL` in `.env.docker` if the API or the
workbench is reached under a different address.

## External database

To use a Postgres already running on the host (e.g. the PIDSMaker one on port 8888),
set in `.env.docker`:

```
db_host=host.docker.internal
db_port=8888
```

and start only the detection services with `--no-deps`, e.g.
`docker compose up -d --no-deps doc-comp_detection_api`.

## Useful commands

```bash
docker compose ps                     # status
docker compose logs -f doc-comp_workbench
docker compose down                   # stop and remove containers (keeps the DB volume)
docker compose down -v                # also delete the database volume doc-comp_postgres_data
```

## Troubleshooting

- **`could not select device driver "nvidia" with capabilities: [[gpu]]`**: the NVIDIA
  Container Toolkit is missing or Docker was not restarted. Repeat step 2.
- **`torch.cuda.is_available()` is `False`** inside the container: check
  `docker compose run --rm --no-deps doc-comp_grasp nvidia-smi` and that the host driver
  supports CUDA 12.9.
- **`Bind for 0.0.0.0:8888 failed: port is already allocated`**: see step 3.
- **Permission errors on `data/` or `pids_runs/`**: export `HOST_UID` and `HOST_GID` to match your user
  (`export HOST_UID=$(id -u) HOST_GID=$(id -g)`).
