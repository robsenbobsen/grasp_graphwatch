# Setup with Docker Compose (GPU)

This guide runs the whole GRASP repository in Docker: PostgreSQL, GRASP training/detection,
the contextualization workbench and the sysdig data pipeline. Everything uses the `doc-comp`
prefix (project `doc-comp`) and runs on NVIDIA GPUs.

| Service | Purpose |
| --- | --- |
| `doc-comp_postgres` | PostgreSQL 17, data in volume `doc-comp_postgres_data`, host port `8888` |
| `doc-comp_grasp` | runs `main.py` with the experiment in `GRASP_EXPERIMENT_CONFIG` |
| `doc-comp_workbench` | contextualization workbench on http://localhost:8501 |
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

The defaults connect the detection services to the bundled `doc-comp_postgres`
(`db_host=doc-comp_postgres`, `db_port=5432`, user/password `grasp`).

Variables that compose itself reads (set them in your shell or in `.env`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `POSTGRES_PORT` | `8888` | host port of `doc-comp_postgres` |
| `WORKBENCH_PORT` | `8501` | host port of the workbench |
| `GRASP_EXPERIMENT_CONFIG` | `.../cadets_e3_default/experiment_cadets_e3.yaml` | experiment run by `doc-comp_grasp` |
| `DUMPS_DIR` | `./db_setup/dumps` | directory with PIDSMaker `.dump` files, mounted at `/dumps` |
| `HOST_UID` / `HOST_GID` | `1000` | user the detection service containers run as |
| `TORCH_VARIANT` | `cu129` | PyTorch/PyG wheel build; `cpu` for a CPU-only image |

**Port 8888 already in use** (e.g. by an older `grasp_graphwatch_postgres` container):
either stop that container, start with another port (`export POSTGRES_PORT=8899`), or use
the existing database instead of the bundled one (see [External database](#external-database)).

## 4. Build

```bash
docker compose build
```

The image is `python:3.12.7-slim` with torch 2.8.0 (CUDA 12.9), PyTorch Geometric,
`requirements.txt`, `grasp_contextualization/requirements.txt` and the `sysdig` CLI.

## 5. Start the database and load data

```bash
docker compose up -d doc-comp_postgres
```

### DARPA datasets (PIDSMaker dumps)

Download the dumps as described in the [db_setup README](db_setup/README.md), put the
`.dump` files in `db_setup/dumps/`, then restore each one, e.g.:

```bash
docker compose exec doc-comp_postgres createdb -U grasp cadets_e3
docker compose exec doc-comp_postgres pg_restore -U grasp -d cadets_e3 /dumps/cadets_e3.dump
```

Optional, for faster graph construction:

```bash
docker compose exec doc-comp_postgres psql -U grasp -d cadets_e3 \
  -c "CREATE INDEX time_index ON event_table (timestamp_rec);"
```

### Sysdig captures

Put `.scap` files in `graphwatch_data_pipeline/sysdig/input/sysdig_scaps/`, then:

```bash
docker compose --profile pipeline run --rm doc-comp_sysdig_pipeline
```

## 6. Run an experiment

```bash
docker compose run --rm doc-comp_grasp
```

Another experiment:

```bash
GRASP_EXPERIMENT_CONFIG=grasp/experiments/all_experiments/theia_e3_default/experiment_theia_e3.yaml \
  docker compose run --rm doc-comp_grasp
```

Or any command inside the image:

```bash
docker compose run --rm doc-comp_grasp python main_pretty_classification_report.py --help
```

## 7. Workbench

```bash
docker compose up -d doc-comp_workbench
```

Open http://localhost:8501. See the
[grasp_contextualization README](grasp_contextualization/README.md) for the workflow.

## External database

To use a Postgres already running on the host (e.g. the PIDSMaker one on port 8888),
set in `.env.docker`:

```
db_host=host.docker.internal
db_port=8888
```

and start only the detection services with `--no-deps`, e.g.
`docker compose run --rm --no-deps doc-comp_grasp`.

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
