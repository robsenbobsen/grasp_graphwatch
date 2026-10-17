# GRASP

Graph-Based Anomaly Detection Through Self-Supervised Classification

## Install

1. Clone and enter the repo:

```bash
git clone <repo-url>
cd grasp_graphwatch
```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Python package/project manager that replaces `pip` + `venv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify it's on your `PATH`:

```bash
uv --version
```

3. Create a virtual environment pinned to Python 3.12.7 (uv downloads that interpreter automatically if it isn't already installed):

```bash
uv venv --python 3.12.7
source .venv/bin/activate
```

4. Install core dependencies:

```bash
uv pip install -r requirements.txt
```

5. Install PyTorch and PyTorch Geometric for your platform (choose the right wheel for your CUDA/CPU setup, see https://pytorch.org/get-started/locally/ and https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html):

```bash
uv pip install torch  # pick version/build matching your CUDA
uv pip install torch_geometric
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
```

Example:
```bash
uv pip install torch==2.8.0 
uv pip install torch_geometric
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

### Data
We used PostgreSQL dumps, of the DARPA datasets, from the related work: 
```
https://ubc-provenance.github.io/PIDSMaker/
```
Please follow the instructions to setup the PostgreSQL database.
This, and our graph construction, ensures comparable results.

Alternatively, you can use a specialized installation guide to set up only the PostgreSQL database of PIDSMaker.
See [db_setup README](db_setup/README.md) for details.

#### Optional: better runtime through index
Creating an index for the timestamp column improves performance when creating graphs. 
This must be done for each database, if desired.
```
CREATE INDEX time_index ON event_table (timestamp_rec);
```

### .env
Create a .env file. 

```bash
cp .env_example .env
```

Edit the .env_example with the credentials and socket information to reach the db. 
If everything runs on the same host and you followed [db_setup README](db_setup/README.md) the defaults will work.


## Configure

- Default experiment: grasp/experiments/experiment_cadets_e3.yaml. 
- For other experiment configurations, see [grasp/experiments](grasp/experiments).

## Run

```bash
python main.py --experiment-config grasp/experiments/all_experiments/cadets_e3_default/experiment_cadets_e3.yaml
```

### Train once, infer later

`main.py` runs training and detection in one go. The same work can be split, so a
trained model is reused for new time ranges with the identical preprocessing:

```bash
python train.py --db-name cadets_e3 --train-start "2018-04-02 00:00:00" --train-end "2018-04-05 23:59:59"
python infer.py --db-name cadets_e3 --inference-start "2018-04-06 00:00:00" --inference-end "2018-04-13 23:59:59" \
  --ground-truth-dir ground_truth/cadets_e3   # optional
```

Both are also available as a REST API (`detection_api/`), see
[README_setup_docker_compose.md](README_setup_docker_compose.md).

## Optional: Explore reported anomalies with contextualization

After a run, you can optionally use the PIDS contextualization workbench to
inspect individual reported anomalies against sampled training examples for
their predicted and true labels — a first-pass aid for deciding whether each
is a false positive or a real anomaly. See the
[grasp_contextualization README](grasp_contextualization/README.md) for the
full workflow (generating a report, then browsing it):

```bash
streamlit run grasp_contextualization/pids_workbench_app.py
```

## Docker

Instead of the local install, the whole stack can run via Docker Compose (on NVIDIA GPUs).
The full step-by-step guide is [README_setup_docker_compose.md](README_setup_docker_compose.md). Everything is
named with the `doc-comp_` prefix (project `doc-comp`):

| Service | Purpose |
| --- | --- |
| `doc-comp_postgres` | optional PostgreSQL 17 (profile `db`, volume `doc-comp_postgres_data`, host port 8888) |
| `doc-comp_detection_api` | REST API on http://localhost:8000 (training/inference jobs) |
| `doc-comp_grasp` | runs `main.py` manually (profile `cli`) |
| `doc-comp_workbench` | contextualization workbench on http://localhost:8501 |
| `doc-comp_console` | GRASP GraphWatch console on http://localhost:8502, interactive front end for the API (profile `console`) |
| `doc-comp_sysdig_pipeline` | sysdig `.scap` → Postgres pipeline (profile `pipeline`) |

```bash
cp .env.docker.example .env.docker      # set GRASP_API_KEY
docker compose build
docker compose up -d                    # detection API + workbench, DB from .env.docker
# full pipeline via main.py instead of the API:
GRASP_EXPERIMENT_CONFIG=grasp/experiments/all_experiments/theia_e3_default/experiment_theia_e3.yaml \
  docker compose --profile cli run --rm doc-comp_grasp
```

Load PIDSMaker dumps (see [db_setup README](db_setup/README.md)) by placing them in
`db_setup/dumps/` (or set `DUMPS_DIR`) and restoring into the bundled DB:

```bash
docker compose exec doc-comp_postgres createdb -U grasp cadets_e3
docker compose exec doc-comp_postgres pg_restore -U grasp -d cadets_e3 /dumps/cadets_e3.dump
```

Sysdig captures: put `.scap` files in `graphwatch_data_pipeline/sysdig/input/sysdig_scaps`, then

```bash
docker compose --profile pipeline run --rm doc-comp_sysdig_pipeline
```

To use an existing database on the host instead, set `db_host=host.docker.internal` and its
host port in `.env.docker`.

The image is built for NVIDIA GPUs (CUDA 12.9) and the detection services reserve all GPUs, which
requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host.

## License

See [LICENSE](LICENSE) for details.
