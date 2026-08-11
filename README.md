# GRASP

Graph-Based Anomaly Detection Through Self-Supervised Classification

## Install

1. Clone and enter the repo:

```bash
git clone <repo-url>
cd grasp
```

2. Install dependencies.

### Option A: uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the virtualenv for you and locks
exact versions. It also picks the right `torch`/`torch-geometric` build (CPU,
or a matching CUDA version) automatically, so you don't need to know your
CUDA version or select a wheel by hand.

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you
don't have it, then:

```bash
./scripts/install.sh
```

This inspects `nvidia-smi` and syncs the matching backend — equivalent to
`uv sync --extra <cpu|cu126|cu128|cu130>`. Pass `--extra` yourself to
override the detected choice, e.g. `./scripts/install.sh --extra cu128`.

Activate the environment with `source .venv/bin/activate`, or prefix
commands with `uv run` (e.g. `uv run python main.py ...`). Verify the
install with:

```bash
uv run check-install
```

### Option B: pip

Unlike uv, pip won't pick a CUDA build for you — choose `cpu`, `cu126`,
`cu128`, or `cu130` based on your setup (see
[PyTorch's install matrix](https://pytorch.org/get-started/locally/)) and use
it consistently in the extra, `--extra-index-url`, and `-f` URL below (shown
here for `cpu`):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install -e ".[cpu]" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -f https://data.pyg.org/whl/torch-2.9.1+cpu.html
```

For a CUDA build, swap `cpu` for `cu126`/`cu128`/`cu130` in all three
places. Dev tooling (pytest) isn't pulled in by either of the above; add it
with:

```bash
pip install pytest==9.0.2 pytest-cov==7.0.0
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

## License

See [LICENSE](LICENSE) for details.
