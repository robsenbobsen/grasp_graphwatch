# Example walkthrough with the web app (GraphWatch console)

The same flows as in [example_walkthrough.md](example_walkthrough.md) and
[example_walkthrough_with_sysdig_data.md](example_walkthrough_with_sysdig_data.md), but by clicking
in the browser instead of `curl`. Setup (GPU, `.env.docker` with `GRASP_API_KEY` and the database,
build) is in [example_walkthrough.md](example_walkthrough.md), steps 1–3.

## 1. Start

```bash
docker compose --profile console up -d                # DB from .env.docker (e.g. DARPA server)
docker compose --profile db --profile console up -d   # same, plus the bundled Postgres (sysdig)
```

| URL | What |
| --- | --- |
| http://localhost:8502 | GraphWatch console |
| http://localhost:8501 | contextualization workbench |
| http://localhost:8000/docs | REST API docs |

**Home** should show *API reachable*. If it shows an API key error, check `GRASP_API_KEY` in
`.env.docker`, or enter the key in the sidebar under *Connection*.

## 2. DARPA cadets_e3: train + infer

1. **Run** → *Prefill with an example*: `DARPA cadets_e3` → **Apply example**.
2. Tab **Pipeline (train + infer)**: check the values and press **Submit job**.
3. **Jobs** → the new job is preselected. *Auto-refresh* shows the live log until the status is
   ✅ `succeeded`.
4. **Show results** (or the **Results** page): summary, anomalies per time window, evaluation
   against `ground_truth/cadets_e3`, anomaly list.
5. **Open in workbench ↗** opens the workbench with this run already selected. Set *Max
   anomalies* to e.g. `20` → **Run analysis** → then **Overview** / **Context View** (pick the
   report under *Choose report* in the sidebar).

## 3. Pretrained model: train once, infer later

1. **Run** → tab **Train** (same example values) → **Submit job**. Wait for ✅ on **Jobs**.
2. **Models**: the new model with its training range and metrics.
3. **Run** → tab **Inference**: any time range, e.g. `2018-04-12 00:00:00` → `2018-04-12 23:59:59`.
   *Model*: keep *latest model of db_name* or pick one → **Submit job** → results as in step 2.

## 4. Sysdig coffeebot: static

Needs `.env.docker` pointing at the bundled Postgres; see the
[sysdig walkthrough](example_walkthrough_with_sysdig_data.md), "Use the bundled PostgreSQL".

1. **Run** → example `sysdig coffeebot (CinITo exp. 2)` → **Apply example**.
2. Tab **Sysdig ingest**: *Capture folder* `…/cinito/experiment2/coffeebot (10 files, …)`
   → **Submit job**. It takes ~10 minutes; follow it on **Jobs**. The log shows `Ingesting 10 new
   file(s)`.
3. Tab **Pipeline (train + infer)** (example values, with the sysdig experiment config) → **Submit
   job** → **Results** → **Open in workbench ↗**.

## 5. Sysdig coffeebot: live

1. Link the first files into a live folder, as in the
   [sysdig walkthrough](example_walkthrough_with_sysdig_data.md), section B
   (`add 0` … `add 6`).
2. **Run** → **Sysdig ingest**: *Database* `sysdig_coffeebot_live`, *Capture folder* the live
   folder → **Submit job**.
3. **Train** on `sysdig_coffeebot_live` (training end `2026-09-09 06:29:59`, sysdig experiment
   config) and wait for ✅.
4. For each new file (`add 7`, `add 8`, `add 9`): **Sysdig ingest** with **Run inference on the newly
   ingested time range** checked → **Submit job**. Each job ingests only the new file and
   classifies its time range. Its results are on **Results**, one per file.

## 6. Troubleshooting

| What you see | Why / what to do |
| --- | --- |
| *invalid or missing API key* | key in `.env.docker` and the sidebar *Connection* differ |
| `HTTP 422: … expected format 'YYYY-MM-DD HH:MM:SS'` | times need date **and** time |
| `HTTP 404: No trained model for db_name …` | train first, or pick a model of the same dataset type |
| Job ❌ with `connection … refused` in the log | `.env.docker` points at the wrong database server |
| Log: `No new complete capture files` | all files ingested already, or still younger than *Min. file age* |
| *Results* shows nothing for a sysdig job | ingest without *Run inference*, or no new files: nothing was classified |

Each form on **Run** has an *Equivalent curl command*, so anything done here can be repeated from
the shell.
