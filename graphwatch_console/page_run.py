"""'Run' page: submit pipeline / train / inference / sysdig-ingest jobs through the API."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from graphwatch_console.api import ApiError
from graphwatch_console.common import api, experiment_configs, ground_truth_dirs, scap_dirs, show_api_error

KINDS = ["pipeline", "train", "inference", "sysdig-ingest"]
DEFAULT_CONFIG = "(dataset default)"
NO_GT = "(none: no evaluation)"
LATEST_MODEL = "(latest model of db_name)"

PRESETS: dict[str, dict[str, Any]] = {
    "DARPA cadets_e3": {
        "db_name": "cadets_e3",
        "train_start": "2018-04-02 00:00:00",
        "train_end": "2018-04-05 23:59:59",
        "inference_start": "2018-04-06 00:00:00",
        "inference_end": "2018-04-13 23:59:59",
        "experiment_config": DEFAULT_CONFIG,
        "ground_truth_dir": "ground_truth/cadets_e3",
    },
    "sysdig coffeebot (CinITo exp. 2)": {
        "db_name": "sysdig_coffeebot_static",
        "train_start": "2026-09-08 17:00:00",
        "train_end": "2026-09-09 08:59:59",
        "inference_start": "2026-09-09 09:00:00",
        "inference_end": "2026-09-09 10:59:59",
        "experiment_config": "grasp/experiments/sysdig_cinito/experiment_sysdig_cinito_coffeebot_2.yaml",
        "ground_truth_dir": NO_GT,
        "scap_dir": "graphwatch_data_pipeline/sysdig/backup_data/cinito/experiment2/coffeebot",
    },
}


def _apply_preset() -> None:
    preset = PRESETS[st.session_state["run_preset"]]
    for kind in KINDS:
        for field, value in preset.items():
            st.session_state[f"{kind}_{field}"] = value


def _time_pair(kind: str, prefix: str, label: str) -> tuple[str, str]:
    c1, c2 = st.columns(2)
    start = c1.text_input(f"{label} start", key=f"{kind}_{prefix}_start", placeholder="YYYY-MM-DD HH:MM:SS")
    end = c2.text_input(f"{label} end", key=f"{kind}_{prefix}_end", placeholder="YYYY-MM-DD HH:MM:SS")
    return start.strip(), end.strip()


def _select(label: str, options: list[str], key: str, help: str | None = None) -> str:
    if key in st.session_state and st.session_state[key] not in options:
        options = [*options, st.session_state[key]]  # e.g. a preset value not listed locally
    return st.selectbox(label, options, key=key, help=help)


def _config(kind: str) -> str | None:
    value = _select(
        "Experiment config (hyperparameters, window sizes)",
        [DEFAULT_CONFIG, *experiment_configs()],
        f"{kind}_experiment_config",
        help="sysdig databases have no default; pick e.g. a sysdig_cinito config.",
    )
    return None if value == DEFAULT_CONFIG else value


def _ground_truth(kind: str) -> str | None:
    value = _select("Ground truth (optional)", [NO_GT, *ground_truth_dirs()], f"{kind}_ground_truth_dir")
    return None if value == NO_GT else value


def _model(kind: str, db_name: str) -> str | None:
    try:
        models = [m for m in api().models() if m.get("status") == "complete"]
    except ApiError:
        models = []
    labels = {LATEST_MODEL: None}
    for m in models:
        labels[f"{m['model_id']}  ({m.get('dataset')}, {m.get('train_start')} → {m.get('train_end')})"] = m["model_id"]
    choice = st.selectbox(
        "Model",
        list(labels),
        key=f"{kind}_model",
        help=f"Default: latest complete model trained on '{db_name or '…'}'. Any model of the "
        "same dataset type works.",
    )
    return labels[choice]


def _body(kind: str) -> dict[str, Any]:
    """Form fields of one job type -> request body."""
    body: dict[str, Any] = {}
    if kind == "sysdig-ingest":
        body["db_name"] = st.text_input("Database (sysdig_<name>, created if missing)", key=f"{kind}_db_name")
        dirs = scap_dirs()
        labels = {f"{d['path']}  ({d['files']} files, {d['gb']:.1f} GB)": d["path"] for d in dirs}
        current = st.session_state.get(f"{kind}_scap_dir")
        if current and current not in labels.values():
            labels[current] = current
        keys = list(labels)
        idx = next((i for i, k in enumerate(keys) if labels[k] == current), 0) if keys else 0
        if keys:
            choice = st.selectbox("Capture folder (scap_dir)", keys, index=idx)
            body["scap_dir"] = labels[choice]
            st.session_state[f"{kind}_scap_dir"] = body["scap_dir"]
        else:
            body["scap_dir"] = st.text_input("Capture folder (scap_dir)", key=f"{kind}_scap_dir")
        c1, c2 = st.columns(2)
        body["min_file_age_seconds"] = c1.number_input(
            "Min. file age (s): skip files still being written", min_value=0, value=60, key=f"{kind}_age"
        )
        max_files = c2.number_input("Max files per job (0 = all)", min_value=0, value=0, key=f"{kind}_max")
        body["max_files"] = int(max_files) or None
        body["reset"] = st.checkbox(
            "Reset: drop the tables and ingestion state of this database first", key=f"{kind}_reset"
        )
        body["run_inference"] = st.checkbox(
            "Run inference on the newly ingested time range", key=f"{kind}_run_inference"
        )
        if body["run_inference"]:
            body["model_id"] = _model(kind, body["db_name"])
            body["ground_truth_dir"] = _ground_truth(kind)
        return body

    body["db_name"] = st.text_input("Database (db_name)", key=f"{kind}_db_name")
    if kind in ("pipeline", "train"):
        body["train_start"], body["train_end"] = _time_pair(kind, "train", "Training")
        body["experiment_config"] = _config(kind)
    if kind in ("pipeline", "inference"):
        body["inference_start"], body["inference_end"] = _time_pair(kind, "inference", "Inference")
        if kind == "inference":
            body["model_id"] = _model(kind, body["db_name"])
        body["ground_truth_dir"] = _ground_truth(kind)
    body["force_reload"] = st.checkbox("Rebuild cached graphs from the DB (force_reload)", key=f"{kind}_force")
    return body


def _curl(kind: str, body: dict[str, Any]) -> str:
    clean = {k: v for k, v in body.items() if v not in (None, "", False)}
    return (
        f"curl -s -X POST $API/jobs/{kind} -H \"X-API-Key: $KEY\" -H \"Content-Type: application/json\" "
        f"-d '{json.dumps(clean)}' | jq"
    )


DESCRIPTIONS = {
    "pipeline": "Train a new model on the training range, then classify the inference range with it.",
    "train": "Only pretrain a model; use it later with *Inference* or live ingestion.",
    "inference": "Classify a time range with a pretrained model (same preprocessing as in training).",
    "sysdig-ingest": "Append new, complete capture files of a folder to a sysdig database (live or "
    "static); optionally classify the newly ingested range right away.",
}


def page_run() -> None:
    st.header("Run a job")
    c1, c2 = st.columns([3, 1])
    c1.selectbox("Prefill with an example", list(PRESETS), key="run_preset")
    c2.button("Apply example", on_click=_apply_preset, use_container_width=True)

    tabs = st.tabs(["Pipeline (train + infer)", "Train", "Inference", "Sysdig ingest"])
    for tab, kind in zip(tabs, KINDS):
        with tab:
            st.caption(DESCRIPTIONS[kind])
            body = _body(kind)
            if st.button("Submit job", key=f"{kind}_submit", type="primary"):
                try:
                    job = api().submit(kind, body)
                except ApiError as exc:
                    show_api_error(exc, "Submitting the job")
                else:
                    st.session_state["selected_job"] = job["id"]
                    st.success(
                        f"Job `{job['id']}` queued ({job['type']}). Follow it on the **Jobs** page."
                    )
                    st.json(job, expanded=False)
            with st.expander("Equivalent curl command"):
                st.code(_curl(kind, body), language="bash")
