"""'Home' (API status, links) and 'Models' pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from graphwatch_console.api import ApiError
from graphwatch_console.common import STATUS_ICON, api, api_docs_url, fmt_time, show_api_error, workbench_url


def page_home() -> None:
    st.title("GRASP GraphWatch")
    st.caption(
        "Interactive console for the detection REST API: train models, run inference, ingest "
        "sysdig captures live, follow jobs and inspect results."
    )
    client = api()
    try:
        client.health()
        st.success(f"API reachable at `{client.base_url}`")
    except ApiError as exc:
        show_api_error(exc, "Health check")
        return
    try:
        jobs = client.jobs()
        models = client.models()
    except ApiError as exc:
        show_api_error(exc, "Loading jobs/models")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs", len(jobs))
    c2.metric("Running / queued", sum(j["status"] in ("running", "queued") for j in jobs))
    c3.metric("Failed", sum(j["status"] == "failed" for j in jobs))
    c4.metric("Trained models", sum(m.get("status") == "complete" for m in models))

    st.subheader("Links")
    c1, c2 = st.columns(2)
    c1.link_button("REST API docs (Swagger) ↗", api_docs_url(), use_container_width=True)
    c2.link_button("Contextualization workbench ↗", workbench_url(), use_container_width=True)

    st.subheader("What you can do here")
    st.markdown(
        "- **Run**: start a *pipeline* (train + infer), *train* only, *inference* with a pretrained "
        "model, or *sysdig ingest* (live/static), with examples to prefill.\n"
        "- **Jobs**: status, live log, cancel.\n"
        "- **Results**: detection summary, anomalies per window, evaluation, anomaly list, and a "
        "button to open the run in the contextualization workbench.\n"
        "- **Models**: trained models with their training range and metrics."
    )

    if jobs:
        st.subheader("Recent jobs")
        st.dataframe(
            pd.DataFrame(
                {
                    "status": f"{STATUS_ICON.get(j['status'], '')} {j['status']}",
                    "id": j["id"],
                    "type": j["type"],
                    "db_name": j.get("params", {}).get("db_name", ""),
                    "created": fmt_time(j.get("created_at")),
                }
                for j in jobs[:8]
            ),
            use_container_width=True,
            hide_index=True,
        )


def page_models() -> None:
    st.header("Models")
    db_filter = st.text_input("Filter by db_name (empty = all)", "")
    try:
        models = api().models(db_filter.strip() or None)
    except ApiError as exc:
        show_api_error(exc, "Loading models")
        return
    if not models:
        st.info("No trained models yet. Train one on the **Run** page (Pipeline or Train).")
        return
    rows = []
    for m in models:
        metrics = m.get("train_metrics") or {}
        rows.append(
            {
                "model_id": m["model_id"],
                "status": m.get("status"),
                "db_name": m.get("db_name"),
                "dataset": m.get("dataset"),
                "train_start": m.get("train_start"),
                "train_end": m.get("train_end"),
                "windows": m.get("train_windows"),
                "classes": m.get("num_classes"),
                "train_accuracy": metrics.get("accuracy"),
                "train_macro_f1": metrics.get("macro_f1_score"),
                "created": fmt_time(m.get("created_at")),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    choice = st.selectbox("Details", [m["model_id"] for m in models])
    try:
        st.json(api().model(choice))
    except ApiError as exc:
        show_api_error(exc, "Loading the model")
