"""'Jobs' page (status, log, cancel) and 'Results' page (report, anomalies, workbench link)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from graphwatch_console.api import ApiError
from graphwatch_console.common import FINISHED, STATUS_ICON, api, fmt_time, show_api_error, workbench_url

WINDOW_RE = re.compile(r"_graph_(\d{18,19})_to_(\d{18,19})")


def window_label(path: str) -> str:
    """'..._graph_<ns>_to_<ns>_extended.pt' -> 'YYYY-MM-DD HH:MM → HH:MM' (UTC)."""
    m = WINDOW_RE.search(str(path))
    if not m:
        return str(path).rsplit("/", 1)[-1]
    start, end = (datetime.fromtimestamp(int(v) / 1e9, tz=timezone.utc) for v in m.groups())
    return f"{start:%Y-%m-%d %H:%M} → {end:%H:%M}"


def _jobs_frame(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "status": f"{STATUS_ICON.get(j['status'], '')} {j['status']}",
            "id": j["id"],
            "type": j["type"],
            "db_name": j.get("params", {}).get("db_name", ""),
            "created": fmt_time(j.get("created_at")),
            "started": fmt_time(j.get("started_at")),
            "finished": fmt_time(j.get("finished_at")),
            "model_id": j.get("model_id") or "",
            "error": j.get("error") or "",
        }
        for j in jobs
    ]
    return pd.DataFrame(rows)


def _job_label(j: dict[str, Any]) -> str:
    return (
        f"{STATUS_ICON.get(j['status'], '')} {j['id']} | {j['type']} | "
        f"{j.get('params', {}).get('db_name', '')} | {fmt_time(j.get('created_at'))}"
    )


def _pick_job(jobs: list[dict[str, Any]], key: str, label: str) -> dict[str, Any] | None:
    if not jobs:
        return None
    ids = [j["id"] for j in jobs]
    selected = st.session_state.get("selected_job")
    idx = ids.index(selected) if selected in ids else 0
    choice = st.selectbox(label, ids, index=idx, format_func=lambda i: _job_label(jobs[ids.index(i)]), key=key)
    st.session_state["selected_job"] = choice
    return jobs[ids.index(choice)]


def _go_to_results() -> None:
    st.session_state["page"] = "Results"


def page_jobs() -> None:
    st.header("Jobs")
    try:
        jobs = api().jobs()
    except ApiError as exc:
        show_api_error(exc, "Loading jobs")
        return
    if not jobs:
        st.info("No jobs yet. Start one on the **Run** page.")
        return
    c1, c2 = st.columns([4, 1])
    c2.button("Refresh", use_container_width=True)
    with c1:
        counts = pd.Series([j["status"] for j in jobs]).value_counts()
        st.caption(" · ".join(f"{STATUS_ICON.get(s, '')} {s}: {n}" for s, n in counts.items()))
    st.dataframe(_jobs_frame(jobs), use_container_width=True, hide_index=True, height=240)

    job = _pick_job(jobs, "jobs_select", "Job")
    if job is None:
        return
    job_id = job["id"]

    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown(f"**{STATUS_ICON.get(job['status'], '')} {job['status']}** · `{job_id}` · {job['type']}")
    if job["status"] not in FINISHED and c2.button("Cancel job", type="secondary"):
        try:
            api().cancel(job_id)
            st.rerun()
        except ApiError as exc:
            show_api_error(exc, "Cancel")
    if job["status"] == "succeeded" and job.get("run_id"):
        c3.button("Show results", on_click=_go_to_results, type="primary")
    if job.get("error"):
        st.error(job["error"])
    with st.expander("Parameters and steps"):
        st.json({"params": job.get("params"), "steps": job.get("steps"), "model_id": job.get("model_id"),
                 "run_id": job.get("run_id")})  # fmt: skip

    c1, c2 = st.columns([1, 3])
    auto = c1.toggle("Auto-refresh (5 s)", value=job["status"] not in FINISHED)
    tail = c2.slider("Log lines", 20, 2000, 200, step=20)

    @st.fragment(run_every=5 if auto else None)
    def live_log() -> None:
        try:
            current = api().job(job_id)
            text = api().log(job_id, tail)
        except ApiError as exc:
            show_api_error(exc, "Loading the log")
            return
        st.caption(f"Status: {STATUS_ICON.get(current['status'], '')} {current['status']}"
                   f" · updated {datetime.now():%H:%M:%S}")  # fmt: skip
        st.code(text or "(no output yet)", language="text")

    live_log()


def _result_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [j for j in jobs if j["status"] == "succeeded" and j.get("run_id")]


def page_results() -> None:
    st.header("Results")
    try:
        jobs = _result_jobs(api().jobs())
    except ApiError as exc:
        show_api_error(exc, "Loading jobs")
        return
    if not jobs:
        st.info("No finished job with inference results yet.")
        return
    job = _pick_job(jobs, "results_select", "Job with inference results")
    if job is None:
        return
    job_id, run_id = job["id"], job["run_id"]

    try:
        report = api().report(job_id)
    except ApiError as exc:
        # e.g. a sysdig-ingest job that found no new files ran no inference
        show_api_error(exc, "Loading the report")
        return

    c1, c2 = st.columns([3, 1])
    c1.markdown(f"Run `{run_id}` · model `{job.get('model_id')}` · db `{job['params'].get('db_name')}`")
    c2.link_button("Open in workbench ↗", workbench_url(run_id), use_container_width=True,
                   help="Contextualization workbench with this run preselected")  # fmt: skip

    summary = report.get("detection_summary", {})
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [("Nodes", "total_nodes"), ("Time windows", "time_windows"),
                                        ("Anomalies", "anomalies"), ("Anomalous nodes", "unique_anomalous_nodes"),
                                        ("Cluster corrections", "corrections_applied")]):  # fmt: skip
        col.metric(label, f"{summary.get(key, 0):,}")

    windows = pd.DataFrame(report.get("time_window_breakdown", []))
    if not windows.empty:
        st.subheader("Anomalies per time window (UTC)")
        windows["window"] = windows["time_window"].map(window_label)
        windows = windows.sort_values("window")
        st.bar_chart(windows.set_index("window")[["corrected_anomalies"]], height=220)
        with st.expander("Table"):
            st.dataframe(windows.drop(columns=["time_window"]), use_container_width=True, hide_index=True)

    hits = report.get("evaluation_hits") or {}
    if hits.get("by_group") or hits.get("by_file"):
        st.subheader("Evaluation against ground truth")
        st.dataframe(pd.DataFrame(hits.get("by_group", [])), use_container_width=True, hide_index=True)
        with st.expander("Per ground-truth file"):
            st.dataframe(pd.DataFrame(hits.get("by_file", [])), use_container_width=True, hide_index=True)
    else:
        st.caption("No ground-truth evaluation for this run.")

    st.subheader("Anomalies")
    c1, c2, c3 = st.columns([1, 1, 2])
    limit = c1.number_input("Page size", 10, 10_000, 200, step=50)
    page = c2.number_input("Page", 1, value=1)
    query = c3.text_input("Filter (command or node)", "")
    try:
        data = api().anomalies(job_id, offset=(int(page) - 1) * int(limit), limit=int(limit))
    except ApiError as exc:
        show_api_error(exc, "Loading anomalies")
        return
    items = pd.DataFrame(data["items"])
    if not items.empty:
        items["window"] = items["window"].map(window_label)
        if query:
            mask = items.astype(str).apply(lambda col: col.str.contains(query, case=False, regex=False))
            items = items[mask.any(axis=1)]
    st.caption(f"{data['total']:,} anomalies in total · showing {len(items):,}")
    st.dataframe(items, use_container_width=True, hide_index=True)

    if st.button("Prepare detailed report download"):
        try:
            st.download_button("Download detailed_report.json", api().detailed_report(job_id),
                               file_name=f"{run_id}_detailed_report.json", mime="application/json")  # fmt: skip
        except ApiError as exc:
            show_api_error(exc, "Download")
