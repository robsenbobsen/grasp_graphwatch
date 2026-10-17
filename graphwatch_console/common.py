"""Shared helpers: API connection, browser links, repository lists for dropdowns."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import streamlit as st

from graphwatch_console.api import DEFAULT_API_URL, Api, ApiError

REPO_ROOT = Path(__file__).resolve().parent.parent
FINISHED = {"succeeded", "failed", "cancelled"}
STATUS_ICON = {"queued": "⏳", "running": "🔄", "succeeded": "✅", "failed": "❌", "cancelled": "⛔"}


def api() -> Api:
    """API client from the sidebar settings (defaults: env GRASP_API_URL / GRASP_API_KEY)."""
    return Api(
        st.session_state.get("api_url", DEFAULT_API_URL),
        st.session_state.get("api_key", os.getenv("GRASP_API_KEY", "")),
    )


def show_api_error(exc: ApiError, action: str = "Request") -> None:
    if exc.status_code == 401:
        st.error(f"{action} failed: invalid or missing API key (sidebar → Connection).")
    elif exc.status_code == 0:
        st.error(f"{action} failed: {exc.detail}")
    else:
        st.error(f"{action} failed: {exc}")


# --- links that work in the user's browser ----------------------------------------------


def _browser_host() -> str:
    """Host name the browser used to reach this console (so links work from other machines)."""
    try:
        host = st.context.headers.get("Host", "")
    except Exception:  # noqa: BLE001
        host = ""
    return host.rsplit(":", 1)[0] if host else "localhost"


def public_url(env_var: str, port_env: str, default_port: str) -> str:
    explicit = os.getenv(env_var)
    if explicit:
        return explicit.rstrip("/")
    return f"http://{_browser_host()}:{os.getenv(port_env, default_port)}"


def api_docs_url() -> str:
    return public_url("API_PUBLIC_URL", "API_PORT", "8000") + "/docs"


def workbench_url(run_id: str | None = None) -> str:
    url = public_url("WORKBENCH_PUBLIC_URL", "WORKBENCH_PORT", "8501")
    return f"{url}/?{urlencode({'run': run_id})}" if run_id else url


# --- repository lists for dropdowns (read-only; the API validates every value) -------------


@st.cache_data(ttl=60, show_spinner=False)
def experiment_configs() -> list[str]:
    root = REPO_ROOT / "grasp" / "experiments"
    return sorted(str(p.relative_to(REPO_ROOT)) for p in root.rglob("*.yaml"))


@st.cache_data(ttl=60, show_spinner=False)
def ground_truth_dirs() -> list[str]:
    root = REPO_ROOT / "ground_truth"
    return sorted(str(p.relative_to(REPO_ROOT)) for p in root.iterdir() if p.is_dir()) if root.exists() else []


@st.cache_data(ttl=30, show_spinner=False)
def scap_dirs(max_depth: int = 5) -> list[dict[str, Any]]:
    """Directories below graphwatch_data_pipeline/sysdig that contain *.scap* files."""
    base = REPO_ROOT / "graphwatch_data_pipeline" / "sysdig"
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        depth = len(Path(dirpath).relative_to(base).parts)
        if depth >= max_depth or Path(dirpath).name == "output":
            dirnames[:] = []
        scaps = [f for f in filenames if ".scap" in f]
        if scaps:
            size = sum((Path(dirpath) / f).stat().st_size for f in scaps if (Path(dirpath) / f).exists())
            out.append(
                {"path": str(Path(dirpath).relative_to(REPO_ROOT)), "files": len(scaps), "gb": size / 1e9}
            )
    return sorted(out, key=lambda d: d["path"])


def fmt_time(iso: str | None) -> str:
    return iso.replace("T", " ")[:19] if iso else ""
