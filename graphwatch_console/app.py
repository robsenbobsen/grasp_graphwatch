"""
GRASP GraphWatch console: interactive front end for the detection REST API.

Everything goes through the API (detection_api/), so the console doubles as an interactive
API test. Results link to the contextualization workbench (grasp_contextualization/).

Run (from the repo root):
  streamlit run graphwatch_console/app.py
Docker: docker compose --profile console up -d  ->  http://localhost:8502
Settings (env): GRASP_API_URL, GRASP_API_KEY; optional API_PUBLIC_URL / WORKBENCH_PUBLIC_URL
for the browser links (default: same host as the console, ports API_PORT / WORKBENCH_PORT).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(page_title="GRASP GraphWatch", page_icon="🛰️", layout="wide")

from graphwatch_console.api import DEFAULT_API_URL  # noqa: E402
from graphwatch_console.common import api_docs_url, workbench_url  # noqa: E402
from graphwatch_console.page_home import page_home, page_models  # noqa: E402
from graphwatch_console.page_jobs import page_jobs, page_results  # noqa: E402
from graphwatch_console.page_run import page_run  # noqa: E402

PAGES = {"Home": page_home, "Run": page_run, "Jobs": page_jobs, "Results": page_results,
         "Models": page_models}  # fmt: skip

st.sidebar.title("GRASP GraphWatch")
page = st.sidebar.radio("Page", list(PAGES), key="page")

with st.sidebar.expander("Connection"):
    st.text_input("API URL", DEFAULT_API_URL, key="api_url")
    st.text_input(
        "API key",
        os.getenv("GRASP_API_KEY", ""),
        key="api_key",
        type="password",
        help="Default: GRASP_API_KEY of the environment (.env.docker in Docker).",
    )
st.sidebar.markdown(f"[REST API docs ↗]({api_docs_url()})  \n[Contextualization workbench ↗]({workbench_url()})")

PAGES[page]()
