"""Thin client for the detection REST API (detection_api/app.py).

The console talks to GRASP only through this client, so everything it shows is what the
API returns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_API_URL = os.getenv("GRASP_API_URL", "http://localhost:8000")


class ApiError(Exception):
    """A non-2xx response, with the API's error detail as message."""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {format_detail(detail)}")


def format_detail(detail: Any) -> str:
    """FastAPI errors: a string, or for 422 a list of {loc, msg} per invalid field."""
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(p) for p in item.get("loc", []) if p != "body")
                parts.append(f"{loc}: {item.get('msg', item)}" if loc else str(item.get("msg", item)))
            else:
                parts.append(str(item))
        return "; ".join(parts)
    return str(detail)


@dataclass
class Api:
    base_url: str
    api_key: str
    timeout: float = 30.0

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            resp = httpx.request(
                method,
                self.base_url.rstrip("/") + path,
                headers={"X-API-Key": self.api_key},
                timeout=self.timeout,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise ApiError(0, f"API not reachable at {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise ApiError(resp.status_code, detail)
        return resp

    def get(self, path: str, **params: Any) -> Any:
        params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", path, params=params).json()

    def post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, json=body).json()

    # --- endpoints ---------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self.get("/health")

    def submit(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        """kind: pipeline | train | inference | sysdig-ingest."""
        return self.post(f"/jobs/{kind}", {k: v for k, v in body.items() if v not in (None, "")})

    def jobs(self) -> list[dict[str, Any]]:
        return self.get("/jobs")

    def job(self, job_id: str) -> dict[str, Any]:
        return self.get(f"/jobs/{job_id}")

    def log(self, job_id: str, tail: int = 200) -> str:
        return self._request("GET", f"/jobs/{job_id}/log", params={"tail": tail}).text

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self.post(f"/jobs/{job_id}/cancel")

    def report(self, job_id: str) -> dict[str, Any]:
        return self.get(f"/jobs/{job_id}/report")

    def anomalies(self, job_id: str, offset: int = 0, limit: int = 1000) -> dict[str, Any]:
        return self.get(f"/jobs/{job_id}/anomalies", offset=offset, limit=limit)

    def detailed_report(self, job_id: str) -> bytes:
        return self._request("GET", f"/jobs/{job_id}/detailed-report").content

    def models(self, db_name: str | None = None) -> list[dict[str, Any]]:
        return self.get("/models", db_name=db_name or None)

    def model(self, model_id: str) -> dict[str, Any]:
        return self.get(f"/models/{model_id}")
