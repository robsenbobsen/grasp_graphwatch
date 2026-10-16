"""REST API of the GRASP detection service.

Jobs (training, inference, or both) are queued and run one at a time; poll
GET /jobs/{id} for the status. Every endpoint except /health needs the X-API-Key header
matching GRASP_API_KEY. Interactive docs: /docs.
"""

import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Security, status
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator, model_validator

from detection_api.jobs import JobManager
from grasp import stages
from grasp.schema import TimeStringFormat

PROJECT_ROOT: Path = stages.PROJECT_ROOT
API_KEY: str = os.getenv("GRASP_API_KEY", "")
if not API_KEY:
    raise RuntimeError("GRASP_API_KEY is not set; refusing to start an unauthenticated API.")

app = FastAPI(
    title="GRASP detection service",
    description="Train GRASP models and run anomaly detection on provenance data.",
    version="1.0.0",
)
jobs = JobManager(jobs_dir=PROJECT_ROOT / "data" / "api" / "jobs", project_root=PROJECT_ROOT)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Annotated[str | None, Security(_api_key_header)]) -> None:
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid X-API-Key")


Auth = [Depends(require_api_key)]

TIME_EXAMPLE = "2018-04-02 00:00:00"


def _check_time(value: str) -> str:
    try:
        datetime.strptime(value, TimeStringFormat.FMT.value)  # noqa: DTZ007
    except ValueError as exc:
        raise ValueError(f"expected format 'YYYY-MM-DD HH:MM:SS', got '{value}'") from exc
    return value


def _check_range(start: str, end: str, label: str) -> None:
    if end <= start:
        raise ValueError(f"{label}_end must be after {label}_start")


def _project_path(value: str | None, must_be_dir: bool) -> str | None:
    """Relative path inside the repository that exists (no escaping via ..)."""
    if value is None:
        return None
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.exists():
        raise ValueError(f"'{value}' does not exist in the repository")
    if must_be_dir != path.is_dir():
        raise ValueError(f"'{value}' must be a {'directory' if must_be_dir else 'file'}")
    return str(path.relative_to(PROJECT_ROOT))


class DbRequest(BaseModel):
    db_name: str = Field(pattern=r"^[A-Za-z0-9_]+$", examples=["cadets_e3"])
    force_reload: bool = Field(False, description="Rebuild cached graphs from the DB")

    @field_validator("db_name")
    @classmethod
    def check_dataset(cls, v: str) -> str:
        stages.dataset_for_db(v)
        return v


class TrainFields(BaseModel):
    train_start: str = Field(examples=[TIME_EXAMPLE])
    train_end: str = Field(examples=["2018-04-05 23:59:59"])
    experiment_config: str | None = Field(
        None,
        description="Settings/hyperparameters YAML in the repo (default: the dataset's *_default).",
        examples=["grasp/experiments/all_experiments/cadets_e3_default/experiment_cadets_e3.yaml"],
    )

    @field_validator("train_start", "train_end")
    @classmethod
    def check_train_times(cls, v: str) -> str:
        return _check_time(v)

    @field_validator("experiment_config")
    @classmethod
    def check_experiment_config(cls, v: str | None) -> str | None:
        return _project_path(v, must_be_dir=False)


class InferenceFields(BaseModel):
    inference_start: str = Field(examples=["2018-04-06 00:00:00"])
    inference_end: str = Field(examples=["2018-04-13 23:59:59"])
    ground_truth_dir: str | None = Field(
        None,
        description="Evaluate against these ground-truth CSVs (optional).",
        examples=["ground_truth/cadets_e3"],
    )

    @field_validator("inference_start", "inference_end")
    @classmethod
    def check_inference_times(cls, v: str) -> str:
        return _check_time(v)

    @field_validator("ground_truth_dir")
    @classmethod
    def check_ground_truth_dir(cls, v: str | None) -> str | None:
        return _project_path(v, must_be_dir=True)


class TrainRequest(DbRequest, TrainFields):
    @model_validator(mode="after")
    def check_ranges(self) -> "TrainRequest":
        _check_range(self.train_start, self.train_end, "train")
        return self


class PipelineRequest(DbRequest, TrainFields, InferenceFields):
    @model_validator(mode="after")
    def check_ranges(self) -> "PipelineRequest":
        _check_range(self.train_start, self.train_end, "train")
        _check_range(self.inference_start, self.inference_end, "inference")
        return self


class InferenceRequest(DbRequest, InferenceFields):
    model_id: str | None = Field(
        None, description="Model to use (default: the latest model trained on db_name)."
    )

    @model_validator(mode="after")
    def check_ranges(self) -> "InferenceRequest":
        _check_range(self.inference_start, self.inference_end, "inference")
        return self


class SysdigIngestRequest(BaseModel):
    db_name: str = Field(
        pattern=r"^[A-Za-z0-9_]+$",
        description="Sysdig database to append to (created if missing), e.g. sysdig_live.",
        examples=["sysdig_live"],
    )
    scap_dir: str = Field(
        description="Directory in the repo that sysdig writes *.scap* files to.",
        examples=["graphwatch_data_pipeline/sysdig/input/live"],
    )
    min_file_age_seconds: float = Field(
        60, ge=0, description="Skip files modified more recently (still being written)."
    )
    max_files: int | None = Field(None, ge=1, description="At most this many files per job.")
    reset: bool = Field(False, description="Drop the DB tables and ingestion state first.")
    run_inference: bool = Field(False, description="Run inference on the ingested time range.")
    model_id: str | None = Field(
        None, description="Model for the inference (default: latest trained on db_name)."
    )
    ground_truth_dir: str | None = Field(None, examples=["ground_truth/sysdig"])

    @field_validator("db_name")
    @classmethod
    def check_sysdig_db(cls, v: str) -> str:
        if stages.dataset_for_db(v) != "sysdig":
            raise ValueError("db_name must be a sysdig database (sysdig or sysdig_<name>)")
        return v

    @field_validator("scap_dir")
    @classmethod
    def check_scap_dir(cls, v: str) -> str:
        return _project_path(v, must_be_dir=True)  # type: ignore[return-value]

    @field_validator("ground_truth_dir")
    @classmethod
    def check_ground_truth_dir(cls, v: str | None) -> str | None:
        return _project_path(v, must_be_dir=True)


def _new_model_id(db_name: str) -> str:
    return f"{db_name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(2)}"


def _train_step(req: TrainRequest | PipelineRequest, model_id: str) -> list[str]:
    step = ["train.py", "--db-name", req.db_name, "--train-start", req.train_start,
            "--train-end", req.train_end, "--model-id", model_id]  # fmt: skip
    if req.experiment_config:
        step += ["--experiment-config", req.experiment_config]
    if req.force_reload:
        step.append("--force-reload")
    return step


def _infer_step(
    req: InferenceRequest | PipelineRequest, model_id: str | None, run_id: str
) -> list[str]:
    step = ["infer.py", "--db-name", req.db_name, "--inference-start", req.inference_start,
            "--inference-end", req.inference_end, "--run-id", run_id]  # fmt: skip
    if model_id:
        step += ["--model-id", model_id]
    if req.ground_truth_dir:
        step += ["--ground-truth-dir", req.ground_truth_dir]
    if req.force_reload:
        step.append("--force-reload")
    return step


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs/pipeline", status_code=202, dependencies=Auth, tags=["jobs"])
def submit_pipeline(req: PipelineRequest) -> dict[str, Any]:
    """Train a new model on the training range, then run inference on the inference range."""
    model_id = _new_model_id(req.db_name)
    run_id = f"{model_id}_inference"
    return jobs.submit(
        "pipeline",
        req.model_dump(),
        [_train_step(req, model_id), _infer_step(req, model_id, run_id)],
        model_id=model_id,
        run_id=run_id,
    )


@app.post("/jobs/train", status_code=202, dependencies=Auth, tags=["jobs"])
def submit_train(req: TrainRequest) -> dict[str, Any]:
    """Only pretrain a model; use it later with /jobs/inference."""
    model_id = _new_model_id(req.db_name)
    return jobs.submit("train", req.model_dump(), [_train_step(req, model_id)], model_id=model_id)


@app.post("/jobs/inference", status_code=202, dependencies=Auth, tags=["jobs"])
def submit_inference(req: InferenceRequest) -> dict[str, Any]:
    """Run inference with a pretrained model (by default the latest one for db_name)."""
    try:
        model_id = stages.resolve_model(model_id=req.model_id, db_name=req.db_name).path.name
    except FileNotFoundError as exc:
        # A model may still be training in the queue; it must exist when the job starts.
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    run_id = f"{model_id}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(2)}"
    return jobs.submit(
        "inference",
        req.model_dump(),
        [_infer_step(req, model_id, run_id)],
        model_id=model_id,
        run_id=run_id,
    )


@app.post("/jobs/sysdig-ingest", status_code=202, dependencies=Auth, tags=["jobs"])
def submit_sysdig_ingest(req: SysdigIngestRequest) -> dict[str, Any]:
    """Append new, complete sysdig capture files of scap_dir to db_name ("live" ingestion).

    Only files not ingested before are processed, so calling this periodically keeps the
    database up to date. With run_inference, the ingested time range is classified too.
    """
    step = ["graphwatch_data_pipeline/sysdig/ingest.py", "--scap-dir", req.scap_dir,
            "--db-name", req.db_name, "--min-file-age", str(req.min_file_age_seconds)]  # fmt: skip
    if req.max_files:
        step += ["--max-files", str(req.max_files)]
    if req.reset:
        step.append("--reset")
    extra: dict[str, Any] = {}
    if req.run_inference:
        try:
            model_id = stages.resolve_model(model_id=req.model_id, db_name=req.db_name).path.name
        except FileNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        run_id = f"{model_id}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(2)}"
        step += ["--run-inference", "--model-id", model_id, "--run-id", run_id]
        if req.ground_truth_dir:
            step += ["--ground-truth-dir", req.ground_truth_dir]
        extra = {"model_id": model_id, "run_id": run_id}
    return jobs.submit("sysdig-ingest", req.model_dump(), [step], **extra)


@app.get("/jobs", dependencies=Auth, tags=["jobs"])
def list_jobs() -> list[dict[str, Any]]:
    return jobs.all()


def _job_or_404(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id) if job_id.isalnum() else None
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job '{job_id}' not found")
    return job


@app.get("/jobs/{job_id}", dependencies=Auth, tags=["jobs"])
def get_job(job_id: str) -> dict[str, Any]:
    return _job_or_404(job_id)


@app.post("/jobs/{job_id}/cancel", dependencies=Auth, tags=["jobs"])
def cancel_job(job_id: str) -> dict[str, Any]:
    _job_or_404(job_id)
    return jobs.cancel(job_id)  # type: ignore[return-value]


@app.get("/jobs/{job_id}/log", dependencies=Auth, tags=["jobs"], response_class=PlainTextResponse)
def get_job_log(job_id: str, tail: Annotated[int, Query(ge=1, le=100_000)] = 200) -> str:
    _job_or_404(job_id)
    log = jobs.log_file(job_id)
    if not log.exists():
        return ""
    return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:])


def _result_file(job_id: str, name: str) -> Path:
    job = _job_or_404(job_id)
    if job["status"] != "succeeded" or not job.get("run_id"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Job has no inference result (yet)")
    path = stages.INFERENCE_DIR / job["run_id"] / name
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{name} not found for job '{job_id}'")
    return path


@app.get("/jobs/{job_id}/report", dependencies=Auth, tags=["results"])
def get_report(job_id: str) -> dict[str, Any]:
    return json.loads(_result_file(job_id, "report.json").read_text(encoding="utf-8"))


@app.get("/jobs/{job_id}/anomalies", dependencies=Auth, tags=["results"])
def get_anomalies(
    job_id: str,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 1000,
) -> dict[str, Any]:
    rows = json.loads(_result_file(job_id, "anomalies.json").read_text(encoding="utf-8"))
    return {"total": len(rows), "offset": offset, "items": rows[offset : offset + limit]}


@app.get("/jobs/{job_id}/detailed-report", dependencies=Auth, tags=["results"])
def get_detailed_report(job_id: str) -> FileResponse:
    return FileResponse(_result_file(job_id, "detailed_report.json"))


@app.get("/models", dependencies=Auth, tags=["models"])
def list_models(db_name: str | None = None) -> list[dict[str, Any]]:
    return [m for m in stages.list_models() if db_name in (None, m.get("db_name"))]


@app.get("/models/{model_id}", dependencies=Auth, tags=["models"])
def get_model(model_id: str) -> dict[str, Any]:
    for meta in stages.list_models():
        if meta["model_id"] == model_id:
            return meta
    raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model '{model_id}' not found")
