"""Job queue: runs train.py / infer.py as subprocesses, one job at a time (one GPU).

Each job is persisted as data/api/jobs/<id>.json with its log next to it, so job state
survives API restarts. Jobs that were running when the API stopped are marked failed.
"""

import json
import logging
import queue
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED = "queued", "running", "succeeded", "failed", "cancelled"
FINISHED = {SUCCEEDED, FAILED, CANCELLED}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, jobs_dir: Path, project_root: Path) -> None:
        self.jobs_dir = jobs_dir
        self.project_root = project_root
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._current: tuple[str, subprocess.Popen] | None = None
        self._recover()
        self._worker = threading.Thread(target=self._work, name="job-worker", daemon=True)
        self._worker.start()

    # --- persistence -------------------------------------------------------------------

    def _job_file(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def log_file(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.log"

    def _save(self, job: dict[str, Any]) -> None:
        tmp = self._job_file(job["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        tmp.replace(self._job_file(job["id"]))

    def get(self, job_id: str) -> dict[str, Any] | None:
        path = self._job_file(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def all(self) -> list[dict[str, Any]]:
        jobs = [json.loads(p.read_text(encoding="utf-8")) for p in self.jobs_dir.glob("*.json")]
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)

    def _update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            assert job is not None
            job.update(fields)
            self._save(job)
            return job

    def _recover(self) -> None:
        for job in sorted(self.all(), key=lambda j: j["created_at"]):
            if job["status"] == RUNNING:
                job.update(status=FAILED, finished_at=_now(), error="Interrupted by API restart")
                self._save(job)
            elif job["status"] == QUEUED:
                self._queue.put(job["id"])

    # --- public API --------------------------------------------------------------------

    def submit(
        self, job_type: str, params: dict[str, Any], steps: list[list[str]], **extra: Any
    ) -> dict[str, Any]:
        """Queue a job made of CLI steps (argument lists run with the project's python)."""
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "type": job_type,
            "status": QUEUED,
            "params": params,
            "steps": steps,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            **extra,
        }
        with self._lock:
            self._save(job)
        self._queue.put(job_id)
        return job

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None or job["status"] in FINISHED:
            return job
        if job["status"] == QUEUED:
            return self._update(job_id, status=CANCELLED, finished_at=_now())
        with self._lock:
            current = self._current
        if current and current[0] == job_id:
            current[1].terminate()
        return self._update(job_id, status=CANCELLED, finished_at=_now())

    # --- worker ------------------------------------------------------------------------

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None or job["status"] != QUEUED:
                continue
            try:
                self._run(job)
            except Exception as exc:  # keep the worker alive whatever happens
                logger.exception("Job %s crashed", job_id)
                self._update(job_id, status=FAILED, finished_at=_now(), error=repr(exc))

    def _run(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        self._update(job_id, status=RUNNING, started_at=_now())
        with self.log_file(job_id).open("a", encoding="utf-8") as log:
            for step in job["steps"]:
                cmd = [sys.executable, *step]
                log.write(f"$ {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(
                    cmd, cwd=self.project_root, stdout=log, stderr=subprocess.STDOUT
                )
                with self._lock:
                    self._current = (job_id, proc)
                returncode = proc.wait()
                with self._lock:
                    self._current = None
                if self.get(job_id)["status"] == CANCELLED:  # type: ignore[index]
                    return
                if returncode != 0:
                    self._update(
                        job_id,
                        status=FAILED,
                        finished_at=_now(),
                        error=f"'{step[0]}' exited with code {returncode}, see the job log",
                    )
                    return
        self._update(job_id, status=SUCCEEDED, finished_at=_now())
