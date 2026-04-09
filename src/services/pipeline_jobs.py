"""Long-running pipeline job orchestration for training and Telegram workflows."""

from __future__ import annotations

import asyncio
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from src.logger import get_logger

logger = get_logger(__name__)

_SENSITIVE_CMD_ARGS = {"--api-hash", "--session-string"}


def _sanitize_command(command: list[str]) -> list[str]:
    """Hide sensitive argument values when exposing command lines."""
    redacted: list[str] = []
    i = 0
    while i < len(command):
        value = command[i]
        if value in _SENSITIVE_CMD_ARGS and i + 1 < len(command):
            redacted.extend((value, "***REDACTED***"))
            i += 2
            continue
        redacted.append(value)
        i += 1
    return redacted


@dataclass
class PipelineJob:
    """Runtime state for a single orchestration job."""

    job_id: str
    job_type: str
    command: list[str]
    cwd: Path
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    status: str = "queued"
    pid: Optional[int] = None
    return_code: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    log_file: Optional[str] = None
    _log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    _stdout_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _stderr_task: Optional[asyncio.Task] = field(default=None, repr=False)

    def to_dict(self, include_log_tail: bool = False) -> dict[str, Any]:
        """Return serializable job snapshot."""
        payload = {
            "id": self.job_id,
            "type": self.job_type,
            "status": self.status,
            "command": _sanitize_command(self.command),
            "cwd": str(self.cwd),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "pid": self.pid,
            "return_code": self.return_code,
            "error": self.error,
            "log_file": self.log_file,
        }
        if include_log_tail:
            payload["log_tail"] = list(self._log_tail)
        return payload

    def append_log(self, line: str) -> None:
        """Append one line to tail logs with UTC timestamp."""
        self._log_tail.append(line)
        self.updated_at = datetime.utcnow().isoformat() + "Z"


class PipelineJobManager:
    """Manage background subprocess jobs for web-triggered pipeline workflows."""

    def __init__(self, log_dir: Path) -> None:
        self._jobs: dict[str, PipelineJob] = {}
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)

    async def list_jobs(self, include_log_tail: bool = False) -> list[dict[str, Any]]:
        """Return all jobs ordered by created time descending."""
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return [job.to_dict(include_log_tail=include_log_tail) for job in jobs]

    async def get_job(self, job_id: str, include_log_tail: bool = False) -> Optional[dict[str, Any]]:
        """Return one job snapshot."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        return job.to_dict(include_log_tail=include_log_tail)

    async def start_job(
        self,
        job_type: str,
        command: list[str],
        cwd: Path,
        env: Optional[dict[str, str]] = None,
    ) -> PipelineJob:
        """Create and run a background subprocess."""
        if not command:
            raise ValueError("command must not be empty")

        job_id = uuid4().hex
        job = PipelineJob(
            job_id=job_id,
            job_type=job_type,
            command=command,
            cwd=cwd,
            log_file=str(self._log_dir / f"{job_id}.log"),
        )
        async with self._lock:
            self._jobs[job_id] = job

        job_loop_task = asyncio.create_task(self._run_job(job, env=env))
        self._tasks[job_id] = job_loop_task
        job.append_log("job queued")
        return job

    async def stop_job(self, job_id: str) -> bool:
        """Stop one job by id."""
        process = self._procs.get(job_id)
        job = self._jobs.get(job_id)
        if job is None:
            return False

        if process is not None and process.returncode is None:
            job.status = "stopping"
            job.append_log("job stop requested")
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            job.status = "stopped"
            job.return_code = process.returncode
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            if process.returncode == 0:
                job.append_log("stopped (requested)")
            else:
                job.append_log(f"stopped (requested, code={process.returncode})")
            self._procs.pop(job_id, None)
            return True

        job.status = "stopped"
        job.completed_at = datetime.utcnow().isoformat() + "Z"
        return True

    async def shutdown(self) -> None:
        """Stop all running jobs. Called when API server exits."""
        for job_id in list(self._jobs.keys()):
            await self.stop_job(job_id)
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
        self._tasks.clear()

    def _build_env(self, env: Optional[dict[str, str]]) -> dict[str, str]:
        runtime_env = os.environ.copy()
        runtime_env["PYTHONUNBUFFERED"] = "1"
        if env:
            runtime_env.update(env)
        return runtime_env

    async def _run_job(self, job: PipelineJob, env: Optional[dict[str, str]]) -> None:
        """Internal routine that runs subprocess and persists logs."""
        process: Optional[asyncio.subprocess.Process] = None
        log_path = Path(job.log_file) if job.log_file else None

        try:
            process = await asyncio.create_subprocess_exec(
                *job.command,
                cwd=str(job.cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(env),
            )
            job.pid = process.pid
            job.status = "running"
            job.started_at = datetime.utcnow().isoformat() + "Z"
            job.updated_at = job.started_at
            self._procs[job.job_id] = process
            if log_path:
                safe_command = " ".join(_sanitize_command(job.command))
                job.append_log(f"started: {safe_command}")
                job.append_log(f"job id={job.job_id} pid={job.pid} cwd={job.cwd}")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                _append_to_log_file(log_path, job._log_tail[-1])
            stdout_reader = asyncio.create_task(self._drain_stream(job, process.stdout, log_path, "stdout"))
            stderr_reader = asyncio.create_task(self._drain_stream(job, process.stderr, log_path, "stderr"))
            job._stdout_task = stdout_reader
            job._stderr_task = stderr_reader
            return_code = await process.wait()
            await stdout_reader
            await stderr_reader
            job.return_code = return_code
            if job.status not in {"stopped", "cancelled"}:
                self._log_process_exit(job, return_code)
        except asyncio.CancelledError:
            if process is not None:
                process.terminate()
                try:
                    await process.wait()
                except Exception:
                    process.kill()
            job.status = "cancelled"
            job.error = "cancelled"
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            logger.exception("pipeline job failed: %s", exc)
        finally:
            if log_path:
                _append_to_log_file(log_path, f"final status={job.status}, return_code={job.return_code}")
            self._procs.pop(job.job_id, None)
            self._tasks.pop(job.job_id, None)
            job.updated_at = datetime.utcnow().isoformat() + "Z"

    def _log_process_exit(self, job: PipelineJob, return_code: int) -> None:
        """Record final process status."""
        job.completed_at = datetime.utcnow().isoformat() + "Z"
        if return_code == 0:
            job.status = "completed"
            job.append_log(f"completed successfully (code={return_code})")
        else:
            job.status = "failed"
            job.append_log(f"failed with exit code={return_code}")
            job.error = f"Process exited with code {return_code}"

    async def _drain_stream(
        self,
        job: PipelineJob,
        stream: asyncio.StreamReader | None,
        log_path: Optional[Path],
        stream_name: str,
    ) -> None:
        """Read process stdout/stderr lines and append to tail + file."""
        if stream is None:
            return
        while True:
            data = await stream.readline()
            if not data:
                break
            line = data.decode("utf-8", errors="replace").rstrip("\n")
            if line == "":
                continue
            text = f"[{stream_name}] {line}"
            job.append_log(text)
            if log_path is not None:
                _append_to_log_file(log_path, text)


def _append_to_log_file(path: Path, line: str) -> None:
    """Append one log line to job file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}Z {line}\n")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
pipeline_job_manager = PipelineJobManager(PROJECT_ROOT / "data" / "pipeline_jobs")
