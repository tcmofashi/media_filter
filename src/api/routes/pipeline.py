"""Pipeline control endpoints for labeling -> training -> deployment -> download orchestration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from ruamel import yaml

from src.services.pipeline_jobs import pipeline_job_manager

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.local.yaml"

_YAML = yaml.YAML()


class TrainJobRequest(BaseModel):
    labels_path: str = "labels.json"
    val_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    num_frames: int = Field(default=8, ge=1)
    long_video_strategy: str = "expand"
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    output_dir: str = "checkpoints/frozen_clip"
    precision: str = "fp32"
    num_workers: int = Field(default=4, ge=1)
    prefetch_factor: int = Field(default=4, ge=1)
    persistent_workers: bool = True
    save_every: int = Field(default=1, ge=1)
    resume: Optional[str] = None
    config: Optional[str] = None
    deepspeed_config: Optional[str] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    clip_batch_size: int = Field(default=64, ge=1)
    max_frames: int = Field(default=32, ge=1)
    max_long_frames: int = Field(default=32, ge=1)
    min_long_frames: Optional[int] = None


class DownloadJobRequest(BaseModel):
    min_score: float = 7.0
    chat_batch_size: int = Field(default=150, ge=1)
    continuous: bool = True
    keep_below_threshold: bool = False
    target_root: Optional[str] = None
    cache_root: Optional[str] = None
    flat_links_root: Optional[str] = None
    state_path: Optional[str] = None
    db: Optional[str] = None
    session_dir: Optional[str] = None
    session_name: Optional[str] = None
    session_string: Optional[str] = None
    api_id: Optional[int] = None
    api_hash: Optional[str] = None
    history_limit: Optional[int] = None
    max_chats: Optional[int] = None
    log_every: Optional[int] = None
    message_concurrency: Optional[int] = None
    cache_max_items: Optional[int] = None
    chat_idle_seconds: Optional[float] = None
    round_idle_seconds: Optional[float] = None
    breadth_rounds: Optional[int] = None
    focus_top_chats: Optional[int] = None
    focus_min_scored: Optional[int] = None
    discover_chat_types: Optional[str] = None


class GlobalPipelineRequest(BaseModel):
    min_score: float = 7.0
    chat_batch_size: int = Field(default=150, ge=1)
    continuous: bool = True
    keep_below_threshold: bool = False
    run_bulk_infer: bool = True
    bulk_limit: Optional[int] = None
    force: bool = False
    progress_every: int = Field(default=100, ge=1)
    prune_below_threshold: bool = False
    skip_rebucket: bool = False
    dry_run: bool = False
    rebucket_mode: str = "symlink"
    skip_download: bool = False
    bucket_root: Optional[str] = None
    target_root: Optional[str] = None
    cache_root: Optional[str] = None
    flat_links_root: Optional[str] = None
    state_path: Optional[str] = None
    db: Optional[str] = None
    session_dir: Optional[str] = None
    session_name: Optional[str] = None
    session_string: Optional[str] = None
    api_id: Optional[int] = None
    api_hash: Optional[str] = None
    history_limit: Optional[int] = None
    max_chats: Optional[int] = None
    log_every: Optional[int] = None
    message_concurrency: Optional[int] = None
    cache_max_items: Optional[int] = None
    chat_idle_seconds: Optional[float] = None
    round_idle_seconds: Optional[float] = None
    breadth_rounds: Optional[int] = None
    focus_top_chats: Optional[int] = None
    focus_min_scored: Optional[int] = None
    discover_chat_types: Optional[str] = None


class DeployUpdate(BaseModel):
    checkpoint: str = Field(..., description="Model checkpoint path")


def _append_arg(args: list[str], key: str, value: Any) -> None:
    """Append a command-line argument when value is not None."""
    if value is None:
        return
    args.append(key)
    args.append(str(value))


def _append_flag(args: list[str], key: str, value: bool) -> None:
    """Append boolean flag style for argparse action store_true."""
    if value:
        args.append(key)


def _safe_yaml_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = _YAML.load(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _safe_yaml_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _YAML.dump(data, path)


def _resolve_project_path(value: Optional[str], default_name: Optional[str]) -> Optional[str]:
    if value is None or value == "":
        return default_name
    value_path = Path(value)
    if value_path.is_absolute():
        return str(value_path)
    if default_name is None:
        return str((PROJECT_ROOT / value_path).resolve())
    return str((PROJECT_ROOT / value_path).resolve())


def _get_effective_deploy_state() -> dict[str, Optional[str]]:
    public = _safe_yaml_load(PUBLIC_CONFIG_PATH)
    local = _safe_yaml_load(LOCAL_CONFIG_PATH)
    model_public = public.get("model", {})
    telegram_public = public.get("telegram", {})
    model_local = local.get("model", {})
    telegram_local = local.get("telegram", {})
    return {
        "model_checkpoint": str(
            model_local.get("checkpoint", model_public.get("checkpoint", "checkpoints/checkpoint_best.pt"))
        ),
        "telegram_checkpoint": str(
            telegram_local.get("checkpoint", telegram_public.get("checkpoint", "checkpoints/checkpoint_best.pt"))
        ),
    }


def _resolve_project_or_abs_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value == "":
        return None
    resolved = Path(value).expanduser()
    if resolved.is_absolute():
        return str(resolved)
    return str((PROJECT_ROOT / resolved).resolve())


@router.get("/jobs")
async def list_jobs(include_logs: bool = False) -> dict[str, list[dict[str, Any]]]:
    """List all pipeline jobs."""
    jobs = await pipeline_job_manager.list_jobs(include_log_tail=include_logs)
    return {"jobs": jobs}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, include_logs: bool = True) -> dict[str, Any]:
    """Get one job by id."""
    job = await pipeline_job_manager.get_job(job_id, include_log_tail=include_logs)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/jobs/{job_id}")
async def stop_job(job_id: str) -> dict[str, Any]:
    """Stop a job."""
    stopped = await pipeline_job_manager.stop_job(job_id)
    if not stopped:
        raise HTTPException(status_code=404, detail="Job not found")
    job = await pipeline_job_manager.get_job(job_id, include_log_tail=True)
    return {"status": "stopped", "job": job}


@router.post("/jobs/train")
async def start_train_job(payload: TrainJobRequest) -> dict[str, Any]:
    """Start a training task."""
    if (payload.score_min is not None) != (payload.score_max is not None):
        raise HTTPException(status_code=400, detail="score_min and score_max must be set together")

    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "train_frozen_clip.py")]
    _append_arg(command, "--labels_path", _resolve_project_or_abs_path(payload.labels_path))
    _append_arg(command, "--val_ratio", payload.val_ratio)
    _append_arg(command, "--num_frames", payload.num_frames)
    _append_arg(command, "--long_video_strategy", payload.long_video_strategy)
    _append_arg(command, "--epochs", payload.epochs)
    _append_arg(command, "--batch_size", payload.batch_size)
    _append_arg(command, "--learning_rate", payload.learning_rate)
    _append_arg(command, "--output_dir", _resolve_project_path(payload.output_dir, None))
    _append_arg(command, "--precision", payload.precision)
    _append_arg(command, "--num_workers", payload.num_workers)
    _append_arg(command, "--prefetch_factor", payload.prefetch_factor)
    if not payload.persistent_workers:
        _append_flag(command, "--no_persistent_workers", True)
    _append_arg(command, "--save_every", payload.save_every)
    _append_arg(command, "--deepspeed_config", _resolve_project_or_abs_path(payload.deepspeed_config))
    _append_arg(command, "--resume", _resolve_project_or_abs_path(payload.resume))
    _append_arg(command, "--config", _resolve_project_or_abs_path(payload.config))
    _append_arg(command, "--score_min", payload.score_min)
    _append_arg(command, "--score_max", payload.score_max)
    _append_arg(command, "--clip_batch_size", payload.clip_batch_size)
    _append_arg(command, "--max_frames", payload.max_frames)
    _append_arg(command, "--max_long_frames", payload.max_long_frames)
    _append_arg(command, "--min_long_frames", payload.min_long_frames)

    job = await pipeline_job_manager.start_job(
        "train",
        command=command,
        cwd=PROJECT_ROOT,
    )
    return {"job_id": job.job_id, "status": job.status}


@router.post("/jobs/download")
async def start_download_job(payload: DownloadJobRequest) -> dict[str, Any]:
    """Start Telegram gated download task only."""
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_tg_gated_download.py")]
    _append_arg(command, "--target-root", _resolve_project_or_abs_path(payload.target_root))
    _append_arg(command, "--cache-root", _resolve_project_or_abs_path(payload.cache_root))
    _append_arg(command, "--flat-links-root", _resolve_project_or_abs_path(payload.flat_links_root))
    _append_arg(command, "--state-path", _resolve_project_or_abs_path(payload.state_path))
    _append_arg(command, "--db", _resolve_project_or_abs_path(payload.db))
    _append_arg(command, "--session-dir", _resolve_project_or_abs_path(payload.session_dir))
    _append_arg(command, "--session-name", payload.session_name)
    _append_arg(command, "--session-string", payload.session_string)
    _append_arg(command, "--discover-chat-types", payload.discover_chat_types)
    _append_arg(command, "--chat-batch-size", payload.chat_batch_size)
    _append_arg(command, "--min-score", payload.min_score)
    _append_arg(command, "--history-limit", payload.history_limit)
    _append_arg(command, "--max-chats", payload.max_chats)
    _append_arg(command, "--log-every", payload.log_every)
    _append_arg(command, "--message-concurrency", payload.message_concurrency)
    _append_arg(command, "--cache-max-items", payload.cache_max_items)
    _append_arg(command, "--chat-idle-seconds", payload.chat_idle_seconds)
    _append_arg(command, "--round-idle-seconds", payload.round_idle_seconds)
    _append_arg(command, "--breadth-rounds", payload.breadth_rounds)
    _append_arg(command, "--focus-top-chats", payload.focus_top_chats)
    _append_arg(command, "--focus-min-scored", payload.focus_min_scored)
    _append_arg(command, "--api-id", payload.api_id)
    _append_arg(command, "--api-hash", payload.api_hash)
    _append_flag(command, "--keep-below-threshold", payload.keep_below_threshold)
    if payload.continuous:
        _append_flag(command, "--continuous", True)
    else:
        command.append("--no-continuous")

    job = await pipeline_job_manager.start_job(
        "tg_download",
        command=command,
        cwd=PROJECT_ROOT,
    )
    return {"job_id": job.job_id, "status": job.status}


@router.post("/jobs/global-pipeline")
async def start_global_pipeline_job(payload: GlobalPipelineRequest) -> dict[str, Any]:
    """Start full Telegram pipeline: download + optional bulk infer + rebucket + prune."""
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_telegram_global_pipeline.py")]
    _append_arg(command, "--target-root", _resolve_project_or_abs_path(payload.target_root))
    _append_arg(command, "--cache-root", _resolve_project_or_abs_path(payload.cache_root))
    _append_arg(command, "--flat-links-root", _resolve_project_or_abs_path(payload.flat_links_root))
    _append_arg(command, "--state-path", _resolve_project_or_abs_path(payload.state_path))
    _append_arg(command, "--db", _resolve_project_or_abs_path(payload.db))
    _append_arg(command, "--session-dir", _resolve_project_or_abs_path(payload.session_dir))
    _append_arg(command, "--session-name", payload.session_name)
    _append_arg(command, "--session-string", payload.session_string)
    _append_arg(command, "--discover-chat-types", payload.discover_chat_types)
    _append_arg(command, "--chat-batch-size", payload.chat_batch_size)
    _append_arg(command, "--min-score", payload.min_score)
    _append_arg(command, "--history-limit", payload.history_limit)
    _append_arg(command, "--max-chats", payload.max_chats)
    _append_arg(command, "--log-every", payload.log_every)
    _append_arg(command, "--message-concurrency", payload.message_concurrency)
    _append_arg(command, "--cache-max-items", payload.cache_max_items)
    _append_arg(command, "--chat-idle-seconds", payload.chat_idle_seconds)
    _append_arg(command, "--round-idle-seconds", payload.round_idle_seconds)
    _append_arg(command, "--breadth-rounds", payload.breadth_rounds)
    _append_arg(command, "--focus-top-chats", payload.focus_top_chats)
    _append_arg(command, "--focus-min-scored", payload.focus_min_scored)
    _append_arg(command, "--api-id", payload.api_id)
    _append_arg(command, "--api-hash", payload.api_hash)
    if payload.keep_below_threshold:
        _append_flag(command, "--keep-below-threshold", True)
    if payload.skip_download:
        _append_flag(command, "--skip-download", True)
    if payload.run_bulk_infer:
        _append_flag(command, "--run-bulk-infer", True)
    _append_arg(command, "--bulk-limit", payload.bulk_limit)
    _append_arg(command, "--progress-every", payload.progress_every)
    if payload.force:
        _append_flag(command, "--force", True)
    if payload.skip_rebucket:
        _append_flag(command, "--skip-rebucket", True)
    _append_arg(command, "--bucket-root", _resolve_project_or_abs_path(payload.bucket_root))
    _append_arg(command, "--rebucket-mode", payload.rebucket_mode)
    if payload.prune_below_threshold:
        _append_flag(command, "--prune-below-threshold", True)
    if payload.dry_run:
        _append_flag(command, "--dry-run", True)
    if payload.continuous:
        _append_flag(command, "--continuous", True)
    else:
        command.append("--no-continuous")

    job = await pipeline_job_manager.start_job(
        "global_pipeline",
        command=command,
        cwd=PROJECT_ROOT,
    )
    return {"job_id": job.job_id, "status": job.status}


@router.get("/deploy")
async def get_deploy_state() -> dict[str, str]:
    """Get current active checkpoint for model and telegram checkpoint."""
    state = _get_effective_deploy_state()
    return {
        "model_checkpoint": state["model_checkpoint"],
        "telegram_checkpoint": state["telegram_checkpoint"],
    }


@router.post("/deploy")
async def set_deploy_state(payload: DeployUpdate) -> dict[str, str]:
    """Set checkpoint in local override config for all pipeline stages."""
    checkpoint = _resolve_project_or_abs_path(payload.checkpoint)
    if checkpoint is None:
        raise HTTPException(status_code=400, detail="checkpoint is required")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=400, detail=f"Checkpoint not found: {payload.checkpoint}")

    local = _safe_yaml_load(LOCAL_CONFIG_PATH)
    model = dict(local.get("model", {}))
    telegram = dict(local.get("telegram", {}))
    model["checkpoint"] = str(checkpoint_path)
    telegram["checkpoint"] = str(checkpoint_path)
    local["model"] = model
    local["telegram"] = telegram
    _safe_yaml_dump(LOCAL_CONFIG_PATH, local)
    return {
        "message": "deploy updated",
        "model_checkpoint": str(checkpoint_path),
        "telegram_checkpoint": str(checkpoint_path),
    }
