"""Red-team benchmark REST & SSE endpoints (/api/v1/benchmark/*)."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app_benchmark import runner as bench_runner
from app_benchmark.service import (
    build_dimension_csv_bytes,
    build_failure_replay_markdown,
    build_report_detail_payload,
    build_report_json_bytes,
    get_results_dir,
    list_reports_metadata,
    load_report_file,
    save_report_to_results_dir,
)
from app_copilot.service import sanitize_error_message
from core_tools.llm_client import LLMConfig

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


class BenchmarkRunRequest(BaseModel):
    models: List[str] = Field(..., min_length=1)
    modes: List[Literal["bare_llm", "harness_copilot"]] = Field(
        default_factory=lambda: ["bare_llm", "harness_copilot"],
        min_length=1,
    )
    mock_mode: bool = True
    include_private: bool = False


@dataclass
class _BenchmarkJob:
    job_id: str
    condition: threading.Condition = field(default_factory=threading.Condition)
    events: List[Dict[str, Any]] = field(default_factory=list)
    finished: bool = False

    def push(self, event: str, data: Dict[str, Any], *, is_terminal: bool = False) -> None:
        with self.condition:
            self.events.append({"event": event, "data": data})
            if is_terminal:
                self.finished = True
            self.condition.notify_all()


_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, _BenchmarkJob] = {}
_MAX_JOBS = 40


def _resolve_safe_report_path(filename: str) -> Path:
    """Resolve report filename inside get_results_dir() while preventing path traversal."""
    cleaned = (filename or "").strip()
    if (
        not cleaned
        or "/" in cleaned
        or "\\" in cleaned
        or ".." in cleaned
        or Path(cleaned).name != cleaned
        or not cleaned.endswith(".json")
    ):
        raise HTTPException(status_code=400, detail="非法的报告文件名。")
    results_dir = get_results_dir().resolve()
    target = (results_dir / cleaned).resolve()
    if not target.is_relative_to(results_dir):
        raise HTTPException(status_code=400, detail="非法的报告文件路径。")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"未找到评测报告文件：{cleaned}")
    return target


def _load_report_or_404(filename: str) -> tuple[Path, Dict[str, Any]]:
    path = _resolve_safe_report_path(filename)
    report = load_report_file(path)
    if report is None:
        raise HTTPException(status_code=422, detail=f"评测报告 {filename} 解析失败或格式非法。")
    return path, report


@router.get("/reports")
def list_reports() -> Dict[str, Any]:
    return {"reports": list_reports_metadata()}


@router.get("/reports/{filename}")
def get_report_detail(filename: str) -> Dict[str, Any]:
    _, report = _load_report_or_404(filename)
    return build_report_detail_payload(filename, report)


@router.post("/reports/import")
async def import_report(file: UploadFile = File(...)) -> Dict[str, str]:
    raw_bytes = await file.read()
    payload, err = bench_runner.validate_report_json_bytes(raw_bytes)
    if err or payload is None:
        raise HTTPException(status_code=400, detail=err or "外部报告 JSON 校验失败。")
    saved_path = save_report_to_results_dir(payload)
    return {"filename": saved_path.name}


@router.get("/reports/{filename}/export")
def export_report(
    filename: str,
    format: Literal["csv", "json", "markdown"] = Query(...),
) -> Response:
    _, report = _load_report_or_404(filename)
    stem = Path(filename).stem
    if format == "csv":
        content = build_dimension_csv_bytes(report.get("runs") or [])
        out_name = f"{stem}_dimensions.csv"
        media_type = "text/csv; charset=utf-8"
    elif format == "json":
        content = build_report_json_bytes(report)
        out_name = f"{stem}.json"
        media_type = "application/json; charset=utf-8"
    else:
        content = build_failure_replay_markdown(report).encode("utf-8")
        out_name = f"{stem}_failure_brief.md"
        media_type = "text/markdown; charset=utf-8"

    quoted = quote(out_name)
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=\"{quoted}\"; filename*=UTF-8''{quoted}",
        },
    )


def _execute_benchmark_job(job: _BenchmarkJob, req: BenchmarkRunRequest, output_dir: Path) -> None:
    def _on_progress(payload: Dict[str, Any]) -> None:
        job.push(
            "progress",
            {
                "completed": int(payload.get("done") or 0),
                "total": int(payload.get("total") or 0),
                "model": str(payload.get("model") or ""),
                "mode": str(payload.get("mode") or ""),
                "case_id": str(payload.get("case_id") or ""),
                "passed": payload.get("passed"),
                "status": "running",
            },
        )

    try:
        report = bench_runner.run_benchmark(
            models=req.models,
            modes=tuple(req.modes),
            mock_mode=req.mock_mode,
            include_private=req.include_private,
            save_report=True,
            output_dir=str(output_dir),
            progress_callback=_on_progress,
        )
        report_path = report.get("report_path")
        filename = Path(report_path).name if report_path else ""
        job.push(
            "done",
            {
                "filename": filename,
                "report_meta": {
                    "created_at": str(report.get("created_at") or "-"),
                    "mock_mode": bool(report.get("mock_mode", False)),
                    "case_count": int(report.get("case_count") or 0),
                    "include_private": bool(report.get("include_private", False)),
                },
            },
            is_terminal=True,
        )
    except Exception as exc:
        job.push(
            "error",
            {"message": sanitize_error_message(f"{type(exc).__name__}: {exc}")},
            is_terminal=True,
        )


@router.post("/runs")
def start_benchmark_run(req: BenchmarkRunRequest) -> Dict[str, str]:
    clean_models = [m.strip() for m in req.models if m and m.strip()]
    if not clean_models:
        fallback = LLMConfig.from_env().default_model
        if not fallback:
            raise HTTPException(status_code=400, detail="请至少指定一个评测模型。")
        clean_models = [fallback]

    normalized_req = BenchmarkRunRequest(
        models=clean_models,
        modes=req.modes,
        mock_mode=req.mock_mode,
        include_private=req.include_private,
    )
    job_id = uuid.uuid4().hex[:12]
    job = _BenchmarkJob(job_id=job_id)

    with _JOBS_LOCK:
        if len(_JOBS) >= _MAX_JOBS:
            finished_keys = [k for k, v in _JOBS.items() if v.finished]
            for old_key in finished_keys[: max(1, len(_JOBS) - _MAX_JOBS + 1)]:
                _JOBS.pop(old_key, None)
        _JOBS[job_id] = job

    output_dir = get_results_dir()
    thread = threading.Thread(
        target=_execute_benchmark_job,
        args=(job, normalized_req, output_dir),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


def _format_sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/runs/{job_id}/stream")
def stream_benchmark_run(job_id: str) -> StreamingResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未找到评测任务：{job_id}")

    def _event_generator() -> Generator[str, None, None]:
        cursor = 0
        while True:
            batch: List[Dict[str, Any]] = []
            finished_now = False
            with job.condition:
                while cursor >= len(job.events) and not job.finished:
                    notified = job.condition.wait(timeout=15.0)
                    if not notified and cursor >= len(job.events) and not job.finished:
                        yield ": ping\n\n"
                if cursor < len(job.events):
                    batch = job.events[cursor:]
                    cursor = len(job.events)
                finished_now = job.finished and cursor >= len(job.events)

            for item in batch:
                yield _format_sse(item["event"], item["data"])

            if finished_now:
                break

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
