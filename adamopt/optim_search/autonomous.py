from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .command_mutator import patch_nanochat_adamw
from .config import AutonomousSearchConfig
from .deployment import RemoteTarget, deploy_candidate_workspace, fetch_deployment_trace
from .validation import validate_candidate_workspace


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


TERMINAL_STATES = {"succeeded", "failed"}


@dataclass
class AutonomousCandidateState:
    candidate_id: str
    index: int
    instruction: str
    status: str = "queued"
    provider: str = "codex"
    candidate_dir: str = ""
    deployment_dir: str | None = None
    deployment_id: str | None = None
    patch_path: str | None = None
    prompt_path: str | None = None
    response_path: str | None = None
    metadata_path: str | None = None
    validation_path: str | None = None
    validation_stdout_path: str | None = None
    validation_stderr_path: str | None = None
    validation_result: dict[str, Any] | None = None
    manifest_path: str | None = None
    fetched_status_path: str | None = None
    fetched_log_tail_path: str | None = None
    fetched_result_path: str | None = None
    remote_state: str | None = None
    remote_detail: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None


@dataclass
class AutonomousRunState:
    run_dir: str
    nanochat_root: str
    target: dict[str, Any]
    config: dict[str, Any]
    run_command_template: str
    candidates: list[AutonomousCandidateState] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)


class AutonomousStateStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.state_path = run_dir / "autonomous_state.json"
        self.events_path = run_dir / "events.jsonl"

    def exists(self) -> bool:
        return self.state_path.exists()

    def load(self) -> AutonomousRunState:
        payload = json.loads(self.state_path.read_text())
        payload["candidates"] = [AutonomousCandidateState(**candidate) for candidate in payload.get("candidates", [])]
        return AutonomousRunState(**payload)

    def save(self, state: AutonomousRunState) -> None:
        state.updated_at = _utc_now()
        payload = asdict(state)
        self.state_path.write_text(json.dumps(payload, indent=2))

    def append_event(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("timestamp", _utc_now())
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")


class AutonomousSearchController:
    def __init__(
        self,
        *,
        root_dir: Path,
        run_dir: Path,
        nanochat_root: Path,
        config: AutonomousSearchConfig,
        target: RemoteTarget,
        run_command_template: str,
    ) -> None:
        self.root_dir = root_dir
        self.run_dir = run_dir
        self.nanochat_root = nanochat_root
        self.config = config
        self.target = target
        self.run_command_template = run_command_template
        self.store = AutonomousStateStore(run_dir)
        self.state_lock = asyncio.Lock()
        self.patch_semaphore = asyncio.Semaphore(max(1, config.mutation_concurrency))
        self.validation_semaphore = asyncio.Semaphore(max(1, config.validation_concurrency))
        self.deploy_semaphore = asyncio.Semaphore(max(1, config.deployment_concurrency))
        self.poll_semaphore = asyncio.Semaphore(max(1, config.poll_concurrency))
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def _candidate_dir(self, candidate_id: str) -> Path:
        return self.run_dir / "candidates" / candidate_id

    def _terminal(self, candidate: AutonomousCandidateState) -> bool:
        return candidate.status in TERMINAL_STATES

    def _render_instruction(self, candidate_id: str, index: int) -> str:
        return (
            self.config.instruction_template
            .replace("{candidate_id}", candidate_id)
            .replace("{index}", str(index))
        )

    def _render_run_command(self, candidate: AutonomousCandidateState) -> str:
        return (
            self.run_command_template
            .replace("{candidate_id}", candidate.candidate_id)
            .replace("{index}", str(candidate.index))
        )

    async def _load_or_initialize(self) -> AutonomousRunState:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.store.exists():
            state = self.store.load()
            changed = False
            for candidate in state.candidates:
                if candidate.status == "patching":
                    candidate.status = "queued"
                    changed = True
                elif candidate.status == "validating":
                    candidate.status = "patched"
                    changed = True
                elif candidate.status == "deploying":
                    if candidate.deployment_dir and Path(candidate.deployment_dir).exists():
                        candidate.status = "running"
                    else:
                        candidate.status = "validated"
                    changed = True
            if changed:
                self.store.save(state)
                self.store.append_event({"event": "run_resumed", "normalized_transient_states": True})
            else:
                self.store.append_event({"event": "run_resumed", "normalized_transient_states": False})
            return state

        candidates = []
        for index in range(self.config.candidate_count):
            candidate_id = f"cand_{index + 1:04d}"
            candidates.append(
                AutonomousCandidateState(
                    candidate_id=candidate_id,
                    index=index,
                    instruction=self._render_instruction(candidate_id, index),
                    provider=self.config.provider,
                    candidate_dir=str(self._candidate_dir(candidate_id)),
                    updated_at=_utc_now(),
                )
            )
        state = AutonomousRunState(
            run_dir=str(self.run_dir),
            nanochat_root=str(self.nanochat_root),
            target=asdict(self.target),
            config=asdict(self.config),
            run_command_template=self.run_command_template,
            candidates=candidates,
        )
        self.store.save(state)
        self.store.append_event({"event": "run_initialized", "candidate_count": len(candidates)})
        return state

    async def _persist(self, state: AutonomousRunState, event: dict[str, Any] | None = None) -> None:
        async with self.state_lock:
            self.store.save(state)
            if event is not None:
                self.store.append_event(event)

    def _task_name(self, candidate: AutonomousCandidateState) -> str:
        return candidate.candidate_id

    def _schedule_ready(self, state: AutonomousRunState) -> None:
        for candidate in state.candidates:
            key = self._task_name(candidate)
            if key in self.tasks:
                continue
            if candidate.status == "queued":
                self.tasks[key] = asyncio.create_task(self._patch_candidate(state, candidate))
            elif candidate.status == "patched":
                self.tasks[key] = asyncio.create_task(self._validate_candidate(state, candidate))
            elif candidate.status == "validated":
                self.tasks[key] = asyncio.create_task(self._deploy_candidate(state, candidate))
            elif candidate.status == "running":
                self.tasks[key] = asyncio.create_task(self._watch_candidate(state, candidate))

    async def _patch_candidate(self, state: AutonomousRunState, candidate: AutonomousCandidateState) -> None:
        candidate.status = "patching"
        candidate.started_at = candidate.started_at or _utc_now()
        candidate.updated_at = _utc_now()
        await self._persist(state, {"event": "candidate_patching", "candidate_id": candidate.candidate_id})
        try:
            async with self.patch_semaphore:
                artifacts = await asyncio.to_thread(
                    patch_nanochat_adamw,
                    nanochat_root=self.nanochat_root,
                    candidate_dir=self._candidate_dir(candidate.candidate_id),
                    candidate_id=candidate.candidate_id,
                    provider=candidate.provider,
                    instruction=candidate.instruction,
                    command_template=self.config.command_template,
                    scope=self.config.scope,
                )
            candidate.status = "patched"
            candidate.patch_path = str(artifacts.patch_path)
            candidate.prompt_path = str(artifacts.prompt_path)
            candidate.response_path = str(artifacts.response_path)
            candidate.metadata_path = str(artifacts.metadata_path)
            candidate.error = None
            candidate.updated_at = _utc_now()
            await self._persist(
                state,
                {
                    "event": "candidate_patched",
                    "candidate_id": candidate.candidate_id,
                    "patch_path": candidate.patch_path,
                },
            )
        except Exception as exc:
            candidate.status = "failed"
            candidate.error = f"patch_failed: {exc}"
            candidate.updated_at = _utc_now()
            candidate.completed_at = _utc_now()
            await self._persist(state, {"event": "candidate_patch_failed", "candidate_id": candidate.candidate_id, "error": str(exc)})

    async def _validate_candidate(self, state: AutonomousRunState, candidate: AutonomousCandidateState) -> None:
        candidate.status = "validating"
        candidate.updated_at = _utc_now()
        await self._persist(state, {"event": "candidate_validating", "candidate_id": candidate.candidate_id})
        try:
            async with self.validation_semaphore:
                artifacts = await asyncio.to_thread(
                    validate_candidate_workspace,
                    candidate_dir=self._candidate_dir(candidate.candidate_id),
                    scope=self.config.scope,
                    python_executable=self.config.validation_python_executable,
                    disable_torch_compile=self.config.validation_disable_torch_compile,
                )
            candidate.validation_path = str(artifacts.summary_path)
            candidate.validation_stdout_path = str(artifacts.stdout_path)
            candidate.validation_stderr_path = str(artifacts.stderr_path)
            candidate.validation_result = artifacts.result
            candidate.updated_at = _utc_now()
            if artifacts.ok:
                candidate.status = "validated"
                candidate.error = None
                await self._persist(
                    state,
                    {
                        "event": "candidate_validated",
                        "candidate_id": candidate.candidate_id,
                        "validation_path": candidate.validation_path,
                    },
                )
            else:
                candidate.status = "failed"
                candidate.error = str(artifacts.result.get("error", "validation_failed"))
                candidate.completed_at = _utc_now()
                await self._persist(
                    state,
                    {
                        "event": "candidate_validation_failed",
                        "candidate_id": candidate.candidate_id,
                        "validation_path": candidate.validation_path,
                        "error": candidate.error,
                    },
                )
        except Exception as exc:
            candidate.status = "failed"
            candidate.error = f"validation_failed: {exc}"
            candidate.updated_at = _utc_now()
            candidate.completed_at = _utc_now()
            await self._persist(
                state,
                {"event": "candidate_validation_error", "candidate_id": candidate.candidate_id, "error": str(exc)},
            )

    async def _deploy_candidate(self, state: AutonomousRunState, candidate: AutonomousCandidateState) -> None:
        candidate.status = "deploying"
        candidate.updated_at = _utc_now()
        await self._persist(state, {"event": "candidate_deploying", "candidate_id": candidate.candidate_id})
        try:
            async with self.deploy_semaphore:
                artifacts = await asyncio.to_thread(
                    deploy_candidate_workspace,
                    candidate_dir=self._candidate_dir(candidate.candidate_id),
                    candidate_id=candidate.candidate_id,
                    target=self.target,
                    run_command=self._render_run_command(candidate),
                )
            candidate.status = "running"
            candidate.deployment_dir = str(artifacts.deployment_dir)
            candidate.deployment_id = artifacts.deployment_id
            candidate.manifest_path = str(artifacts.manifest_path)
            candidate.fetched_status_path = str(artifacts.fetched_status_path)
            candidate.fetched_log_tail_path = str(artifacts.fetched_log_tail_path)
            candidate.fetched_result_path = str(artifacts.fetched_result_path)
            candidate.remote_state = "running"
            candidate.error = None
            candidate.updated_at = _utc_now()
            await self._persist(
                state,
                {
                    "event": "candidate_deployed",
                    "candidate_id": candidate.candidate_id,
                    "deployment_id": candidate.deployment_id,
                },
            )
        except Exception as exc:
            candidate.status = "failed"
            candidate.error = f"deploy_failed: {exc}"
            candidate.updated_at = _utc_now()
            candidate.completed_at = _utc_now()
            await self._persist(state, {"event": "candidate_deploy_failed", "candidate_id": candidate.candidate_id, "error": str(exc)})

    async def _watch_candidate(self, state: AutonomousRunState, candidate: AutonomousCandidateState) -> None:
        deployment_dir = Path(candidate.deployment_dir or "")
        while True:
            try:
                async with self.poll_semaphore:
                    trace = await asyncio.to_thread(fetch_deployment_trace, deployment_dir, tail_lines=200)
                status = trace["status"]
                candidate.remote_state = status.get("state")
                candidate.remote_detail = status.get("detail")
                candidate.updated_at = _utc_now()
                if trace.get("fetched_status_path"):
                    candidate.fetched_status_path = str(trace["fetched_status_path"])
                if trace.get("fetched_log_tail_path"):
                    candidate.fetched_log_tail_path = str(trace["fetched_log_tail_path"])
                if trace.get("fetched_result_path"):
                    candidate.fetched_result_path = str(trace["fetched_result_path"])
                if trace.get("result") is not None:
                    candidate.result = trace["result"]
                if candidate.remote_state == "succeeded":
                    candidate.status = "succeeded"
                    candidate.completed_at = _utc_now()
                    await self._persist(
                        state,
                        {
                            "event": "candidate_succeeded",
                            "candidate_id": candidate.candidate_id,
                            "deployment_id": candidate.deployment_id,
                            "result_present": candidate.result is not None,
                        },
                    )
                    return
                if candidate.remote_state == "failed":
                    candidate.status = "failed"
                    candidate.error = candidate.remote_detail or "remote_failed"
                    candidate.completed_at = _utc_now()
                    await self._persist(
                        state,
                        {
                            "event": "candidate_failed",
                            "candidate_id": candidate.candidate_id,
                            "deployment_id": candidate.deployment_id,
                            "detail": candidate.remote_detail,
                        },
                    )
                    return
                await self._persist(
                    state,
                    {
                        "event": "candidate_polled",
                        "candidate_id": candidate.candidate_id,
                        "deployment_id": candidate.deployment_id,
                        "remote_state": candidate.remote_state,
                    },
                )
            except Exception as exc:
                candidate.updated_at = _utc_now()
                candidate.error = f"poll_error: {exc}"
                await self._persist(
                    state,
                    {"event": "candidate_poll_error", "candidate_id": candidate.candidate_id, "error": str(exc)},
                )
            await asyncio.sleep(self.config.poll_interval_s)

    def _cleanup_tasks(self) -> None:
        finished = [key for key, task in self.tasks.items() if task.done()]
        for key in finished:
            task = self.tasks.pop(key)
            task.result()

    def _all_terminal(self, state: AutonomousRunState) -> bool:
        return all(self._terminal(candidate) for candidate in state.candidates)

    async def run(self) -> dict[str, Any]:
        state = await self._load_or_initialize()
        self._schedule_ready(state)

        while self.tasks or not self._all_terminal(state):
            if not self.tasks:
                self._schedule_ready(state)
                if not self.tasks:
                    await asyncio.sleep(self.config.poll_interval_s)
                    continue
            done, _ = await asyncio.wait(self.tasks.values(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            self._cleanup_tasks()
            self._schedule_ready(state)

        summary = {
            "run_dir": str(self.run_dir),
            "candidate_count": len(state.candidates),
            "succeeded": [candidate.candidate_id for candidate in state.candidates if candidate.status == "succeeded"],
            "failed": [candidate.candidate_id for candidate in state.candidates if candidate.status == "failed"],
            "result_candidates": [
                {"candidate_id": candidate.candidate_id, "result": candidate.result}
                for candidate in state.candidates
                if candidate.result is not None
            ],
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        await self._persist(state, {"event": "run_completed", "summary": summary})
        return summary
