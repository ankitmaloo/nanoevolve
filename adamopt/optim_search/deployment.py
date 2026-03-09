from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_quote(value: str) -> str:
    return json.dumps(value)


@dataclass
class RemoteTarget:
    name: str
    transport: str = "ssh"
    host: str = ""
    user: str | None = None
    port: int = 22
    identity_file: str | None = None
    ssh_options: list[str] = field(default_factory=list)
    remote_base_dir: str = "~/adamopt_remote"

    def ssh_destination(self) -> str:
        if not self.host:
            raise ValueError("host is required for ssh transport")
        if self.user:
            return f"{self.user}@{self.host}"
        return self.host


@dataclass
class DeploymentArtifacts:
    deployment_id: str
    candidate_id: str
    target: RemoteTarget
    deployment_dir: Path
    payload_dir: Path
    remote_dir: str
    remote_workspace_dir: str
    remote_trace_dir: str
    remote_log_path: str
    remote_status_path: str
    remote_pid_path: str
    remote_result_path: str
    manifest_path: Path
    launcher_stdout_path: Path
    launcher_stderr_path: Path
    fetched_status_path: Path
    fetched_log_tail_path: Path
    fetched_result_path: Path
    launch_script_path: Path
    command_script_path: Path


def _ssh_base_command(target: RemoteTarget) -> list[str]:
    command = ["ssh"]
    if target.port:
        command.extend(["-p", str(target.port)])
    if target.identity_file:
        command.extend(["-i", target.identity_file])
    command.extend(target.ssh_options)
    command.append(target.ssh_destination())
    return command


def _run_command(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def _run_shell(command: str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["/bin/sh", "-lc", command], cwd=cwd, text=True, capture_output=True, check=False)


def _copy_workspace(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def _make_payload(
    *,
    candidate_dir: Path,
    deployment_dir: Path,
    run_command: str,
    env: dict[str, str],
) -> tuple[Path, Path, Path]:
    payload_dir = deployment_dir / "payload"
    workspace_src = candidate_dir / "workspace"
    if not workspace_src.exists():
        raise ValueError(f"Candidate workspace does not exist: {workspace_src}")

    workspace_dest = payload_dir / "workspace"
    _copy_workspace(workspace_src, workspace_dest)

    command_script_path = payload_dir / "trace" / "run_command.sh"
    env_exports = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    command_script = f"""#!/usr/bin/env bash
set -euo pipefail
TRACE_DIR="$(cd "$(dirname "$0")" && pwd)"
export ADAMOPT_TRACE_DIR="$TRACE_DIR"
export ADAMOPT_STATUS_PATH="$TRACE_DIR/status.json"
export ADAMOPT_LOG_PATH="$TRACE_DIR/run.log"
export ADAMOPT_RESULT_PATH="$TRACE_DIR/result.json"
export ADAMOPT_CANDIDATE_ID={shlex.quote(candidate_dir.name)}
{env_exports}
cd "$(dirname "$0")/../workspace"
{run_command}
"""
    _write_text(command_script_path, command_script, executable=True)

    launch_script_path = payload_dir / "trace" / "launch.sh"
    launch_script = """#!/usr/bin/env bash
set -uo pipefail
TRACE_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$TRACE_DIR/../workspace"
STATUS_PATH="$TRACE_DIR/status.json"
LOG_PATH="$TRACE_DIR/run.log"
EXIT_CODE_PATH="$TRACE_DIR/exit_code.txt"
CMD_PATH="$TRACE_DIR/run_command.sh"
PID_PATH="$TRACE_DIR/pid.txt"

write_status() {
  python3 - "$STATUS_PATH" "$1" "$2" "$3" "$4" <<'PY'
import json
import sys
from datetime import datetime, UTC

path, state, phase, detail, pid = sys.argv[1:]
payload = {
    "state": state,
    "phase": phase,
    "detail": detail,
    "pid": pid,
    "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY
}

mkdir -p "$TRACE_DIR"
echo "$$" > "$PID_PATH"
write_status running start "" "$$"
(
  cd "$WORKSPACE_DIR"
  bash "$CMD_PATH"
) >"$LOG_PATH" 2>&1
exit_code=$?
printf "%s\n" "$exit_code" > "$EXIT_CODE_PATH"
if [ "$exit_code" -eq 0 ]; then
  write_status succeeded finished "exit_code=0" "$$"
else
  write_status failed finished "exit_code=$exit_code" "$$"
fi
exit "$exit_code"
"""
    _write_text(launch_script_path, launch_script, executable=True)
    return payload_dir, launch_script_path, command_script_path


def _ssh_stage_payload(payload_dir: Path, target: RemoteTarget, remote_dir: str) -> None:
    destination = shlex.quote(remote_dir)
    dest_host = shlex.quote(target.ssh_destination())
    mkdir_command = _ssh_base_command(target) + [f"mkdir -p {destination}"]
    mkdir_result = _run_command(mkdir_command)
    if mkdir_result.returncode != 0:
        raise RuntimeError(f"remote mkdir failed: {mkdir_result.stderr.strip()}")

    ssh_prefix = []
    if target.port:
        ssh_prefix.extend(["-p", str(target.port)])
    if target.identity_file:
        ssh_prefix.extend(["-i", target.identity_file])
    ssh_prefix.extend(target.ssh_options)
    remote_shell = " ".join(shlex.quote(part) for part in ssh_prefix + [target.ssh_destination(), f"tar -C {destination} -xf -"])
    tar_command = f"tar -C {shlex.quote(str(payload_dir))} -cf - . | ssh {remote_shell}"
    result = _run_shell(tar_command)
    if result.returncode != 0:
        raise RuntimeError(f"remote stage failed: {result.stderr.strip()}")


def _local_stage_payload(payload_dir: Path, remote_dir: Path) -> None:
    remote_dir.parent.mkdir(parents=True, exist_ok=True)
    if remote_dir.exists():
        shutil.rmtree(remote_dir)
    shutil.copytree(payload_dir, remote_dir)


def _ssh_launch(target: RemoteTarget, remote_launch_script: str, remote_trace_dir: str) -> subprocess.CompletedProcess[str]:
    command = (
        f"mkdir -p {shlex.quote(remote_trace_dir)} && "
        f"nohup bash {shlex.quote(remote_launch_script)} >/dev/null 2>&1 & "
        f"echo $!"
    )
    return _run_command(_ssh_base_command(target) + [command])


def _local_launch(remote_launch_script: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["bash", str(remote_launch_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )


def deploy_candidate_workspace(
    *,
    candidate_dir: Path,
    candidate_id: str,
    target: RemoteTarget,
    run_command: str,
    deployment_root: Path | None = None,
    deployment_label: str | None = None,
    env: dict[str, str] | None = None,
) -> DeploymentArtifacts:
    deployment_id = deployment_label or f"dep_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    deployment_dir = (deployment_root or (candidate_dir / "deployments")) / deployment_id
    deployment_dir.mkdir(parents=True, exist_ok=True)
    payload_dir, launch_script_path, command_script_path = _make_payload(
        candidate_dir=candidate_dir,
        deployment_dir=deployment_dir,
        run_command=run_command,
        env=env or {},
    )

    remote_dir = f"{target.remote_base_dir.rstrip('/')}/{candidate_id}/{deployment_id}"
    remote_workspace_dir = f"{remote_dir}/workspace"
    remote_trace_dir = f"{remote_dir}/trace"
    remote_log_path = f"{remote_trace_dir}/run.log"
    remote_status_path = f"{remote_trace_dir}/status.json"
    remote_pid_path = f"{remote_trace_dir}/pid.txt"
    remote_result_path = f"{remote_trace_dir}/result.json"
    manifest_path = deployment_dir / "manifest.json"
    launcher_stdout_path = deployment_dir / "launcher.stdout.txt"
    launcher_stderr_path = deployment_dir / "launcher.stderr.txt"
    fetched_status_path = deployment_dir / "fetched_status.json"
    fetched_log_tail_path = deployment_dir / "fetched_log_tail.txt"
    fetched_result_path = deployment_dir / "fetched_result.json"

    if target.transport == "ssh":
        _ssh_stage_payload(payload_dir, target, remote_dir)
        launch_result = _ssh_launch(target, f"{remote_trace_dir}/launch.sh", remote_trace_dir)
        launcher_stdout_path.write_text(launch_result.stdout or "")
        launcher_stderr_path.write_text(launch_result.stderr or "")
        if launch_result.returncode != 0:
            raise RuntimeError(f"remote launch failed: {launch_result.stderr.strip()}")
        remote_pid = (launch_result.stdout or "").strip()
    elif target.transport == "local":
        local_remote_dir = Path(os.path.expanduser(remote_dir)).resolve()
        _local_stage_payload(payload_dir, local_remote_dir)
        process = _local_launch(local_remote_dir / "trace" / "launch.sh")
        launcher_stdout_path.write_text("")
        launcher_stderr_path.write_text("")
        remote_pid = str(process.pid)
        remote_dir = str(local_remote_dir)
        remote_workspace_dir = str(local_remote_dir / "workspace")
        remote_trace_dir = str(local_remote_dir / "trace")
        remote_log_path = str(local_remote_dir / "trace" / "run.log")
        remote_status_path = str(local_remote_dir / "trace" / "status.json")
        remote_pid_path = str(local_remote_dir / "trace" / "pid.txt")
        remote_result_path = str(local_remote_dir / "trace" / "result.json")
    else:
        raise ValueError(f"Unsupported transport: {target.transport}")

    manifest = {
        "deployment_id": deployment_id,
        "candidate_id": candidate_id,
        "created_at": _utc_now(),
        "target": asdict(target),
        "run_command": run_command,
        "remote_dir": remote_dir,
        "remote_workspace_dir": remote_workspace_dir,
        "remote_trace_dir": remote_trace_dir,
        "remote_log_path": remote_log_path,
        "remote_status_path": remote_status_path,
        "remote_pid_path": remote_pid_path,
        "remote_result_path": remote_result_path,
        "remote_pid": remote_pid,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return DeploymentArtifacts(
        deployment_id=deployment_id,
        candidate_id=candidate_id,
        target=target,
        deployment_dir=deployment_dir,
        payload_dir=payload_dir,
        remote_dir=remote_dir,
        remote_workspace_dir=remote_workspace_dir,
        remote_trace_dir=remote_trace_dir,
        remote_log_path=remote_log_path,
        remote_status_path=remote_status_path,
        remote_pid_path=remote_pid_path,
        remote_result_path=remote_result_path,
        manifest_path=manifest_path,
        launcher_stdout_path=launcher_stdout_path,
        launcher_stderr_path=launcher_stderr_path,
        fetched_status_path=fetched_status_path,
        fetched_log_tail_path=fetched_log_tail_path,
        fetched_result_path=fetched_result_path,
        launch_script_path=launch_script_path,
        command_script_path=command_script_path,
    )


def _ssh_read_text(target: RemoteTarget, remote_path: str, missing_ok: bool = False) -> str:
    test = f"test -f {shlex.quote(remote_path)}"
    if missing_ok:
        command = f"{test} && cat {shlex.quote(remote_path)} || true"
    else:
        command = f"{test} && cat {shlex.quote(remote_path)}"
    result = _run_command(_ssh_base_command(target) + [command])
    if result.returncode != 0 and not missing_ok:
        raise RuntimeError(f"remote read failed for {remote_path}: {result.stderr.strip()}")
    return result.stdout or ""


def _ssh_tail_text(target: RemoteTarget, remote_path: str, lines: int) -> str:
    command = f"test -f {shlex.quote(remote_path)} && tail -n {int(lines)} {shlex.quote(remote_path)} || true"
    result = _run_command(_ssh_base_command(target) + [command])
    if result.returncode != 0:
        raise RuntimeError(f"remote tail failed for {remote_path}: {result.stderr.strip()}")
    return result.stdout or ""


def fetch_deployment_trace(deployment_dir: Path, *, tail_lines: int = 200) -> dict[str, object]:
    manifest_path = deployment_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Deployment manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    target = RemoteTarget(**manifest["target"])
    status_path = manifest["remote_status_path"]
    log_path = manifest["remote_log_path"]
    result_path = manifest.get("remote_result_path")

    if target.transport == "ssh":
        status_text = _ssh_read_text(target, status_path, missing_ok=True)
        log_tail = _ssh_tail_text(target, log_path, tail_lines)
        result_text = _ssh_read_text(target, result_path, missing_ok=True) if result_path else ""
    elif target.transport == "local":
        status_text = Path(status_path).read_text() if Path(status_path).exists() else ""
        log_file = Path(log_path)
        if log_file.exists():
            log_lines = log_file.read_text().splitlines()
            log_tail = "\n".join(log_lines[-tail_lines:])
        else:
            log_tail = ""
        result_file = Path(result_path) if result_path else None
        result_text = result_file.read_text() if result_file and result_file.exists() else ""
    else:
        raise ValueError(f"Unsupported transport: {target.transport}")

    fetched_status_path = deployment_dir / "fetched_status.json"
    fetched_log_tail_path = deployment_dir / "fetched_log_tail.txt"
    fetched_result_path = deployment_dir / "fetched_result.json"
    fetched_status_path.write_text(status_text or "{}")
    fetched_log_tail_path.write_text(log_tail)
    if result_text:
        fetched_result_path.write_text(result_text)

    status_payload = json.loads(status_text) if status_text.strip() else {"state": "unknown"}
    result_payload = json.loads(result_text) if result_text.strip() else None
    trace = {
        "deployment_id": manifest["deployment_id"],
        "candidate_id": manifest["candidate_id"],
        "target_name": target.name,
        "transport": target.transport,
        "remote_dir": manifest["remote_dir"],
        "status": status_payload,
        "fetched_status_path": str(fetched_status_path),
        "fetched_log_tail_path": str(fetched_log_tail_path),
        "fetched_result_path": str(fetched_result_path) if result_text else None,
        "result": result_payload,
        "tail_lines": tail_lines,
    }
    (deployment_dir / "trace_snapshot.json").write_text(json.dumps(trace, indent=2))
    return trace
