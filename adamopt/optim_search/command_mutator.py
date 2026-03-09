from __future__ import annotations

import difflib
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .diff_engine import apply_search_replace_blocks
from .optimizer_targets import get_patch_target


DEFAULT_COMMAND_TEMPLATES = {
    "codex": "codex exec {prompt}",
    "claude": "claude -p {prompt}",
}


@dataclass
class CommandMutationArtifacts:
    candidate_id: str
    provider: str
    command: list[str]
    prompt_path: Path
    response_path: Path
    patch_path: Path
    metadata_path: Path
    workspace_dir: Path
    target_file: Path
    changed_files: list[str] = field(default_factory=list)


def default_command_template(provider: str) -> str:
    if provider not in DEFAULT_COMMAND_TEMPLATES:
        raise ValueError(f"Unsupported provider: {provider}")
    return DEFAULT_COMMAND_TEMPLATES[provider]


def _extract_block(text: str, anchor: str) -> str:
    lines = text.splitlines()
    start = None
    start_indent = 0
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if (
            stripped.startswith(f"def {anchor}(")
            or stripped.startswith(f"class {anchor}(")
            or stripped.startswith(f"class {anchor}:")
        ):
            start = index
            start_indent = len(line) - len(stripped)
            break
    if start is None:
        raise ValueError(f"Unable to find block for {anchor!r}")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped and not stripped.startswith("#") and (stripped.startswith("def ") or stripped.startswith("class ")) and indent <= start_indent:
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _extract_block_if_present(text: str, anchor: str) -> str | None:
    try:
        return _extract_block(text, anchor)
    except ValueError:
        return None


def build_adamw_mutation_prompt(
    nanochat_root: Path,
    *,
    instruction: str,
    scope: str = "adamw_math",
) -> str:
    patch_target = get_patch_target(scope)

    required_sections: list[str] = []
    for relpath, anchor in patch_target.required_blocks:
        text = (nanochat_root / relpath).read_text()
        block = _extract_block(text, anchor)
        required_sections.append(f"Required reference from `{relpath}`:\n{block}")

    optional_sections: list[str] = []
    for relpath, anchor in patch_target.optional_blocks:
        text = (nanochat_root / relpath).read_text()
        block = _extract_block_if_present(text, anchor)
        if block:
            optional_sections.append(f"Optional reference from `{relpath}`:\n{block}")

    prompt = f"""You are mutating a constrained NanoChat optimizer patch scope.

Target repository root:
{nanochat_root}

Constraints:
- Only modify `{patch_target.target_relpath}`.
- Scope: `{patch_target.scope}`.
- {patch_target.description}
- {' '.join(patch_target.constraints)}
- Return only SEARCH/REPLACE blocks.
- No markdown fences.
- Keep the code runnable.
- Avoid training-script changes unless absolutely necessary.

Mutation request:
{instruction}

{"\n\n".join(required_sections)}
{"\n\n" + "\n\n".join(optional_sections) if optional_sections else ""}
"""
    return prompt


def _render_command(template: str, *, prompt: str, prompt_file: Path, workspace_dir: Path, candidate_id: str) -> list[str]:
    rendered = template.format(
        prompt=prompt,
        prompt_file=str(prompt_file),
        workspace_dir=str(workspace_dir),
        candidate_id=candidate_id,
    )
    return shlex.split(rendered)


def _copy_nanochat_workspace(nanochat_root: Path, workspace_dir: Path) -> None:
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    shutil.copytree(
        nanochat_root,
        workspace_dir,
        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", "runs", "*.pyc"),
    )


def patch_nanochat_adamw(
    *,
    nanochat_root: Path,
    candidate_dir: Path,
    candidate_id: str,
    provider: str,
    instruction: str,
    command_template: str | None = None,
    scope: str = "adamw_math",
) -> CommandMutationArtifacts:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = candidate_dir / "workspace"
    _copy_nanochat_workspace(nanochat_root, workspace_dir)
    patch_target = get_patch_target(scope)

    prompt = build_adamw_mutation_prompt(nanochat_root, instruction=instruction, scope=scope)
    prompt_path = candidate_dir / "prompt.txt"
    response_path = candidate_dir / "response.txt"
    patch_path = candidate_dir / "mutation.diff"
    metadata_path = candidate_dir / "metadata.json"
    prompt_path.write_text(prompt)

    template = command_template or default_command_template(provider)
    command = _render_command(template, prompt=prompt, prompt_file=prompt_path, workspace_dir=workspace_dir, candidate_id=candidate_id)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    raw_response = (result.stdout or "").strip()
    if result.stderr:
        raw_response = f"{raw_response}\n\n# STDERR\n{result.stderr.strip()}".strip()
    response_path.write_text(raw_response)
    if result.returncode != 0:
        raise RuntimeError(f"{provider} command failed with exit code {result.returncode}: {result.stderr.strip()}")

    target_file = workspace_dir / patch_target.target_relpath
    original_source = target_file.read_text()
    patched_source, stats = apply_search_replace_blocks(original_source, result.stdout)
    target_file.write_text(patched_source)

    unified_diff = "".join(
        difflib.unified_diff(
            original_source.splitlines(keepends=True),
            patched_source.splitlines(keepends=True),
            fromfile=f"a/{patch_target.target_relpath}",
            tofile=f"b/{patch_target.target_relpath}",
        )
    )
    patch_path.write_text(unified_diff)

    metadata_path.write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "provider": provider,
                "scope": scope,
                "command_template": template,
                "command": command,
                "target_relpath": patch_target.target_relpath,
                "workspace_dir": str(workspace_dir),
                "patch_path": str(patch_path),
                "prompt_path": str(prompt_path),
                "response_path": str(response_path),
                "apply_stats": stats,
            },
            indent=2,
        )
    )
    return CommandMutationArtifacts(
        candidate_id=candidate_id,
        provider=provider,
        command=command,
        prompt_path=prompt_path,
        response_path=response_path,
        patch_path=patch_path,
        metadata_path=metadata_path,
        workspace_dir=workspace_dir,
        target_file=target_file,
        changed_files=[patch_target.target_relpath],
    )
