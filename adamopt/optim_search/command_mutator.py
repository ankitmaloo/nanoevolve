from __future__ import annotations

import difflib
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .diff_engine import apply_search_replace_blocks
from .optimizer_targets import get_patch_target


# ---------------------------------------------------------------------------
# Provider registry — fully configurable at runtime
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """One LLM provider that can generate code mutations via CLI."""
    name: str
    command_template: str
    enabled: bool = True


# Shipped defaults.  Callers can add/remove/override via MutatorConfig.
_BUILTIN_PROVIDERS: list[ProviderConfig] = [
    ProviderConfig(name="codex",   command_template='codex -q "$(cat {prompt_file})"'),
    ProviderConfig(name="claude",  command_template="claude -p {prompt}"),
    ProviderConfig(name="copilot", command_template="cat {prompt_file} | gh copilot suggest -t code"),
]


@dataclass
class MutatorConfig:
    """Top-level configuration for the code-mutation subsystem.

    providers:
        Ordered list of provider configs.  Only providers with enabled=True
        are used by ensemble operations.  You can add custom providers
        (e.g. a local model behind llama.cpp) by appending to this list.

    diversity_review:
        Optional callback invoked after an ensemble generates patches from
        multiple providers.  Receives the EnsembleMutationResult and must
        return a DiversityReview.  If the review marks the ensemble as
        lacking diversity, the ensemble result carries that signal so the
        caller can re-prompt, discard duplicates, or take other action.

        Set to None to skip the review (default).
    """
    providers: list[ProviderConfig] = field(default_factory=lambda: list(_BUILTIN_PROVIDERS))
    diversity_review: DiversityReviewFn | None = None

    # -- helpers --

    @property
    def enabled_providers(self) -> list[ProviderConfig]:
        return [p for p in self.providers if p.enabled]

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    @property
    def enabled_provider_names(self) -> list[str]:
        return [p.name for p in self.enabled_providers]

    def get_template(self, provider_name: str) -> str:
        for p in self.providers:
            if p.name == provider_name:
                return p.command_template
        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Registered: {self.provider_names}"
        )

    def add_provider(self, name: str, command_template: str, *, enabled: bool = True) -> None:
        # Replace if exists, else append.
        for i, p in enumerate(self.providers):
            if p.name == name:
                self.providers[i] = ProviderConfig(name=name, command_template=command_template, enabled=enabled)
                return
        self.providers.append(ProviderConfig(name=name, command_template=command_template, enabled=enabled))


# Convenience: the default singleton used when no config is passed.
def default_mutator_config() -> MutatorConfig:
    return MutatorConfig()


# Legacy compat
def default_command_template(provider: str) -> str:
    return default_mutator_config().get_template(provider)


SUPPORTED_PROVIDERS = tuple(p.name for p in _BUILTIN_PROVIDERS)


# ---------------------------------------------------------------------------
# Diversity review — the hook for checking solution-space coverage
# ---------------------------------------------------------------------------

@dataclass
class PatchSummary:
    """Lightweight summary of one provider's patch, for the diversity reviewer."""
    provider: str
    candidate_id: str
    diff_text: str
    response_text: str
    patch_path: Path
    response_path: Path


@dataclass
class DiversityVerdict:
    """Per-pair or per-patch verdict from the diversity reviewer."""
    provider: str
    is_duplicate_of: str | None = None
    novelty_notes: str = ""


@dataclass
class DiversityReview:
    """Result of reviewing an ensemble for solution-space coverage.

    covers_solution_space:
        True if the reviewer considers the patches sufficiently diverse.
        False if multiple providers converged on the same idea.

    verdicts:
        Per-provider analysis.  If a provider's patch is flagged as a
        duplicate, is_duplicate_of names the provider it duplicates.

    summary:
        Free-text explanation of the review.  This is the place to note
        which axes of the solution space are covered and which are missing.

    suggested_reprompts:
        If coverage is poor, the reviewer can suggest alternative mutation
        instructions that would push toward unexplored parts of the space.
    """
    covers_solution_space: bool
    verdicts: list[DiversityVerdict] = field(default_factory=list)
    summary: str = ""
    suggested_reprompts: list[str] = field(default_factory=list)


# The signature for the user-supplied diversity review function.
DiversityReviewFn = Callable[["EnsembleMutationResult"], DiversityReview]


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Command execution helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single-provider mutation
# ---------------------------------------------------------------------------

def patch_nanochat_adamw(
    *,
    nanochat_root: Path,
    candidate_dir: Path,
    candidate_id: str,
    provider: str,
    instruction: str,
    command_template: str | None = None,
    scope: str = "adamw_math",
    config: MutatorConfig | None = None,
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

    cfg = config or default_mutator_config()
    template = command_template or cfg.get_template(provider)
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


# ---------------------------------------------------------------------------
# Ensemble mutation — multiple providers, same prompt
# ---------------------------------------------------------------------------

@dataclass
class EnsembleMutationResult:
    """Result from running the same mutation prompt through multiple providers."""
    candidate_id_prefix: str
    instruction: str
    scope: str
    provider_results: dict[str, CommandMutationArtifacts | None]
    provider_errors: dict[str, str]
    diversity_review: DiversityReview | None = None

    @property
    def successful_providers(self) -> list[str]:
        return [p for p, a in self.provider_results.items() if a is not None]

    @property
    def failed_providers(self) -> list[str]:
        return list(self.provider_errors)

    @property
    def patch_summaries(self) -> list[PatchSummary]:
        """Build PatchSummary list for the diversity reviewer."""
        summaries = []
        for provider, artifacts in self.provider_results.items():
            if artifacts is None:
                continue
            summaries.append(PatchSummary(
                provider=provider,
                candidate_id=artifacts.candidate_id,
                diff_text=artifacts.patch_path.read_text() if artifacts.patch_path.exists() else "",
                response_text=artifacts.response_path.read_text() if artifacts.response_path.exists() else "",
                patch_path=artifacts.patch_path,
                response_path=artifacts.response_path,
            ))
        return summaries


def ensemble_patch_nanochat(
    *,
    nanochat_root: Path,
    base_candidate_dir: Path,
    candidate_id_prefix: str,
    providers: list[str] | None = None,
    instruction: str,
    command_templates: dict[str, str] | None = None,
    scope: str = "adamw_math",
    config: MutatorConfig | None = None,
) -> EnsembleMutationResult:
    """Run the same mutation prompt through multiple providers.

    Each provider gets its own candidate subdirectory and workspace copy.
    After all providers run, the diversity_review callback (if configured)
    is invoked to check that the patches cover different parts of the
    solution space rather than converging on the same idea.
    """
    cfg = config or default_mutator_config()
    if providers is None:
        providers = cfg.enabled_provider_names

    command_templates = command_templates or {}
    results: dict[str, CommandMutationArtifacts | None] = {}
    errors: dict[str, str] = {}

    for provider in providers:
        candidate_id = f"{candidate_id_prefix}_{provider}"
        candidate_dir = base_candidate_dir / candidate_id
        try:
            artifacts = patch_nanochat_adamw(
                nanochat_root=nanochat_root,
                candidate_dir=candidate_dir,
                candidate_id=candidate_id,
                provider=provider,
                instruction=instruction,
                command_template=command_templates.get(provider),
                scope=scope,
                config=cfg,
            )
            results[provider] = artifacts
        except Exception as exc:
            results[provider] = None
            errors[provider] = str(exc)
            candidate_dir.mkdir(parents=True, exist_ok=True)
            (candidate_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}")

    ensemble_result = EnsembleMutationResult(
        candidate_id_prefix=candidate_id_prefix,
        instruction=instruction,
        scope=scope,
        provider_results=results,
        provider_errors=errors,
    )

    # --- Diversity review ---
    # If a review function is configured and we got patches from 2+ providers,
    # run the review and attach the result.  This is where you check that
    # the providers aren't all proposing the same change.
    review_fn = cfg.diversity_review
    if review_fn is not None and len(ensemble_result.successful_providers) >= 2:
        try:
            review = review_fn(ensemble_result)
            ensemble_result.diversity_review = review
        except Exception as exc:
            # Review failure is non-fatal — log it but keep the patches.
            ensemble_result.diversity_review = DiversityReview(
                covers_solution_space=False,
                summary=f"diversity review raised {type(exc).__name__}: {exc}",
            )

    # Write ensemble summary (including review if present)
    summary: dict[str, object] = {
        "candidate_id_prefix": candidate_id_prefix,
        "instruction": instruction,
        "scope": scope,
        "providers_attempted": providers,
        "providers_succeeded": ensemble_result.successful_providers,
        "providers_failed": list(errors),
        "errors": errors,
        "artifacts": {
            provider: {
                "candidate_id": a.candidate_id,
                "workspace_dir": str(a.workspace_dir),
                "patch_path": str(a.patch_path),
                "response_path": str(a.response_path),
            }
            for provider, a in results.items()
            if a is not None
        },
    }
    if ensemble_result.diversity_review is not None:
        dr = ensemble_result.diversity_review
        summary["diversity_review"] = {
            "covers_solution_space": dr.covers_solution_space,
            "summary": dr.summary,
            "verdicts": [
                {"provider": v.provider, "is_duplicate_of": v.is_duplicate_of, "novelty_notes": v.novelty_notes}
                for v in dr.verdicts
            ],
            "suggested_reprompts": dr.suggested_reprompts,
        }

    base_candidate_dir.mkdir(parents=True, exist_ok=True)
    (base_candidate_dir / f"{candidate_id_prefix}_ensemble.json").write_text(
        json.dumps(summary, indent=2)
    )

    return ensemble_result
