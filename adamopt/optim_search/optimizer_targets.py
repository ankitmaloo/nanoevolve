from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerPatchTarget:
    scope: str
    description: str
    target_relpath: str
    reference_relpaths: tuple[str, ...]
    required_blocks: tuple[tuple[str, str], ...]
    optional_blocks: tuple[tuple[str, str], ...] = ()
    constraints: tuple[str, ...] = ()


OPTIMIZER_PATCH_TARGETS: dict[str, OptimizerPatchTarget] = {
    "adamw_math": OptimizerPatchTarget(
        scope="adamw_math",
        description="Patch only AdamW math with shared single-GPU and distributed coverage.",
        target_relpath="nanochat/optim.py",
        reference_relpaths=("nanochat/gpt.py",),
        required_blocks=(
            ("nanochat/gpt.py", "setup_optimizer"),
            ("nanochat/optim.py", "adamw_step_fused"),
        ),
        optional_blocks=(
            ("nanochat/optim.py", "MuonAdamW"),
            ("nanochat/optim.py", "DistMuonAdamW"),
        ),
        constraints=(
            "Only modify the AdamW path (`adamw_step_fused`, `_step_adamw`, `_compute_adamw`, `_reduce_adamw`).",
            "Do not modify the Muon path.",
        ),
    ),
    "muon_math": OptimizerPatchTarget(
        scope="muon_math",
        description="Patch only matrix optimizer math with shared single-GPU and distributed coverage.",
        target_relpath="nanochat/optim.py",
        reference_relpaths=("nanochat/gpt.py",),
        required_blocks=(
            ("nanochat/gpt.py", "setup_optimizer"),
            ("nanochat/optim.py", "muon_step_fused"),
        ),
        optional_blocks=(
            ("nanochat/optim.py", "MuonAdamW"),
            ("nanochat/optim.py", "DistMuonAdamW"),
        ),
        constraints=(
            "Only modify the Muon path (`muon_step_fused`, `_step_muon`, `_compute_muon`).",
            "Do not modify the AdamW path.",
        ),
    ),
    "optimizer_routing": OptimizerPatchTarget(
        scope="optimizer_routing",
        description="Patch only parameter routing and optimizer group construction.",
        target_relpath="nanochat/gpt.py",
        reference_relpaths=("nanochat/optim.py",),
        required_blocks=(
            ("nanochat/gpt.py", "setup_optimizer"),
        ),
        optional_blocks=(
            ("nanochat/optim.py", "adamw_step_fused"),
            ("nanochat/optim.py", "muon_step_fused"),
        ),
        constraints=(
            "Only modify optimizer routing/group construction inside `setup_optimizer`.",
            "Do not modify training scripts.",
        ),
    ),
}


def get_patch_target(scope: str) -> OptimizerPatchTarget:
    if scope not in OPTIMIZER_PATCH_TARGETS:
        raise ValueError(f"Unsupported optimizer patch scope: {scope}")
    return OPTIMIZER_PATCH_TARGETS[scope]

