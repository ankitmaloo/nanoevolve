from .autonomous import AutonomousSearchController
from .config import AutonomousSearchConfig, EvaluationConfig, SearchConfig
from .command_mutator import patch_nanochat_adamw
from .deployment import RemoteTarget, deploy_candidate_workspace, fetch_deployment_trace
from .eval_candidate import ToyNanoChatBackend, compare_baseline_candidate
from .spec import MatrixOptimizerSpec
from .tournament import OptimizerTournament
from .validation import validate_candidate_workspace

__all__ = [
    "AutonomousSearchConfig",
    "AutonomousSearchController",
    "EvaluationConfig",
    "SearchConfig",
    "patch_nanochat_adamw",
    "RemoteTarget",
    "deploy_candidate_workspace",
    "fetch_deployment_trace",
    "validate_candidate_workspace",
    "ToyNanoChatBackend",
    "compare_baseline_candidate",
    "MatrixOptimizerSpec",
    "OptimizerTournament",
]
