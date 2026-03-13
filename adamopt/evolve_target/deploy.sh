#!/bin/bash
# deploy.sh — Deploy a validated optimizer spec to nanochat.
#
# Usage: bash deploy.sh <winner_spec_file>
#
# This is the final step in the pipeline:
#   alphaevolve → adamopt (validate) → nanochat (deploy)
#
# It takes the winning optimizer_spec.py, extracts the JSON spec,
# and writes it as a config file that nanochat's training script can load.

set -euo pipefail

WINNER_FILE="${1:-}"
if [ -z "$WINNER_FILE" ]; then
    echo "Usage: deploy.sh <winner_spec_file>"
    echo "  winner_spec_file: Path to the winning optimizer_spec.py from alphaevolve"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NANOCHAT_DIR="$REPO_ROOT/nanochat"
DEPLOY_DIR="$NANOCHAT_DIR/optimizer_configs"

# Extract spec JSON from the winning file
SPEC_JSON=$(python3 "$WINNER_FILE")

# Validate one more time before deploying
python3 -c "
import sys, json
sys.path.insert(0, '$REPO_ROOT')
from adamopt.optim_search.spec import MatrixOptimizerSpec
spec = MatrixOptimizerSpec.from_dict(json.loads('''$SPEC_JSON'''))
spec.validate()
print(f'Validated: {spec.name}')
print(f'  momentum={spec.momentum}, nesterov={spec.nesterov}')
print(f'  trust_ratio={spec.trust_ratio.mode}, clip={spec.clip.mode}')
print(f'  decay={spec.decay.mode}, wd={spec.decay.weight_decay}')
print(f'  stateful_control={spec.stateful_control.enabled}')
"

# Write the spec config to nanochat
mkdir -p "$DEPLOY_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEPLOY_PATH="$DEPLOY_DIR/evolved_spec_${TIMESTAMP}.json"

echo "$SPEC_JSON" | python3 -m json.tool > "$DEPLOY_PATH"

echo ""
echo "Deployed to: $DEPLOY_PATH"
echo ""
echo "To use in training, load this config in nanochat's optimizer setup."
echo "The config can be consumed by adamopt/optim_search/candidate_optimizer.py"
