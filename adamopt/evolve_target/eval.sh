#!/bin/bash
# eval.sh — Evaluate a candidate optimizer spec for alphaevolve.
#
# Usage: bash eval.sh <candidate_file>
#
# Pipeline:
#   1. Run candidate file → get spec JSON
#   2. Validate spec locally (fast, no GPU)
#   3. Optionally run 20-step GPU eval via Modal (if --gpu flag or GPU_EVAL=1)
#   4. Print JSON result to stdout
#
# The JSON output follows the alphaevolve generic evaluator contract:
# {"valid": bool, "aggregate_score": float, "metrics": {...}, "failure_reasons": [...]}

set -euo pipefail

CANDIDATE_FILE="${1:-${CANDIDATE_FILE:-}}"
if [ -z "$CANDIDATE_FILE" ]; then
    echo '{"valid": false, "aggregate_score": -1.0, "failure_reasons": ["No candidate file provided"]}'
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Step 1: Run the candidate to extract spec JSON
SPEC_JSON=$(python3 "$CANDIDATE_FILE" 2>/tmp/eval_stderr.txt)
if [ $? -ne 0 ]; then
    STDERR=$(cat /tmp/eval_stderr.txt | head -5 | tr '\n' ' ')
    echo "{\"valid\": false, \"aggregate_score\": -1.0, \"failure_reasons\": [\"Candidate file failed to execute: $STDERR\"]}"
    exit 0
fi

# Step 2: Validate spec locally (no GPU, instant)
VALIDATION=$(python3 -c "
import sys, json
sys.path.insert(0, '$REPO_ROOT')
try:
    from adamopt.optim_search.spec import MatrixOptimizerSpec
    spec_dict = json.loads('''$SPEC_JSON''')
    spec = MatrixOptimizerSpec.from_dict(spec_dict)
    spec.validate()
    print(json.dumps({'valid': True, 'spec_name': spec.name}))
except Exception as e:
    print(json.dumps({'valid': False, 'error': f'{type(e).__name__}: {e}'}))
" 2>/dev/null)

IS_VALID=$(echo "$VALIDATION" | python3 -c "import sys,json; print(json.load(sys.stdin).get('valid', False))")

if [ "$IS_VALID" != "True" ]; then
    ERROR_MSG=$(echo "$VALIDATION" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error', 'unknown'))")
    echo "{\"valid\": false, \"aggregate_score\": -1.0, \"failure_reasons\": [\"Spec validation failed: $ERROR_MSG\"]}"
    exit 0
fi

# Step 3: GPU evaluation (if requested)
GPU_EVAL="${GPU_EVAL:-0}"
if [ "$GPU_EVAL" = "1" ]; then
    # Write spec to temp file for modal_validate_spec.py
    SPEC_TMPFILE=$(mktemp /tmp/spec_XXXXXX.json)
    echo "$SPEC_JSON" > "$SPEC_TMPFILE"

    # Run GPU validation and capture output
    GPU_RESULT=$(cd "$REPO_ROOT" && source .venv/bin/activate && \
        uv run scripts/modal_validate_spec.py --spec "$SPEC_TMPFILE" --out /tmp/gpu_result.json 2>/dev/null)

    rm -f "$SPEC_TMPFILE"

    if [ -f /tmp/gpu_result.json ]; then
        # Parse the GPU results into alphaevolve format
        python3 -c "
import json
with open('/tmp/gpu_result.json') as f:
    r = json.load(f)
cm = r.get('candidate', {}).get('metrics')
bm = r.get('baseline', {}).get('metrics')
valid = r.get('valid', False)
if cm and bm:
    delta_bpb = bm['final_validation_bpb'] - cm['final_validation_bpb']
    speed_ratio = cm['mean_step_time_ms'] / max(bm['mean_step_time_ms'], 1e-8)
    # Score: higher is better. Reward lower BPB, penalize slowness.
    score = delta_bpb * 100 - max(0, speed_ratio - 1.5) * 10
    metrics = {
        'delta_bpb': delta_bpb,
        'final_val_bpb': cm['final_validation_bpb'],
        'baseline_val_bpb': bm['final_validation_bpb'],
        'speed_ratio': speed_ratio,
        'step_time_ms': cm['mean_step_time_ms'],
        'tokens_per_sec': cm['tokens_per_sec'],
        'nan_failures': cm['nan_failures'],
        'inf_failures': cm['inf_failures'],
        'grad_spikes': cm['grad_norm_spikes'],
    }
    print(json.dumps({
        'valid': valid,
        'aggregate_score': score,
        'metrics': metrics,
        'failure_reasons': r.get('notes', []),
    }))
else:
    print(json.dumps({
        'valid': False,
        'aggregate_score': -1.0,
        'failure_reasons': ['GPU eval produced no metrics'],
    }))
" 2>/dev/null
        rm -f /tmp/gpu_result.json
        exit 0
    fi
fi

# Step 3 (no GPU): Return local validation result with a neutral score.
# alphaevolve can still rank candidates by parsing the spec structure.
SPEC_NAME=$(echo "$VALIDATION" | python3 -c "import sys,json; print(json.load(sys.stdin).get('spec_name', 'unknown'))")
echo "{\"valid\": true, \"aggregate_score\": 0.0, \"metrics\": {\"spec_valid\": 1.0, \"spec_name\": \"$SPEC_NAME\"}, \"failure_reasons\": []}"
