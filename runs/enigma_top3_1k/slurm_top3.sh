#!/bin/bash
#SBATCH --job-name=enigma-1k
#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/sharefs/user54/nanoe/runs/enigma_top3_1k/slurm_%A_%a.out
#SBATCH --error=/mnt/sharefs/user54/nanoe/runs/enigma_top3_1k/slurm_%A_%a.err

set -euo pipefail

NANOE_DIR="$HOME/nanoe"
RUN_DIR="$NANOE_DIR/runs/enigma_top3_1k"

source "$NANOE_DIR/nanochat/.venv/bin/activate"
export PYTHONPATH="$NANOE_DIR:$NANOE_DIR/adamopt:${PYTHONPATH:-}"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

SPEC_FILES=($(ls -1 "$RUN_DIR/specs/"*.json | sort))
SPEC_FILE="${SPEC_FILES[$SLURM_ARRAY_TASK_ID]}"
SPEC_NAME=$(basename "$SPEC_FILE" .json)

echo "=== ENIGMA Top 3 — 1000 Steps GPT-2: $SPEC_NAME ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"

python -c "
import json, sys, os, math, random, time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, '$NANOE_DIR/adamopt')
sys.path.insert(0, '$NANOE_DIR')

from optim_search.spec import MatrixOptimizerSpec
from optim_search.candidate_optimizer import SpecCandidateOptimizer
from optim_search.types import CurvePoint, EvaluationOutcome, StepTelemetry, TrialMetrics
from enigma.run_stage2 import Stage2Optimizer

import torch

os.environ.setdefault('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.loss_eval import evaluate_bpb

spec_file = Path('$SPEC_FILE')
spec_name = '$SPEC_NAME'
results_dir = Path('$RUN_DIR/results')
results_dir.mkdir(parents=True, exist_ok=True)

SEED = 42
STEPS = 1000
EVAL_EVERY = 50
DEPTH = 12
MAX_SEQ_LEN = 512
DEVICE_BATCH_SIZE = 2
TOTAL_BATCH_SIZE = 1024
DEVICE = 'cuda'

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

spec_dict = json.loads(spec_file.read_text())
spec = MatrixOptimizerSpec.from_dict(spec_dict)
code_mutation = spec.metadata.get('code_mutation', 'none') if spec.metadata else 'none'
if not code_mutation:
    code_mutation = 'none'

print(f'Spec: {spec.name}, code_mutation: {code_mutation}')

model_config = GPTConfig(n_layer=DEPTH, sequence_len=MAX_SEQ_LEN)
model = GPT(model_config).to(DEVICE)
model.train()

model_dim = model.config.n_embd
dmodel_lr_scale = (model_dim / 768) ** -0.5
all_block_params = list(model.transformer.h.parameters())
matrix_params = [p for p in all_block_params if p.ndim >= 2]
block_scalar_params = [p for p in all_block_params if p.ndim < 2]
embedding_params = list(model.transformer.wte.parameters())
value_embeds_params = list(model.value_embeds.parameters())
lm_head_params = list(model.lm_head.parameters())
resid_params = [model.resid_lambdas]
x0_params = [model.x0_lambdas]
adam_betas = (0.8, 0.95)

param_groups = [
    dict(kind='adamw', group_name='lm_head', params=lm_head_params,
         lr=0.004 * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
    dict(kind='adamw', group_name='embedding', params=embedding_params,
         lr=0.2 * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
    dict(kind='adamw', group_name='value_embeds', params=value_embeds_params,
         lr=0.2 * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
    dict(kind='adamw', group_name='block_non_matrix', params=block_scalar_params,
         lr=0.5 * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
    dict(kind='adamw', group_name='resid_scalars', params=resid_params,
         lr=0.5 * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
    dict(kind='adamw', group_name='x0_scalars', params=x0_params,
         lr=0.5, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
]
for shape in sorted({p.shape for p in matrix_params}):
    group_params = [p for p in matrix_params if p.shape == shape]
    param_groups.append(dict(
        kind='matrix_candidate', group_name=f'matrix_{shape}',
        params=group_params, lr=0.02, weight_decay=0.2,
    ))
param_groups = [g for g in param_groups if g['params']]

if code_mutation != 'none':
    optimizer = Stage2Optimizer(param_groups, spec, code_mutation=code_mutation)
else:
    optimizer = SpecCandidateOptimizer(param_groups, spec)
for group in optimizer.param_groups:
    group['initial_lr'] = group['lr']

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=DEVICE)
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, split='train', device=DEVICE,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, split='val', device=DEVICE,
)

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

x, y, _ = next(train_loader)

warmup_steps = max(1, STEPS // 10)
def get_lr_multiplier(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, STEPS - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

curve = []
step_times_ms = []
update_ratios = []
max_ratio = 0.0
grad_norm_spikes = 0
running_grad_norm = 0.0
nan_failures = 0
inf_failures = 0
failure_type = None

t_total = time.perf_counter()
for step in range(1, STEPS + 1):
    model.train()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    for _ in range(grad_accum_steps):
        loss = model(x, y)
        accumulated_loss += loss.detach().item()
        (loss / grad_accum_steps).backward()
        x, y, _ = next(train_loader)
    train_loss = accumulated_loss / grad_accum_steps

    if not math.isfinite(train_loss):
        failure_type = 'non_finite_loss'
        nan_failures += int(math.isnan(train_loss))
        inf_failures += int(math.isinf(train_loss))
        break

    grad_sq = sum(float(p.grad.float().pow(2).sum().item()) for p in model.parameters() if p.grad is not None)
    grad_norm = math.sqrt(grad_sq)
    if running_grad_norm > 0 and grad_norm > 4.0 * running_grad_norm:
        grad_norm_spikes += 1
    running_grad_norm = grad_norm if running_grad_norm == 0 else 0.95 * running_grad_norm + 0.05 * grad_norm

    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * lrm

    optimizer.set_step_context(loss_value=train_loss, step=step, total_steps=STEPS)
    optimizer.step()

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    step_times_ms.append(elapsed_ms)
    update_ratios.append(optimizer.last_step_stats.mean_update_param_ratio)
    max_ratio = max(max_ratio, optimizer.last_step_stats.max_update_param_ratio)

    invalid = False
    for p in model.parameters():
        if torch.isnan(p).any():
            nan_failures += 1; invalid = True; break
        if torch.isinf(p).any():
            inf_failures += 1; invalid = True; break
    if invalid:
        failure_type = 'non_finite_parameters'
        break

    train_bpb = train_loss / math.log(2.0)
    point = {'step': step, 'train_bpb': train_bpb, 'tokens_seen': step * TOTAL_BATCH_SIZE, 'val_bpb': None}

    if step % EVAL_EVERY == 0 or step == STEPS:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = max(1, 512 // tokens_per_fwdbwd)
        with torch.no_grad():
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        point['val_bpb'] = float(val_bpb)
        model.train()
        print(f'  step {step:>4d}: train={train_bpb:.4f} val={point[\"val_bpb\"]:.4f} ({elapsed_ms:.0f}ms)')

    curve.append(point)

t_total_s = time.perf_counter() - t_total
val_points = [p['val_bpb'] for p in curve if p['val_bpb'] is not None]
final_bpb = val_points[-1] if val_points else curve[-1]['train_bpb']
best_bpb = min(val_points) if val_points else final_bpb

result = {
    'candidate_id': spec_name,
    'spec_name': spec.name,
    'code_mutation': code_mutation,
    'valid': failure_type is None,
    'failure_type': failure_type,
    'final_validation_bpb': final_bpb,
    'best_validation_bpb': best_bpb,
    'mean_step_time_ms': sum(step_times_ms) / max(1, len(step_times_ms)),
    'tokens_per_sec': (STEPS * TOTAL_BATCH_SIZE) / t_total_s if t_total_s > 0 else 0,
    'stability_penalty': float((nan_failures + inf_failures) * 10 + grad_norm_spikes),
    'grad_norm_spikes': grad_norm_spikes,
    'nan_failures': nan_failures,
    'inf_failures': inf_failures,
    'total_time_s': t_total_s,
    'steps_completed': len(curve),
    'curve': curve,
}

out_file = results_dir / f'{spec_name}_real.json'
out_file.write_text(json.dumps(result, indent=2))
print(f'Done: {spec_name} valid={result[\"valid\"]} bpb={final_bpb:.6f} best={best_bpb:.6f} ({t_total_s:.1f}s)')
"

echo "=== Task $SLURM_ARRAY_TASK_ID done ==="
