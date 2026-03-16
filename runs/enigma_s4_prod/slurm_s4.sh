#!/bin/bash
#SBATCH --job-name=enigma-s4
#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=/mnt/sharefs/user54/nanoe/runs/enigma_s4_prod/slurm_%A_%a.out
#SBATCH --error=/mnt/sharefs/user54/nanoe/runs/enigma_s4_prod/slurm_%A_%a.err

set -euo pipefail

NANOE_DIR="$HOME/nanoe"
RUN_DIR="$NANOE_DIR/runs/enigma_s4_prod"

source "$NANOE_DIR/nanochat/.venv/bin/activate"
export PYTHONPATH="$NANOE_DIR:$NANOE_DIR/adamopt:${PYTHONPATH:-}"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# Disable torch.compile (triton compilation fails on this cluster)
export TORCHDYNAMO_DISABLE=1

# Mutation array: index -> mutation_id
MUTATIONS=(
    "none"                    # 0: Production baseline (unpatched)
    "H81_trust_ratio"         # 1: Add trust ratio to production Muon
    "H70_embed_wd"            # 2: Embedding weight decay 0.01
    "H71_beta1_warmup"        # 3: AdamW beta1 warmup 0.7→0.8
    "H73_eps_schedule"        # 4: Epsilon schedule 1e-6→1e-10
    "H78_scalar_warmup"       # 5: Scalar LR warmup 200 steps
    "H60_beta2_warmup"        # 6: NorMuon beta2 warmup 0.8→0.95
    "H63_momentum_overshoot"  # 7: Momentum 0.85→0.97→0.95
    "H62_wd_cosine"           # 8: WD cosine decay instead of linear
    "H64_nesterov_schedule"   # 9: Nesterov blend 0.7→0.95 schedule
)

MUTATION="${MUTATIONS[$SLURM_ARRAY_TASK_ID]}"
export ENIGMA_MUTATION="$MUTATION"

STEPS=5000
EVAL_EVERY=250
DEPTH=12
RESULTS_DIR="$RUN_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "=== ENIGMA Stage 4 — Production GPT-2: $MUTATION ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Steps: $STEPS, Depth: $DEPTH"

# Run production training with monkey-patch wrapper
cd "$NANOE_DIR/nanochat"

python -c "
import os, sys, json, time, math, re
from pathlib import Path

mutation = os.environ.get('ENIGMA_MUTATION', 'none')
steps = $STEPS
depth = $DEPTH
eval_every = $EVAL_EVERY
results_dir = Path('$RESULTS_DIR')

print(f'Mutation: {mutation}')
print(f'Steps: {steps}, Depth: {depth}')

# Apply fused-kernel-level patches BEFORE importing base_train
if mutation not in ('none', 'H70_embed_wd', 'H78_scalar_warmup', 'H60_beta2_warmup',
                     'H63_momentum_overshoot', 'H62_wd_cosine', 'H64_nesterov_schedule'):
    sys.path.insert(0, '$NANOE_DIR')
    import enigma.stage4_patch  # applies monkey patches to nanochat.optim

# Now import production components
from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import MuonAdamW, muon_step_fused
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.loss_eval import evaluate_bpb

import torch
import random

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = 'cuda'

# Build model (production config)
config = GPTConfig(n_layer=depth, sequence_len=512)
model = GPT(config).to(DEVICE)
model.train()

# Build optimizer (production method)
optimizer = model.setup_optimizer(weight_decay=0.2)
for group in optimizer.param_groups:
    group['initial_lr'] = group['lr']

# ── Post-optimizer patches ────────────────────────────────
if mutation == 'H70_embed_wd':
    for group in optimizer.param_groups:
        if group['kind'] == 'adamw':
            # Embedding groups have lr > 0.1 (after dmodel_lr_scale)
            if group['lr'] > 0.1:
                group['weight_decay'] = 0.01
                print(f'  [H70] Set weight_decay=0.01 on group with lr={group[\"lr\"]:.4f}')

# ── Production schedules (from base_train.py) ─────────────
def get_lr_multiplier(it):
    warmdown_iters = round(0.5 * steps)
    if it <= steps - warmdown_iters:
        return 1.0
    progress = (steps - it) / warmdown_iters
    return progress  # linear warmdown to 0

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(it):
    return 0.2 * (1 - it / steps)

# ── Schedule mutation overrides ───────────────────────────
if mutation == 'H63_momentum_overshoot':
    def get_muon_momentum(it):
        if it < 300:
            return 0.85 + (0.97 - 0.85) * (it / 300)
        elif it < 1000:
            return 0.97 + (0.95 - 0.97) * ((it - 300) / 700)
        return 0.95

if mutation == 'H62_wd_cosine':
    def get_weight_decay(it):
        progress = it / steps
        return 0.2 * 0.5 * (1.0 + math.cos(math.pi * progress))

# ── Data loaders ──────────────────────────────────────────
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=DEVICE)
DEVICE_BATCH_SIZE = 2
MAX_SEQ_LEN = 512
TOTAL_BATCH_SIZE = 1024

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

# ── Training loop (mirrors production) ────────────────────
curve = []
step_times_ms = []
failure_type = None

t_total = time.perf_counter()
for step in range(steps + 1):
    last_step = (step == steps)

    # Eval
    if step % eval_every == 0 or last_step:
        model.eval()
        val_loader = build_val_loader()
        eval_steps_count = max(1, 512 // tokens_per_fwdbwd)
        with torch.no_grad():
            val_bpb = evaluate_bpb(model, val_loader, eval_steps_count, token_bytes)
        curve.append({'step': step, 'val_bpb': float(val_bpb)})
        model.train()
        print(f'  step {step:>5d}: val_bpb={val_bpb:.6f}')

    if last_step:
        break

    # Forward + backward
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
        break

    # Apply schedules (production-style)
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_wd = get_weight_decay(step)
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * lrm
        if group['kind'] == 'muon':
            group['momentum'] = muon_momentum
            group['weight_decay'] = muon_wd

    # Schedule mutations that modify per-step group params
    if mutation == 'H60_beta2_warmup':
        beta2 = 0.8 + (0.95 - 0.8) * min(1.0, step / 500)
        for group in optimizer.param_groups:
            if group['kind'] == 'muon':
                group['beta2'] = beta2

    if mutation == 'H78_scalar_warmup':
        if step < 200:
            warmup_mult = (step + 1) / 200
            for group in optimizer.param_groups:
                if group['kind'] == 'adamw' and group.get('initial_lr', 0) < 0.1:
                    group['lr'] = group['initial_lr'] * lrm * warmup_mult

    if mutation == 'H64_nesterov_schedule':
        # Can't easily decouple in compiled kernel, so modify momentum schedule
        # to approximate: use lower momentum (=nesterov blend) early
        blend = 0.7 + (0.95 - 0.7) * min(1.0, step / 500)
        for group in optimizer.param_groups:
            if group['kind'] == 'muon':
                group['momentum'] = blend

    optimizer.step()
    model.zero_grad(set_to_none=True)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    step_times_ms.append(elapsed_ms)

    if step % 500 == 0:
        train_bpb = train_loss / math.log(2.0)
        print(f'  step {step:>5d}: train_bpb={train_bpb:.4f} ({elapsed_ms:.0f}ms)')

t_total_s = time.perf_counter() - t_total

# Results
val_points = [p['val_bpb'] for p in curve]
final_bpb = val_points[-1] if val_points else float('inf')
best_bpb = min(val_points) if val_points else float('inf')

result = {
    'mutation': mutation,
    'valid': failure_type is None,
    'failure_type': failure_type,
    'final_validation_bpb': final_bpb,
    'best_validation_bpb': best_bpb,
    'mean_step_time_ms': sum(step_times_ms) / max(1, len(step_times_ms)),
    'total_time_s': t_total_s,
    'steps_completed': len(step_times_ms),
    'curve': curve,
    'production_baseline': True,
    'schedules': {
        'lr': 'linear_warmdown_50pct',
        'momentum': 'warmup_0.85_to_0.95_over_300',
        'weight_decay': 'linear_decay_to_0',
    },
}

out_file = results_dir / f'{mutation}_real.json'
out_file.write_text(json.dumps(result, indent=2))
print(f'Done: {mutation} valid={result[\"valid\"]} bpb={final_bpb:.6f} best={best_bpb:.6f} ({t_total_s:.1f}s)')
"

echo "=== Task $SLURM_ARRAY_TASK_ID ($MUTATION) done ==="
