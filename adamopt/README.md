# AdamOpt

AdamOpt is a constrained optimizer-search lab built from the AlphaEvolve MVP patterns in `alphaevolve/mvp`, but retargeted to optimizer specs instead of arbitrary code diffs.

The staged search plan is documented in:

- [`EVOLUTION_STRATEGY.md`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/EVOLUTION_STRATEGY.md)
- [`WIN_HIERARCHY.md`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/WIN_HIERARCHY.md)

The composite workspace that pairs `adamopt/` with `nanochat/` is documented in:

- [`../README.md`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/README.md)

Phase 1 focuses on one thing only:

- keep NanoChat's non-matrix optimizer path fixed
- evolve only the matrix optimizer path
- compare candidates on validation bpb at a fixed short-horizon budget

The code ships with a deterministic toy backend that mirrors NanoChat's parameter split so the evaluator, archive, mutation, and tournament logic can be tested locally without a NanoChat checkout. The NanoChat-facing assumptions are isolated behind the parameter-split builder and the evaluator contract.

A local NanoChat fork is now available at:

- `/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/nanochat`

## Layout

- `optim_search/spec.py`: constrained optimizer DSL and baseline specs
- `optim_search/candidate_optimizer.py`: spec-driven optimizer and NanoChat-style parameter split
- `optim_search/eval_candidate.py`: short-horizon evaluator and toy backend
- `optim_search/command_mutator.py`: CLI-driven code mutation + patch tracking for NanoChat
- `optim_search/validation.py`: per-candidate local preflight validation against a patched NanoChat workspace
- `optim_search/deployment.py`: traceable remote deployment + log/status fetch
- `optim_search/autonomous.py`: persistent autonomous patch/deploy/poll controller
- `optim_search/diff_engine.py`: SEARCH/REPLACE diff parser/applier
- `optim_search/score.py`: composite scoring and Pareto frontier logic
- `optim_search/tournament.py`: generation loop and multi-seed promotion
- `optim_search/archive.py`: candidate archive and persistence
- `optim_search/mutations.py`: bounded mutations over the optimizer DSL
- `scripts/search_optimizer.py`: CLI entrypoint
- `scripts/patch_nanochat_adamw.py`: patch NanoChat's AdamW path via `codex` or `claude`
- `scripts/deploy_candidate.py`: launch a patched candidate on a remote target and fetch trace snapshots

## Baseline parity target

The baseline spec matches NanoChat's current public optimizer split as of March 7, 2026:

- transformer matrix params -> Muon-like path
- `wte`, `value_embeds`, `lm_head`, and scalar lambdas -> AdamW-like path

The exact split reference used here is:

- [NanoChat `scripts/base_train.py`](https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py)
- [NanoChat `nanochat/gpt.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py)
- [NanoChat `nanochat/optim.py`](https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py)

## Run one baseline-vs-candidate comparison

```bash
python adamopt/scripts/search_optimizer.py compare \
  --backend toy \
  --out adamopt/examples/baseline_vs_trust_ratio.metrics.json
```

This runs:

- baseline: NanoChat-style Muon baseline spec
- candidate: baseline + per-layer trust ratio

The output JSON includes:

- final validation bpb
- train/validation AUC
- step time and tokens/sec
- grad norm spikes
- update/param norm ratios
- estimated optimizer-state bytes
- pass/fail status

## Run a small tournament

```bash
python adamopt/scripts/search_optimizer.py tournament \
  --backend toy \
  --generations 3 \
  --population 6 \
  --promotion-seeds 7,13,29 \
  --out-dir adamopt/runs/toy_demo
```

## Generate and track a code mutation against NanoChat

```bash
python adamopt/scripts/patch_nanochat_adamw.py \
  --provider codex \
  --scope adamw_math \
  --candidate-id cand_0001 \
  --instruction "Add an optional trust-ratio style scale inside the AdamW update, guarded by a group flag."
```

Or swap `--provider claude`.

Supported patch scopes:

- `adamw_math`: patch `nanochat/optim.py` around `adamw_step_fused` and AdamW wrappers
- `muon_math`: patch `nanochat/optim.py` around `muon_step_fused` and Muon wrappers
- `optimizer_routing`: patch `nanochat/gpt.py` around `setup_optimizer`

These are the low-touch patch points in the real Karpathy NanoChat clone:

- `nanochat/nanochat/gpt.py` `setup_optimizer`: routing between AdamW-like vs Muon groups
- `nanochat/nanochat/optim.py` `adamw_step_fused`: shared AdamW math for single-GPU and distributed paths
- `nanochat/nanochat/optim.py` `muon_step_fused`: shared Muon math for single-GPU and distributed paths

That keeps training-script changes out of the default mutation path. In practice, optimizer experiments should land in `nanochat/gpt.py` and `nanochat/optim.py`, with no changes to `scripts/base_train.py` unless a run wrapper truly needs it.

Each mutation gets its own tracked directory under `adamopt/runs/code_mutations/<candidate-id>/` with:

- `workspace/`: isolated patched NanoChat copy
- `prompt.txt`: prompt sent to the CLI
- `response.txt`: raw CLI output
- `mutation.diff`: unified diff after patch application
- `metadata.json`: command, paths, and apply stats

## Validate a mutation locally before deploy

Every patched candidate should pass a local preflight gate before it consumes remote GPU time.

```bash
python adamopt/scripts/search_optimizer.py validate-code \
  --candidate-dir adamopt/runs/code_mutations/cand_0001 \
  --scope adamw_math
```

This runs the candidate in a fresh subprocess and records:

- `validation.json`: structured pass/fail summary
- `validation.stdout.txt`: validator stdout
- `validation.stderr.txt`: validator stderr

The validator currently checks:

- syntax via `py_compile`
- importability of `nanochat.optim` and `nanochat.gpt`
- real `GPT` construction from the patched workspace
- `setup_optimizer()` on the mutated code
- one synthetic forward/backward/optimizer step
- `optimizer.state_dict()` round-tripability

## Deploy and trace a remote run

Use this when you want a patched candidate to run on a remote GPU box over SSH, including Azure VM or CoreWeave instances you can SSH into.

```bash
python adamopt/scripts/search_optimizer.py deploy-code \
  --candidate-dir adamopt/runs/code_mutations/cand_0001 \
  --candidate-id cand_0001 \
  --target-name azure-a100 \
  --transport ssh \
  --host your-hostname \
  --user ankit \
  --remote-base-dir ~/adamopt_remote \
  --run-command "python scripts/base_train.py --depth=4 --num-iterations=20"
```

Then fetch the latest traceable status and output:

```bash
python adamopt/scripts/search_optimizer.py trace-deployment \
  --deployment-dir adamopt/runs/code_mutations/cand_0001/deployments/<deployment-id>
```

Each deployment records:

- `manifest.json`: target, remote paths, launch command, remote pid
- `payload/`: exact staged workspace plus wrapper scripts
- `launcher.stdout.txt` and `launcher.stderr.txt`: submission-time output
- `fetched_status.json`: latest remote status snapshot
- `fetched_log_tail.txt`: latest fetched remote log tail
- `trace_snapshot.json`: consolidated local trace record

On the remote side, each deployment gets:

- `workspace/`: the patched NanoChat candidate
- `trace/run.log`: stdout/stderr from the launched command
- `trace/status.json`: machine-readable state (`running`, `succeeded`, `failed`)
- `trace/pid.txt`: launched process id

## Fully Autonomous Loop

This is the restartable controller that closes the async gap. It does all of the following without manual intervention:

- creates candidate mutation jobs
- runs `codex` or `claude` patch generation under an `asyncio` semaphore
- validates each patched candidate locally under a separate validation semaphore
- deploys patched workspaces under a separate deployment semaphore
- polls remote status/log/result files under a polling semaphore
- persists candidate state in `autonomous_state.json`
- appends lifecycle events to `events.jsonl`
- resumes safely if you rerun the command against the same `--out-dir`

Example with local transport:

```bash
python adamopt/scripts/search_optimizer.py autonomous-run \
  --provider codex \
  --candidate-count 4 \
  --instruction-template "Add an AdamW mutation for {candidate_id}." \
  --command-template "codex exec {prompt}" \
  --target-name local-dev \
  --transport local \
  --remote-base-dir /tmp/adamopt_remote \
  --nanochat-root nanochat \
  --run-command-template "python3 scripts/base_train.py --depth=4 --num-iterations=20"
```

For a real remote box over SSH, switch `--transport ssh` and set `--host`, `--user`, and optionally `--identity-file`.

The remote command gets these environment variables automatically:

- `ADAMOPT_TRACE_DIR`
- `ADAMOPT_STATUS_PATH`
- `ADAMOPT_LOG_PATH`
- `ADAMOPT_RESULT_PATH`
- `ADAMOPT_CANDIDATE_ID`

If your remote training/eval wrapper writes JSON metrics to `ADAMOPT_RESULT_PATH`, the autonomous controller fetches and stores them automatically.

Candidate lifecycle now goes through:

- `queued -> patching -> patched -> validating -> validated -> deploying -> running -> succeeded/failed`

## Example output

An example comparison artifact is stored at:

- `adamopt/examples/baseline_vs_trust_ratio.metrics.json`

## TODO

- Add a NanoChat subprocess backend that executes the real short-horizon train/eval path on NanoChat data.
- Feed command-generated NanoChat patches back into the tournament/evaluator loop for real code-level selection.
- Rank autonomous remote runs by fetched result payloads and feed winners into parent selection automatically.
- Tighten baseline parity against a live NanoChat checkout by comparing parameter-group counts, state shapes, and early-curve drift.
- Add optional code-level proposal support after the evaluator is trusted.
- Cache baseline multi-seed runs across tournaments to avoid duplicate work.
- Add richer failure taxonomies for OOM, divergence, and slowdown regressions.
