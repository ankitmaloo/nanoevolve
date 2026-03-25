# `train_gpt.py` Topology And Attack Map

This document maps the root [`train_gpt.py`](/home/coder/parameter-golf/train_gpt.py) as it exists now.

The goal is not to restate every line mechanically. The goal is to preserve:

- the execution topology
- the state transitions through training and export
- the places where the current root script is strong
- the places where it is now clearly behind the frontier
- the highest-leverage attack surfaces if we want to beat the current accepted record

## Important Framing

The root script is a readable starter, not the real frontier implementation.

That matters because there are two very different questions:

1. What does the root script do, line by line?
2. Where should we attack if the real goal is to beat SOTA?

This file answers both, but it keeps them separate.

## The Topology At A Glance

The root script is a single-path training and deployment pipeline:

1. Read env vars into `Hyperparameters`
2. Define Muon optimizer internals
3. Define tokenizer-aware BPB evaluation helpers
4. Define post-training int8 quantization helpers
5. Define shard loader and deterministic token stream
6. Define transformer modules
7. Build model and optimizers
8. Warm up compiled paths, then rewind state
9. Run the training loop under a 600s wallclock cap
10. Export the trained weights
11. Quantize to int8 + zlib
12. Reload the quantized artifact
13. Re-evaluate the round-tripped model

This is important: the script already evaluates the deployed artifact, not just the pre-export checkpoint. That is the right objective. The main weakness is that the deployment path is much weaker than the current frontier.

## Section Map

### 1. File header and imports

Lines:

- [`train_gpt.py:1`](/home/coder/parameter-golf/train_gpt.py#L1) to [`train_gpt.py:28`](/home/coder/parameter-golf/train_gpt.py#L28)

Role:

- Declares that this file is intentionally readable and not intended to be the SOTA submission path.
- Imports only standard training dependencies plus SentencePiece, NumPy, and PyTorch.

State transition:

- No model state yet.
- Sets up the module namespace the rest of the file depends on.

### 2. Hyperparameter capture

Lines:

- [`train_gpt.py:39`](/home/coder/parameter-golf/train_gpt.py#L39) to [`train_gpt.py:87`](/home/coder/parameter-golf/train_gpt.py#L87)

Role:

- Captures the full run spec from environment variables.

Key groups:

- Data paths and tokenizer
- Validation cadence
- Training budget
- Model shape
- Optimizer hyperparameters

State transition:

- External env vars become immutable runtime configuration.
- The main run contract is frozen here.

Important observations:

- Default model is `9L / 512d / 8H / 4KV / MLP x2 / seq1024`.
- Default training is `524,288` tokens per step.
- Default schedule is still a readable baseline schedule, not a frontier one.
- `grad_clip_norm` exists but defaults to `0.0`.

### 3. Muon update construction

Lines:

- [`train_gpt.py:96`](/home/coder/parameter-golf/train_gpt.py#L96) to [`train_gpt.py:168`](/home/coder/parameter-golf/train_gpt.py#L168)

Role:

- Turns matrix gradients into orthogonalized Muon updates.

Subgraph:

- `zeropower_via_newtonschulz5`
- `Muon.step`

State transition:

- Raw 2D gradient `g`
- momentum buffer update
- optional Nesterov combination
- Newton-Schulz orthogonalization
- flattened bf16 update buffer
- distributed all-reduce
- in-place parameter step

Important observations:

- This is plain Muon, not NorMuon.
- There is no Muon weight decay in the root script.
- `updates_flat` is freshly allocated every step.
- Distributed communication is simple and correct, but not the faster banked/overlapped style used in frontier scripts.

### 4. Tokenizer-agnostic BPB setup

Lines:

- [`train_gpt.py:180`](/home/coder/parameter-golf/train_gpt.py#L180) to [`train_gpt.py:278`](/home/coder/parameter-golf/train_gpt.py#L278)

Role:

- Converts token loss into challenge metric `val_bpb`.

Subgraph:

- `build_sentencepiece_luts`
- `load_validation_tokens`
- `eval_val`

State transition:

- SentencePiece vocabulary pieces
- byte-count LUTs
- fixed validation token stream
- CE loss accumulation
- token-byte accounting
- final `val_loss` and `val_bpb`

Important observations:

- Validation is non-overlapping and fixed-window at `train_seq_len`.
- There is no sliding eval here.
- There is no BOS reset or document-isolated eval.
- There is no score-first TTT path.

This is one of the cleanest places where the root script lags modern record stacks.

### 5. Post-training quantization

Lines:

- [`train_gpt.py:288`](/home/coder/parameter-golf/train_gpt.py#L288) to [`train_gpt.py:422`](/home/coder/parameter-golf/train_gpt.py#L422)

Role:

- Defines the deployment artifact format.

Subgraph:

- control tensor passthrough rules
- `quantize_float_tensor`
- `quantize_state_dict_int8`
- `dequantize_state_dict_int8`

State transition:

- fp32 or bf16 tensor
- per-row or per-tensor clipping
- int8 values plus scale tensors
- serialized quantized object
- dequantized reconstruction for roundtrip eval

Important observations:

- This is simple int8 clip-and-round quantization.
- Compression is `zlib`.
- Large 2D tensors become int8; small float tensors are stored as fp16 or fp32 passthrough.
- This is currently the single largest technical gap between root and the frontier.

Why:

- Frontier scripts moved to GPTQ-style quantization, mixed int5/int6 schemes, better compression, and more careful passthrough rules.
- Root export quality is leaving too much BPB on the table.

### 6. Data loading and stream model

Lines:

- [`train_gpt.py:429`](/home/coder/parameter-golf/train_gpt.py#L429) to [`train_gpt.py:494`](/home/coder/parameter-golf/train_gpt.py#L494)

Role:

- Defines the training token stream.

Subgraph:

- `load_data_shard`
- `TokenStream`
- `DistributedTokenLoader`

State transition:

- shard file
- validated header
- token tensor
- sequential global stream
- disjoint per-rank span
- shifted `(x, y)` training batch

Important observations:

- The stream is deterministic and sequential.
- There is no shuffle.
- There is no curriculum.
- There is no shard ordering logic.

That made sense for a starter script. It is now a meaningful opportunity cost, because frontier work has started to treat shard ordering as a real budget-allocation mechanism under the 600s cap.

### 7. Transformer primitives

Lines:

- [`train_gpt.py:500`](/home/coder/parameter-golf/train_gpt.py#L500) to [`train_gpt.py:724`](/home/coder/parameter-golf/train_gpt.py#L724)

Role:

- Defines the actual model family.

Subgraph:

- `RMSNorm`
- `CastedLinear`
- `restore_low_dim_params_to_fp32`
- `Rotary`
- `apply_rotary_emb`
- `CausalSelfAttention`
- `MLP`
- `Block`
- `GPT`

#### 7a. Normalization and matmul wrappers

Lines:

- [`train_gpt.py:500`](/home/coder/parameter-golf/train_gpt.py#L500) to [`train_gpt.py:521`](/home/coder/parameter-golf/train_gpt.py#L521)

State transition:

- hidden states are RMS-normalized
- linear weights stay high-precision in storage but are cast to compute dtype at matmul time

Observation:

- `RMSNorm` has no learnable affine weight.
- Small or control parameters are explicitly restored to fp32.

#### 7b. RoPE cache

Lines:

- [`train_gpt.py:524`](/home/coder/parameter-golf/train_gpt.py#L524) to [`train_gpt.py:552`](/home/coder/parameter-golf/train_gpt.py#L552)

State transition:

- sequence length
- cached cos/sin tables
- rotated q/k channels

Observation:

- Full head-dim RoPE only.
- No partial RoPE.

#### 7c. Attention block

Lines:

- [`train_gpt.py:555`](/home/coder/parameter-golf/train_gpt.py#L555) to [`train_gpt.py:603`](/home/coder/parameter-golf/train_gpt.py#L603)

State transition:

- hidden state
- q/k/v projections
- q/k RMS norm
- RoPE
- per-head `q_gain`
- SDPA call
- output projection

Observation:

- Uses GQA and PyTorch SDPA.
- No XSA.
- No gated attention.
- No VRL.
- No parameter banking.

#### 7d. MLP

Lines:

- [`train_gpt.py:606`](/home/coder/parameter-golf/train_gpt.py#L606) to [`train_gpt.py:617`](/home/coder/parameter-golf/train_gpt.py#L617)

State transition:

- linear up-projection
- `relu`
- square
- linear down-projection

Observation:

- This is classic `relu^2`.
- Frontier code moved to `LeakyReLU(0.5)^2` because it preserves negative gradient flow while keeping the squared nonlinearity.

#### 7e. Residual block

Lines:

- [`train_gpt.py:620`](/home/coder/parameter-golf/train_gpt.py#L620) to [`train_gpt.py:645`](/home/coder/parameter-golf/train_gpt.py#L645)

State transition:

- current hidden state and initial embedding are mixed by `resid_mix`
- attention branch runs
- attention residual added with learned `attn_scale`
- MLP branch runs
- MLP residual added with learned `mlp_scale`

Observation:

- This is already stronger than a plain transformer block.
- `resid_mix` starts as `[1, 0]`, so the embedding shortcut is initially disabled.

#### 7f. Full GPT model

Lines:

- [`train_gpt.py:648`](/home/coder/parameter-golf/train_gpt.py#L648) to [`train_gpt.py:724`](/home/coder/parameter-golf/train_gpt.py#L724)

State transition:

- token ids
- token embeddings
- first-half encoder blocks produce skip stack
- second-half decoder blocks consume reversed skip stack
- final norm
- tied or untied output head
- tanh logit softcap
- cross entropy

Observation:

- The topology is a shallow U-Net-ish transformer with learnable skip reinjection.
- Good starter trunk.
- Missing most current frontier additions:
  - 11-layer trunk
  - MLP 3x
  - BigramHash
  - SmearGate
  - XSA
  - Partial RoPE
  - LN scale
  - VE128
  - VRL

### 8. Runtime bootstrapping

Lines:

- [`train_gpt.py:731`](/home/coder/parameter-golf/train_gpt.py#L731) to [`train_gpt.py:844`](/home/coder/parameter-golf/train_gpt.py#L844)

Role:

- Builds the actual execution environment.

State transition:

- source code string captured
- hyperparameters instantiated
- Muon backend compiled
- distributed runtime initialized
- SDPA backend selected
- logging set up
- tokenizer and val data loaded
- model instantiated and compiled

Important observations:

- CUDA is required.
- Root uses flash SDPA via PyTorch dispatch, not more specialized frontier attention machinery.
- The model is compiled with `torch.compile(..., dynamic=False, fullgraph=True)`.

### 9. Optimizer partitioning

Lines:

- [`train_gpt.py:846`](/home/coder/parameter-golf/train_gpt.py#L846) to [`train_gpt.py:910`](/home/coder/parameter-golf/train_gpt.py#L910)

Role:

- Splits parameters into optimizer lanes.

State transition:

- block parameters
- matrix params to Muon
- low-dim and control params to Adam
- embeddings to Adam
- optional untied head to Adam

Observation:

- This split is sensible.
- It is not yet export-aware enough:
  - no Muon weight decay
  - no EMA
  - no SWA
  - no late-QAT machinery

### 10. Warmup and schedule

Lines:

- [`train_gpt.py:916`](/home/coder/parameter-golf/train_gpt.py#L916) to [`train_gpt.py:961`](/home/coder/parameter-golf/train_gpt.py#L961)

Role:

- Primes compiled code paths, then restores the initial state.

State transition:

- initial model state snapshotted
- initial optimizer state snapshotted
- `warmup_steps` real train steps executed
- weights and optimizer states restored
- dataloader reset

Observation:

- This is clever and correct.
- Default `WARMUP_STEPS=20` is larger than necessary for a competitive run.
- Recovering those steps is a throughput attack surface.

### 11. Main training loop

Lines:

- [`train_gpt.py:963`](/home/coder/parameter-golf/train_gpt.py#L963) to [`train_gpt.py:1055`](/home/coder/parameter-golf/train_gpt.py#L1055)

Role:

- Owns the main state machine.

State transition per iteration:

1. Determine whether this is the last step
2. Optionally validate
3. Stop if needed
4. Compute current LR scale from wallclock
5. Zero grads
6. Run grad accumulation microsteps
7. Average train loss
8. Warm Muon momentum toward target
9. Apply scaled learning rates
10. Optional gradient clipping
11. Step all optimizers
12. Log
13. Detect wallclock cap

Observation:

- The control flow is very clean.
- There is no EMA.
- There is no SWA.
- There is no QAT enable window.
- There is no post-step perturbation or regularization lane.
- Validation runs inside training, but evaluation time is excluded from the wallclock budget by explicitly pausing the timer.

### 12. Serialization and roundtrip validation

Lines:

- [`train_gpt.py:1057`](/home/coder/parameter-golf/train_gpt.py#L1057) to [`train_gpt.py:1126`](/home/coder/parameter-golf/train_gpt.py#L1126)

Role:

- Produces final artifacts and evaluates the deployed weights.

State transition:

- raw state dict saved
- quantized object produced
- `torch.save`
- `zlib.compress`
- artifact written
- artifact reloaded
- dequantized weights restored into the base model
- final evaluation run on quantized roundtrip model

Observation:

- This is the correct final objective.
- The problem is not that the script ignores deployment.
- The problem is that the deployment stack is obsolete relative to the frontier.

## State Machine View

If we compress the whole file into a single state graph, it looks like this:

`env -> args -> tokenizer/data setup -> model graph -> optimizer split -> warmup compile traces -> training loop -> raw checkpoint -> quantized artifact -> dequantized eval model -> final BPB`

The most important thing to notice is where score is actually determined:

- not at raw train loss
- not at pre-export validation
- at post-export roundtrip BPB

That means the attack surfaces that matter most are the ones that survive export and improve deployed scoring.

## Where The Root Script Is Already Good

- Clean, readable control flow
- Correct wallclock-aware schedule
- Correct token-to-byte accounting
- Correct deployment-time roundtrip evaluation
- Sensible Muon/Adam split
- Reasonable starter architecture
- Good mixed-precision handling for low-dim parameters

These are not the places to attack first.

## Where The Root Script Is Clearly Behind

This is the shortest list of real bottlenecks.

### 1. Export path is weak

Current:

- simple int8
- simple clipping
- `zlib`

Missing:

- GPTQ-family quantization
- mixed int5/int6 policy
- more artifact-aware passthrough rules
- stronger compression

This is the biggest bottleneck.

### 2. Eval policy is weak

Current:

- non-overlapping fixed-window eval
- no BOS reset
- no doc isolation
- no sliding context
- no legal score-first TTT

This is the second biggest bottleneck.

### 3. Trunk is behind frontier architecture

Current:

- 9 layers
- MLP x2
- no XSA
- no Partial RoPE
- no LN scale
- no VE128
- no VRL

This is the main pre-export model bottleneck.

### 4. Training-to-deployment bridge is weak

Current:

- no EMA
- no SWA
- no late export-aligned QAT
- no Muon WD

The current model is not being explicitly shaped for quantized deployment.

### 5. Training data ordering is naive

Current:

- sequential shard order forever

Missing:

- shard ordering
- curriculum
- budget-aware data prioritization

This is less important than export and eval, but it is now a real lane.

## Attack Surfaces Ranked

This ranking is for "what should we attack if we actually want to beat SOTA", not "what is easiest to add to the root script."

### Tier 0: Immediate must-attack surfaces

#### A. Export replacement

Best attack:

- replace the root int8+zlib export lane with GPTQ or at least GPTQ-lite / strong int6 export

Why:

- root deployment quality is too far behind
- modern record scripts spend a large fraction of their gain here

Where:

- [`train_gpt.py:321`](/home/coder/parameter-golf/train_gpt.py#L321) to [`train_gpt.py:422`](/home/coder/parameter-golf/train_gpt.py#L422)
- [`train_gpt.py:1076`](/home/coder/parameter-golf/train_gpt.py#L1076) to [`train_gpt.py:1099`](/home/coder/parameter-golf/train_gpt.py#L1076)

#### B. Eval replacement

Best attack:

- replace plain `eval_val` with sliding eval
- add BOS/document reset logic
- add legal score-first TTT as an optional final scoring lane

Why:

- the root scoring path is now too weak even for a strong checkpoint

Where:

- [`train_gpt.py:219`](/home/coder/parameter-golf/train_gpt.py#L219) to [`train_gpt.py:278`](/home/coder/parameter-golf/train_gpt.py#L278)

#### C. Frontier trunk port

Best attack:

- stop treating the 9L/2x trunk as sacred
- move toward 11L, stronger context path, and export-friendly refinements

Why:

- even perfect export on the current trunk likely leaves real score on the table

Where:

- [`train_gpt.py:555`](/home/coder/parameter-golf/train_gpt.py#L555) to [`train_gpt.py:724`](/home/coder/parameter-golf/train_gpt.py#L724)

### Tier 1: Highest-confidence mechanisms

These are the best first-wave hypotheses.

#### 1. Full GPTQ or strong GPTQ-lite export

Mechanism:

- quantization error becomes Hessian-aware rather than plain clip-and-round

Expected effect:

- biggest single deployment lift

Target lines:

- quantization section
- final export section

#### 2. LeakyReLU(0.5)^2

Mechanism:

- preserves negative-gradient learning in the MLP while keeping the squared nonlinearity

Expected effect:

- small but reliable gain

Target lines:

- [`train_gpt.py:615`](/home/coder/parameter-golf/train_gpt.py#L615) to [`train_gpt.py:617`](/home/coder/parameter-golf/train_gpt.py#L617)

#### 3. EMA

Mechanism:

- smooths the trained weights before export

Expected effect:

- better deployed checkpoint than last-step raw weights

Target lines:

- training loop post-step area
- serialization path

#### 4. XSA

Mechanism:

- deeper layers get access to broader sequence context than the base local causal window

Expected effect:

- meaningful quality gain that tends to survive export

Target lines:

- attention forward path
- model constructor

#### 5. Partial RoPE plus LN scale

Mechanism:

- improve positional geometry and deep-layer scaling without large parameter cost

Expected effect:

- stackable medium gain

Target lines:

- rotary application
- block constructor and forward

#### 6. VE128 or VRL

Mechanism:

- strengthen the value path cheaply

Expected effect:

- second-wave architectural lift once export headroom exists

Target lines:

- attention block
- GPT constructor

### Tier 2: Strong helper mechanisms

#### 7. Muon weight decay

Mechanism:

- keeps matrix weights tighter and more export-friendly

Target lines:

- [`train_gpt.py:163`](/home/coder/parameter-golf/train_gpt.py#L163) to [`train_gpt.py:166`](/home/coder/parameter-golf/train_gpt.py#L163)

#### 8. Longer warmdown plus lower warmup waste

Mechanism:

- spend more of the 600s budget in useful late training and less in throwaway compile priming

Target lines:

- [`train_gpt.py:924`](/home/coder/parameter-golf/train_gpt.py#L924) to [`train_gpt.py:933`](/home/coder/parameter-golf/train_gpt.py#L924)
- [`train_gpt.py:937`](/home/coder/parameter-golf/train_gpt.py#L937) to [`train_gpt.py:961`](/home/coder/parameter-golf/train_gpt.py#L937)

#### 9. Export-aligned late QAT

Mechanism:

- make training aware of the eventual low-bit deployment lane near the end of training

Target lines:

- `CastedLinear.forward`
- training loop
- export lane

#### 10. Shard ordering / curriculum

Mechanism:

- reallocate the fixed time budget toward more valuable data earlier

Target lines:

- [`train_gpt.py:446`](/home/coder/parameter-golf/train_gpt.py#L446) to [`train_gpt.py:494`](/home/coder/parameter-golf/train_gpt.py#L446)

### Tier 3: Demoted ideas

These are not dead. They are just not where I would attack first.

- NorMuon
- solo SmearGate
- solo OrthoInit
- label smoothing
- MTP auxiliary loss
- compile-autotune as a score hypothesis

## Best Practical Attack Plan

If the goal is a real shot at beating accepted SOTA, I would not continue iterating from the clean root script in tiny patches.

I would do this instead:

1. Treat root `train_gpt.py` as the baseline reference and invariant scoring contract.
2. Fork from a stronger scaffold such as [`stage3/frontier_train_gpt.py`](/home/coder/parameter-golf/stage3/frontier_train_gpt.py) or a record-style script.
3. Attack in this order:

- export lane first
- eval lane second
- architecture lane third
- training-to-export bridge fourth
- data ordering fifth

Concretely:

1. Add strong export:
   - GPTQ or strong GPTQ-lite
   - low-bit mixed quantization
   - better compression
2. Add stronger scoring:
   - sliding eval
   - BOS/doc reset
   - legal score-first TTT
3. Port frontier architecture:
   - 11L
   - MLP x3
   - XSA
   - Partial RoPE
   - LN scale
   - VE and or VRL
4. Improve training-to-export bridge:
   - EMA
   - Muon WD
   - late QAT
5. Optimize budget allocation:
   - shard ordering
   - reduced warmup waste

## This Was Not Ambitious Enough

The earlier version of this document was still too anchored to:

- "what can we patch into the root script safely?"
- "what explains the gap from the readable baseline?"

That is not the same optimization problem as:

- "what has the highest probability of beating the current frontier?"

Those are different games.

If we are seriously chasing current SOTA, we should assume:

- the clean root file is not the main vehicle
- a record-style or frontier-style scaffold is the main vehicle
- the right target is not "slightly better than the root"
- the right target is "beat the strongest live no-TTT and TTT stacks"

So the aggressive framing is:

- root `train_gpt.py` is the anatomy reference
- `stage3/frontier_train_gpt.py` is the nearer scaffold
- record and frontier stacks are the actual design space

## True SOTA Attack Surface

If we care about the real frontier, the attack surfaces split cleanly into two lanes:

1. no-TTT frontier
2. TTT frontier

These should not be mentally mixed.

### Lane A: No-TTT SOTA

This is the "strong base model plus strong deployment" lane.

If the goal is to beat the no-TTT frontier, the real attack surfaces are:

#### A1. Full deployment stack

Not just "better quantization". The real target is a full deployment lane:

- GPTQ or stronger Hessian-aware quantization
- aggressive low-bit policy for the right tensor families
- compression-aware post-processing
- export-time calibration that matches the final eval path

Why this is ambitious:

- this is not a local polish
- this is a whole score-critical subsystem
- it can dominate the final delta even when training is unchanged

#### A2. XSA-all, not just XSA4

The non-ambitious version is:

- "add some XSA to the deepest few layers"

The ambitious version is:

- treat `XSA-all` as the real end-state architecture and ask whether export/compression can fund it

That means:

- architecture, export, and throughput must be co-designed
- the question is not "does XSA help?"
- the question is "how much XSA can the artifact and runtime support?"

#### A3. Stronger value path

The real frontier is not just about attention context. It is also about value-path expressivity.

That means the right ambitious targets are:

- VE-style augmentations
- VRL-style residual value reuse
- exact or stronger n-gram priors when they survive compression

The non-ambitious version is:

- solo SmearGate
- solo BigramHash

Those are now helper ideas, not the center of gravity.

#### A4. Data-budget allocation as a first-class mechanism

For the root script, shard order looks like an implementation detail.
For SOTA chasing, it is a real hypothesis:

- which shards arrive early under the 600s cap matters
- curriculum and shard ordering are part of optimization, not just data plumbing

That means the dataloader is not a support function anymore.
It is part of the model-quality path.

#### A5. Training-to-export alignment

The frontier is no longer just:

- train a good checkpoint
- quantize it at the end

The ambitious version is:

- train with the final export regime in mind
- maintain EMA or other smoothing that survives deployment
- shape the late training phase around what the quantized artifact will actually score as

So the bridge between training and export is a primary attack surface.

### Lane B: TTT SOTA

This is the "backward-looking legal adaptation" lane.

If the goal is to beat the TTT frontier, the attack surface is not "add a tiny TTT pass to the end."

That is too small.

The ambitious TTT questions are:

- how to score-first legally with minimal wasted work
- how to batch and parallelize adaptation
- how to choose the adaptation family
- how to use multiple trajectories or epoch selection
- how to keep adaptation faithful to the competition rules while maximizing gain

That implies these real attack surfaces:

#### B1. Sliding score-first infrastructure

The score-first protocol is the base requirement.

Without it:

- TTT is either illegal or underpowered

With it:

- the whole evaluation loop becomes a new optimization surface

#### B2. Better adaptation family

The non-ambitious version is:

- a small vanilla LoRA or SGD pass

The ambitious version is:

- K-LoRA or stronger low-rank adaptation on the most leverage-heavy projections
- selective freezing policy
- stronger epoch-selection logic
- document-aware adaptation schedule

#### B3. Multi-trajectory or Min-NLL selection

This is a qualitatively different level of ambition from "one TTT run and take the final weights."

The bigger question is:

- can we legally run multiple backward-looking trajectories and select the better scoring path token-wise or chunk-wise?

That is how the TTT lane stops being a small add-on and becomes a real frontier attack.

## What A Real SOTA Plan Looks Like

If we are serious, we should stop thinking in terms of "one patch at a time inside root."

The real plan is:

### Plan N: Beat no-TTT frontier

1. Start from a frontier-style scaffold, not the clean root.
2. Make export the center of the design.
3. Port the architecture toward:
   - 11L or stronger funded depth
   - MLP 3x class width
   - LeakyReLU(0.5)^2
   - XSA-all
   - Partial RoPE
   - LN scaling
   - stronger value path
4. Treat EMA plus late training policy as part of deployment, not as garnish.
5. Treat shard ordering as part of the optimization budget.
6. Only then spend time polishing helper ideas.

### Plan T: Beat TTT frontier

1. Build a legal score-first sliding eval substrate first.
2. Make adaptation faster and better:
   - more selective target modules
   - better batching
   - stronger trajectory selection
3. Optimize chunking, stride, freeze policy, and adaptation epochs jointly.
4. Consider multi-path or Min-NLL-style selection rather than a single final adapted path.

## My Updated Recommendation

If we want to be intellectually honest about the live race:

- the root script is not the main target
- a stronger scaffold should become the working file
- the ambitious attack should be on export, eval policy, and context/value architecture together

So the attack order I would endorse now is:

1. move onto a frontier-style scaffold
2. implement the strongest export lane we can support
3. push architecture toward XSA-all plus stronger value path
4. build a serious score-first eval substrate
5. decide explicitly whether we are chasing no-TTT SOTA or TTT SOTA

That is a materially more ambitious plan than "upgrade the root baseline."

## Root-Script Attack Order If We Stay In This File

If we insist on attacking the root file directly, the safest order is:

1. `LeakyReLU(0.5)^2`
2. `EMA`
3. `Muon WD`
4. `sliding eval`
5. `stronger export`
6. `11L + MLP3x`
7. `Partial RoPE + LN scale`
8. `XSA`
9. `VE and or VRL`
10. `legal score-first TTT`
11. `shard ordering`

This order is not because these are the globally best ideas in that sequence. It is because it keeps the file debuggable while moving score-critical pieces earlier.

## What To Attack Next

If I were continuing immediately after writing this document, I would attack these exact three surfaces first:

### Attack 1: Replace export

Reason:

- biggest likely score delta
- minimal disturbance to training
- most directly aligned with final judged metric

### Attack 2: Replace eval

Reason:

- cheap score lift on the same checkpoint
- needed anyway if we want to compete with modern accepted records

### Attack 3: Port the trunk toward 11L + XSA + LeakyReLU² + Partial RoPE + LN scale + EMA

Reason:

- once export and eval stop wasting score, the trunk becomes the real limiter

## Summary

The root script is not losing because its control flow is bad. It is losing because the modern frontier moved in four directions at once:

- stronger deployment quantization
- stronger scoring policy
- stronger context and value-path architecture
- stronger training-to-export alignment

So the core lesson is:

- do not spend the next cycle polishing the baseline’s small details first
- attack export, eval, and frontier trunk structure first
