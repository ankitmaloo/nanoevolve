# Stage 5 Context — AlphaEvolve Search from the Stage 4 Frontier

## What Stage 5 Is

Stage 5 is the first **AlphaEvolve-style search loop** after the Stage 4 production-faithful correction and the Stage 4 extension runs.

The goal is not to retune one known winner. The goal is to generate a **broad, mechanism-first search slate** from the real frontier:

- one promoted parent baseline,
- one pure production baseline,
- trusted wins,
- confounded wins that must be reimplemented faithfully,
- explicit negative knowledge,
- explicit open attack surfaces,
- explicit fake-win risks.

This context file is the handoff document a strategy model should read before generating the next 20 hypotheses.

## What Is Being Optimized

We are improving the **production NanoChat MuonAdamW optimizer** under the **real production training loop**:

- Model: NanoChat GPT-2 (12 layers, 768 dim, 6 heads)
- Dataset: `karpathy/climbmix-400b-shuffle`
- Batch: 1024 tokens (`device_batch=2`, `seq_len=512`)
- Hardware: B200 GPU
- Metric: validation BPB
- Runtime target: zero or near-zero overhead relative to production

The search is over **mechanism changes**, especially:

- new schedules,
- new state variables,
- new feedback laws,
- cross-path coordination rules,
- faithful ports of promising mechanisms.

Weak Stage 5 proposals are pure coefficient nudges with no causal story.

## The Two Baselines Stage 5 Must Keep

Every Stage 5 portfolio should include **both**:

1. **Promoted parent baseline**
   - Current promoted parent: `H60_solo`
   - Why: it is the highest-confidence frontier winner right now
   - Evidence:
     - Stage 4 5k: `1.508119` vs production `1.539510` (`+0.031391`)
     - Stage 4 extension 5k: `1.479033` vs batch baseline `1.531135` (`+0.052102`)
     - Stage 4 extension 20k: `1.420609` vs production `1.456351` (`+0.035742`)

2. **Pure production baseline**
   - Unmodified production optimizer and schedules
   - Why: all deltas must still be interpretable against the real deployed system

This is mandatory. A Stage 5 candidate should report:

- delta vs promoted parent (`H60_solo`)
- delta vs pure production baseline

That prevents us from mistaking “better than production” for “better than the current frontier.”

## Optional Secondary Parent

If Stage 5 needs a second parent to preserve another live family, use:

- `H73_solo`

Why:

- It is a trusted AdamW-path win on a different surface than H60
- Stage 4 5k: `+0.045924`
- Stage 4 extension 5k: `+0.031266`
- It remains plausible that `H60` and `H73` compound cleanly, but that has not been validated at long horizon

Do **not** use H64 as a promoted parent until it is faithfully ported into the extension runner.

## Production Baseline Regime

All Stage 5 runs must preserve the Stage 4 production-faithful evaluator regime:

- LR: linear warmdown over the last 50% to 0
- Muon momentum: `0.85 -> 0.95` over 300 steps
- Muon weight decay: linear decay to 0
- AdamW path gets LR schedule only unless explicitly mutated

If a candidate is measured under a weaker or different baseline, its result is not comparable.

## What Is Trusted Right Now

### Trusted Positive Findings

#### H60 — Muon beta2 warmup `0.8 -> 0.95`

- Surface: Muon second moment / variance adaptation
- Trust level: **highest**
- Why trusted:
  - wins in Stage 4,
  - wins in Stage 4 extension 5k,
  - wins strongly at 20k,
  - improves late train loss as well as validation BPB

#### H73 — AdamW epsilon schedule `1e-6 -> 1e-10`

- Surface: AdamW denominator / step-size stabilization
- Trust level: **high**
- Why trusted:
  - Stage 4 runner used the real production AdamW fused path
  - Stage 4 extension still shows a clear 5k gain
- Missing: long-horizon 20k confirmation

#### H71 — AdamW beta1 warmup `0.7 -> 0.8`

- Surface: AdamW first moment
- Trust level: **medium**
- Why:
  - positive in Stage 4
  - slightly negative solo in Stage 4 extension
- Interpretation:
  - may be conditional or interaction-dependent
  - should not be killed entirely, but should not be promoted alone

### Trusted Negative Findings

- Trust ratio on production Muon is harmful in the production-faithful regime
- Embedding weight decay `0.01` is harmful
- Scalar LR warmup is harmful
- Non-monotonic momentum overshoot is worse than simple monotonic schedules

## What Is Confounded Or Not Yet Trusted

### H64 — Decoupled Nesterov blend schedule

Stage 4 H64 was the strongest single win:

- `1.476537` vs `1.539510` at 5k (`+0.062973`)

But the Stage 4 extension runner did **not** implement H64 faithfully.

Stage 4 H64 mechanism:

- decouple Nesterov blend from Muon momentum
- schedule blend independently

Stage 4 extension approximation:

- overwrite `group["momentum"]`

That is not the same algorithm.

Therefore:

- Stage 4 H64 remains a live and important family
- Stage 4 extension results that include H64 are **provisional**
- A faithful H64 port is a top Stage 5 path

## What The 20k Run Taught Us

The 20k H60 follow-up is the most important new evidence.

### Observed Curve Shape

H60 is:

- worse early,
- still worse through 5k,
- close to tied around 10k-11k,
- clearly better from roughly 15k onward,
- strongly better at 20k

Selected checkpoints:

- step 5000: baseline `1.678841`, H60 `1.742318`, delta `-0.063476`
- step 10000: baseline `1.662976`, H60 `1.672837`, delta `-0.009861`
- step 15000: baseline `1.559737`, H60 `1.516922`, delta `+0.042815`
- step 16000: baseline `1.551249`, H60 `1.492287`, delta `+0.058962`
- step 20000: baseline `1.456351`, H60 `1.420609`, delta `+0.035742`

### Why This Likely Matters

The best current explanation is:

- H60 is not an “always better” mutation
- it is a **phase-selective** mutation
- lower beta2 early creates a more responsive second-moment estimate
- this may be noisier and worse short-term
- but it appears to produce a better-conditioned optimizer state for the warmdown regime

This means Stage 5 should not over-index on 5k-only rankings for schedule families.

## What A Real Win Looks Like In Stage 5

A real Stage 5 win should satisfy most of:

- beats pure production baseline
- beats promoted parent baseline (`H60_solo`)
- stays production-faithful
- has a causal story, not just a numerical delta
- does not rely on a confounded implementation shortcut
- preferably shows support in both val BPB and train-loss dynamics
- does not introduce meaningful step-time overhead

## What Counts As A Fake Win

- wins only against production but loses to the promoted parent
- wins only at 5k while clearly degrading the later regime
- depends on an implementation shortcut that changes the original mechanism
- overlaps an already-tested neighborhood without explaining what changed
- retunes a constant without introducing a new mechanism or new evidence
- improves one path by silently violating production schedule semantics

## Stage 5 Search State: What Families Are Alive

### Alive Family A — Muon second-moment control

Why alive:

- H60 is the strongest trustworthy frontier win
- the 20k result suggests phase-aware second-moment behavior is a real lever

Examples of legitimate descendants:

- longer beta2 warmup
- warmdown-aware beta2 schedule
- two-phase beta2 schedule
- loss-reactive beta2
- per-shape or per-matrix-family beta2

### Alive Family B — AdamW denominator / stability control

Why alive:

- H73 is a strong and distinct win
- it attacks a different optimizer path than H60

Examples of legitimate descendants:

- longer-horizon epsilon schedules
- phase-aware epsilon schedules
- per-group epsilon schedules
- epsilon tied to LR warmdown or second-moment maturity

### Alive Family C — AdamW momentum adaptation

Why alive:

- H71 had at least one positive loop
- still plausible as a helper mutation even if weak solo

Examples of legitimate descendants:

- per-group beta1 schedules
- late-phase beta1 reduction instead of early warmup
- beta1 schedules conditioned on Muon volatility

### Alive Family D — Faithful mechanism ports and evaluator fidelity

Why alive:

- H64 is still one of the strongest raw ideas
- current extension evidence is muddied by a non-faithful port

Examples:

- faithful H64 port
- faithful H64 + H60
- faithful H64 + H73
- faithful H64 + H60 + H73 with dual baselines

### Alive Family E — Cross-path coordination

Why alive:

- Muon and AdamW still operate largely independently
- H60’s late-phase win suggests optimizer-state interaction across phases matters

Examples:

- AdamW schedules triggered by Muon update statistics
- Muon second-moment schedule triggered by warmdown onset
- shared phase variable controlling both paths
- path-specific schedules that deliberately diverge by training phase

## What Families Are Disfavored

Do not spend many Stage 5 slots on:

- trust-ratio revival without a clearly changed assumption
- embedding weight decay revival without a new causal mechanism
- scalar warmup variants that only rephrase H78
- non-monotonic “clever” schedules without strong new evidence
- pure coefficient re-tuning of already-live mechanisms

One explicit ablation or revisit is fine. A whole portfolio cluster is not.

## What The Strategy Model Must Know To Produce Distinct Paths

The next 20 hypotheses must not collapse into “10 versions of H60.”

A good Stage 5 hypothesis set should spread across:

1. **Faithful ports / implementation-quality restorations**
   - Example: true H64 port into the extension runner

2. **Muon late-phase control**
   - warmdown-aware beta2
   - multi-phase second moment
   - per-shape second moment

3. **AdamW long-horizon control**
   - longer epsilon schedules
   - late-phase epsilon or beta1 schedules
   - per-group AdamW schedules

4. **Cross-path coordination**
   - Muon-to-AdamW feedback
   - shared phase state
   - volatility-aware AdamW adaptation

5. **Stateful control**
   - new tracked phase marker
   - loss-improvement EMA
   - optimizer-volatility estimator
   - schedule keyed to detected regime changes

6. **High-information wildcards**
   - a risky but diagnostic probe
   - a hypothesis that could kill a whole family quickly

## Suggested Distribution For The Initial 20 Hypotheses

Not mandatory, but this is the intended spread:

- 4 around Muon second-moment / H60 descendants
- 4 around AdamW epsilon / H73 descendants
- 3 around AdamW momentum / H71 descendants
- 3 around faithful H64 port and H64 compounds
- 4 around cross-path or stateful coordination
- 2 high-risk wildcards

That spread gives real frontier coverage instead of local churn.

## Distinctness Rules For The 20 Hypotheses

At least half of the 20 should differ on **two or more** of:

- bottleneck attacked
- optimizer path attacked (Muon vs AdamW vs cross-path)
- change class (schedule vs state vs policy vs structural)
- training phase targeted (early vs warmdown vs late vs adaptive)
- evidence pattern expected

Do not let more than 3 hypotheses occupy the exact same mutation neighborhood unless:

- one is the main proposal,
- one is an ablation,
- one is a sharper or broader child.

## Five Set-Level Gaps The Model Should Keep In Mind

These are not yet the final gap pass. They are the likely Stage 5 gap categories.

1. **No faithful H64 path in the current extension runner**
   - This is a missing capability, not just a missing hypothesis.

2. **No long-horizon test of H73**
   - AdamW-path late behavior is still undermeasured.

3. **No mechanism keyed explicitly to warmdown onset**
   - The 20k H60 result strongly suggests the warmdown transition matters.

4. **No per-group or per-shape schedule specialization**
   - Muon still uses one global beta2 schedule across all matrix groups.

5. **No explicit shared phase/state variable coordinating Muon and AdamW**
   - The optimizer still lacks a mechanism for detecting and reacting to training regime shifts.

## Recommended Immediate Stage 5 Workflow

1. Read:
   - `alphaevolve/md/README.md`
   - `alphaevolve/md/prompt_templates.md`
   - `runs/enigma_stage4_postmortem.json`
   - `runs/enigma_stage4ext_postmortem.json`
   - this file

2. Generate **20 hypotheses**
   - wide coverage
   - mechanism-first
   - not all local descendants

3. Prune to **10**
   - remove overlap
   - remove low-information knob tweaks

4. Run a **5-gap pass**
   - sample from the whole space, not just current winners

5. Add the best gap-fillers, then prune to **8 live candidates**

6. Build the evaluation slate:
   - promoted parent baseline: `H60_solo`
   - optional second parent: `H73_solo` if needed
   - pure production baseline
   - remaining candidate slots from the top 8

## File References

- Stage 4 context: `enigma/stage4_context.md`
- Stage 4 postmortem: `runs/enigma_stage4_postmortem.json`
- Stage 4 extension postmortem: `runs/enigma_stage4ext_postmortem.json`
- Extension runner: `enigma/run_stage5.py`
- Extension patch module: `enigma/stage5_patch.py`
- Compound batch artifacts: `runs/enigma_s4ext_compound/`
- 20k H60 artifacts: `runs/enigma_s4ext_20k_h60/`

## Bottom Line

The Stage 5 frontier is:

- **promote H60 as the current trustworthy parent**
- **keep H73 alive as the strongest distinct second family**
- **repair H64 fidelity before using its extension-batch results**
- **treat warmdown / late-phase behavior as a first-class search target**
- **generate hypotheses across Muon, AdamW, and cross-path coordination rather than overfitting to one winning neighborhood**
