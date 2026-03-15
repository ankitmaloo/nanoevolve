# Prompt Templates

These prompts are for the strategy layer. Keep them tied to the files in this folder.

## 1. Variable Mapper
Use this first.

```text
You are mapping the optimization search space for a target program.

Read:
- variables.md
- README.md
- the task context
- the mutable code regions
- the evaluator contract

Your job is to fill variables.md with:
1. the controllable levers,
2. immutable constraints,
3. likely bottlenecks,
4. what must be measured,
5. which regions of the space are currently untouched.

Also identify:
- fake-win risks,
- evaluator blind spots,
- missing evidence that should block overconfident hypotheses.

Do not propose code changes yet. Do not generate hypotheses yet.
```

## 2. Attack Surface Mapper
Use this after variables are mapped.

```text
Map the attack surfaces for this target before generating hypotheses.

Read:
- EXPERIMENTATION_STRATEGY.md
- variables.md
- README.md
- the mutable code
- the evaluator

Your job:
- identify major mutable regions,
- identify plausible bottlenecks per region,
- cross each region with change classes: structural, policy, state, schedule,
- fill the attack surface matrix in variables.md,
- pick 5 to 8 surfaces worth first-pass probing,
- label each chosen surface as scout, exploit candidate, or wildcard.

Do not generate code yet.
Do not collapse onto one region or one change class.
```

## 3. Hypothesis Generator
Use this after variables are mapped.

```text
Generate 10 distinct optimization hypotheses for this target.

Read:
- variables.md
- negative_knowledge.md
- mutation_ledger.md
- the target context

Write hypotheses.md using the required template.

Rules:
- each hypothesis must state mechanism, expected win, risk, and evidence needed,
- each hypothesis must state bottleneck attacked, cheapest test, and disproof signal,
- prefer mechanism-level code changes over pure scalar retuning,
- allow invention of a new state variable or schedule if there is a causal story for why it should matter,
- spread them across different families,
- avoid repeating anything blocked by negative_knowledge.md,
- avoid producing hypotheses that occupy the same mutation neighborhood unless you explicitly mark one as an ablation,
- optimize for search coverage, not just plausibility.
```

## 4. Hypothesis Pruner
Use this after the initial 10 are written.

```text
Prune the initial hypothesis list.

Read:
- variables.md
- hypotheses.md
- negative_knowledge.md
- mutation_ledger.md

Your job:
- remove redundant hypotheses,
- kill vague hypotheses with no cheap falsification path,
- mark strong survivors as shortlisted,
- kill or merge hypotheses that occupy the same mutation neighborhood,
- explain why each killed hypothesis was eliminated.

Do not add new hypotheses in this step unless one is needed to replace an obviously invalid assumption.
```

## 5. Gap Finder
Use this after initial pruning.

```text
Review variables.md, hypotheses.md, and the shortlisted set.

Find major search-space gaps:
- important bottlenecks with no live hypothesis,
- overconcentration in one family,
- missing high-information probes,
- missing robustness or holdout-oriented probes.

Write the missing coverage into gaps.md and propose 2 to 3 additional hypotheses only if they close real gaps.
```

## 6. Portfolio Selector
Use this before implementation.

```text
Select the top 5 hypotheses for the next loop.

Read:
- hypotheses.md
- gaps.md
- negative_knowledge.md
- mutation_ledger.md

Write portfolio.md.

Selection criteria:
- upside,
- feasibility,
- distinctness,
- information gain,
- transferability.

Do not choose five near-duplicates. Explain why each selected hypothesis is in the portfolio now.
Write the overlap check explicitly for each slot.
Use an explicit mix of scout, exploit, and wildcard roles.
```

## 7. Candidate Implementer
Use this once per portfolio slot.

```text
Implement exactly one candidate for the assigned portfolio slot.

Read:
- portfolio.md
- hypotheses.md
- negative_knowledge.md
- mutation_ledger.md
- relevant task context

Rules:
- tie the change directly to the named hypothesis,
- preserve invariants,
- avoid move families already marked as failed unless the hypothesis explicitly states what is different,
- prefer a coherent test of the mechanism over broad opportunistic edits,
- if a constant becomes adaptive, explain the governing signal and why that signal is informative,
- if you introduce a new state variable, explain what hidden regime it is trying to capture,
- do not collide with another active slot's mutation neighborhood unless this is a labeled ablation,
- do not claim success from one narrow slice if holdouts or stability regress.

At the end, state:
- what changed,
- what hypothesis was tested,
- what evidence should confirm or refute it.
```

## 8. Evaluator Skeptic
Use this after results appear and before final conclusions.

```text
Review the latest candidate results with skepticism.

Read:
- variables.md
- portfolio.md
- loop results
- loop_log.md

Your job:
- identify fake wins,
- identify overfitting to one slice,
- identify variance or evaluator noise that weakens conclusions,
- state which candidate results are trustworthy, tentative, or misleading.

Prefer disconfirming weak conclusions over celebrating small wins.
```

## 9. Postmortem Analyst
Use this after evaluation.

```text
Analyze one completed loop.

Read:
- portfolio.md
- hypotheses.md
- loop results
- negative_knowledge.md
- mutation_ledger.md

Write:
- loop_log.md updates,
- promoted hypotheses,
- killed hypotheses,
- child hypotheses,
- durable negative knowledge.

Also update:
- which mutation neighborhoods should be retired,
- which can remain active,
- which can be reopened only under a changed assumption.

Focus on causal learning, not narrative summary.
```

## 10. Cold Start Handoff
Use this when handing the folder to another LLM.

```text
You are taking over an AlphaEvolve-style search effort.

Before making any code change:
1. read README.md,
2. read IMPLEMENTER_SPEC.md,
3. read EXPERIMENTATION_STRATEGY.md,
4. read variables.md,
5. read negative_knowledge.md,
6. read mutation_ledger.md,
7. read hypotheses.md,
8. read gaps.md,
9. read portfolio.md,
10. read loop_log.md,
11. inspect the target code and evaluator.

Then answer, in your own words:
- what is being optimized,
- what constitutes a real win,
- what constitutes a fake win,
- what move families are currently alive,
- what move families are currently disfavored,
- which mutation neighborhoods are already occupied or retired,
- what attack surfaces are still underexplored,
- whether the next loop should invent a new rule, state variable, or schedule rather than retune an existing knob,
- what the next loop should test first and why.

Only after that may you modify code.
```

## 11. Search Conductor
Use this to run the whole doctrine.

```text
You are the search conductor for an AlphaEvolve-style optimization loop.

Operate only through the doctrine in this folder:
- README.md
- IMPLEMENTER_SPEC.md
- EXPERIMENTATION_STRATEGY.md
- variables.md
- hypotheses.md
- gaps.md
- portfolio.md
- negative_knowledge.md
- mutation_ledger.md
- loop_log.md
- prompt_templates.md

Run this cycle:
1. map variables,
2. map attack surfaces,
3. generate 10 hypotheses,
4. prune weak or redundant ones,
5. identify gaps,
6. add gap-filling hypotheses,
7. select the top 5 with explicit scout/exploit/wildcard roles,
8. verify that the five selected slots do not overlap in mutation neighborhood,
9. implement and evaluate one candidate per top hypothesis,
10. update the mutation ledger,
11. run evaluator skepticism before drawing conclusions,
12. record lessons,
13. update negative knowledge,
14. propose the next loop.

Your objective is not just to find one win. Your objective is to improve the search policy itself over time.
```
