# Stage 5 Singles Rerun — Hypotheses

## Carryovers

1. `none`
   - role: pure production baseline
   - why kept: all deltas still need to be interpretable against the deployed optimizer

2. `H73_solo`
   - role: prior single winner carryover
   - mechanism: global AdamW epsilon schedule
   - why kept: best prior single at 20k

3. `H64_H60_H73`
   - role: accidental frontier composite carryover
   - why kept: best raw result from the previous batch, useful as a frontier sentinel

## New Singles

4. `H531_raw_grad_wd`
   - mechanism: Muon cautious weight decay mask uses raw gradient sign instead of post-orthogonalized update sign
   - hypothesis: weight-decay gating should key off the original descent direction, not the transformed Muon update

5. `H532_muon_vreset`
   - mechanism: at warmdown onset, shrink Muon second-momentum buffers once
   - hypothesis: H60’s late win may come from entering warmdown with less stale variance memory

6. `H533_shape_beta2`
   - mechanism: rectangular Muon groups get H60-style beta2 warmup; square-ish groups stay at baseline beta2
   - hypothesis: variance adaptation demand differs meaningfully by matrix aspect ratio

7. `H534_ns_stage`
   - mechanism: Muon uses lower orthogonalization depth early, then restores full depth
   - hypothesis: early training may benefit from a less rigid orthogonalization transform before late full stabilization

8. `H535_embed_eps`
   - mechanism: only `embedding` and `value_embeds` receive the epsilon decay schedule
   - hypothesis: the H73 win may primarily be an embedding-path denominator effect rather than a whole-AdamW effect

9. `H536_x0_beta1_late`
   - mechanism: `x0_lambdas` beta1 decays during warmdown
   - hypothesis: late x0 blending should track recent gradients faster once LR starts falling

10. `H537_embed_mom_reset`
    - mechanism: at warmdown onset, shrink AdamW first-moment buffers for `embedding` and `value_embeds`
    - hypothesis: embedding-like groups may carry too much stale inertia into the late regime

11. `H538_seed_vsq`
    - mechanism: initialize `exp_avg_sq` for `embedding` and `value_embeds` from the first observed gradient
    - hypothesis: embedding-like groups may benefit from a less abrupt denominator bootstrap than zero-init

## Explicit Rejections From The Prior Batch

- `H504_beta2_phase2`
  - rejected: clear loser at 20k and too close to the H60 lineage without new enough mechanism content

- `H517_shared_phase`
  - rejected: interesting curve shape but still lost to production; too underpowered to spend another singles slot on immediately
