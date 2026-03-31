`val_bpb` is the model’s held-out next-token prediction loss on the validation split, expressed as `bits per byte` of original text rather than `nats per token`.

In this repo, the exact quantity is:

```text
val_bpb = sum_t -log p(y_t | x_<t) / (ln 2 * sum_t bytes(y_t))
```

That is implemented in [nanochat/nanochat/loss_eval.py:9](nanoevolve/nanochat/nanochat/loss_eval.py#L9). The key details are:

- It sums token-level negative log-likelihood over validation targets.
- It divides by the total UTF-8 byte count of those target tokens, not by token count.
- Special tokens like BOS are excluded by giving them `0` bytes in [nanochat/scripts/tok_train.py:76](nanoevolve/nanochat/scripts/tok_train.py#L76).
- Ignored/masked targets also do not contribute.

So conceptually: `val_bpb` is “how many bits the model needs, on average, to encode one byte of unseen validation text.” Lower is better.

One important nuance: in training it is not a full-pass over the entire validation set. It is a sampled estimate over `args.eval_tokens` tokens from the validation loader, as wired in [nanochat/scripts/base_train.py:334](nanoevolve/nanochat/scripts/base_train.py#L334) and [nanochat/scripts/base_train.py:416](nanoevolve/nanochat/scripts/base_train.py#L416).

Why use this instead of plain val loss? Because it is much more comparable across different tokenizers and vocab sizes. The README states that directly in [nanochat/README.md:73](nanoevolve/nanochat/README.md#L73).

If you want, I can also explain how `val_bpb` relates numerically to ordinary `val_loss` in the logs.