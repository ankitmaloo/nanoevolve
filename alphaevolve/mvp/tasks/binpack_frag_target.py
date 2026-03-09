"""Fragmentation-aware bin-packing heuristic target for evolution.

This is the recommended first target to optimize because:
- it is fast to evaluate,
- it has clear trade-offs (fit vs future flexibility),
- small formula changes produce meaningful behavior changes.
"""

# EVOLVE-BLOCK-START
def heuristic_score(required, free):
    cpu_fit = required["cpu"] / max(free["cpu"], 1e-9)
    mem_fit = required["mem"] / max(free["mem"], 1e-9)

    residual_cpu = max(0.0, free["cpu"] - required["cpu"])
    residual_mem = max(0.0, free["mem"] - required["mem"])

    residual_total = residual_cpu + residual_mem + 1e-9
    residual_imbalance = abs(residual_cpu - residual_mem) / residual_total

    # Tiny residuals are often hard to use later (fragmentation signal).
    tiny_hole_penalty = 0.0
    if 0.0 < residual_cpu < 1.5:
        tiny_hole_penalty += 1.0
    if 0.0 < residual_mem < 1.5:
        tiny_hole_penalty += 1.0

    return -1.0 * (cpu_fit + mem_fit) - 0.40 * residual_imbalance - 0.15 * tiny_hole_penalty


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def tie_break_score(required, free):
    residual_cpu = max(0.0, free["cpu"] - required["cpu"])
    residual_mem = max(0.0, free["mem"] - required["mem"])
    residual_sum = residual_cpu + residual_mem
    residual_gap = abs(residual_cpu - residual_mem)
    return -0.5 * residual_sum - 0.5 * residual_gap


# EVOLVE-BLOCK-END


def evaluate():
    """Placeholder for compatibility with paper-style evaluate signature."""
    return {"status": "Use mvp.evaluator.Evaluator.evaluate(program_source)."}
