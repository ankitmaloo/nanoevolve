"""Seed evolvable program for AlphaEvolve MVP.

The evaluator imports this source and expects:
- heuristic_score(required, free)
- optional tie_break_score(required, free)

Both evolve blocks are intended mutation targets.
"""

# EVOLVE-BLOCK-START
def heuristic_score(required, free):
    cpu_residual = required["cpu"] / max(free["cpu"], 1e-9)
    mem_residual = required["mem"] / max(free["mem"], 1e-9)
    return -1.0 * (cpu_residual + mem_residual)


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def tie_break_score(required, free):
    return free["cpu"] + free["mem"]


# EVOLVE-BLOCK-END


def evaluate():
    """Placeholder. The real evaluation entrypoint is mvp.evaluator.Evaluator.evaluate."""
    return {"status": "use mvp.evaluator.Evaluator.evaluate(program_source)"}
