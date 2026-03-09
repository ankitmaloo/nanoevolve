"""A* routing target (harder, real-world-ish) for AlphaEvolve MVP.

Use case inspiration: vehicle routing, robot navigation, game pathfinding.
The evolvable parts are priority and tie-break scoring formulas.
"""

# EVOLVE-BLOCK-START
def priority_score(features):
    g_cost = features["g_cost"]
    h_cost = features["h_cost"]

    # Available feature keys for evolution:
    # g_cost, h_cost, turn_penalty, crowding, progress, step_cost, escape_routes
    # Higher score means node is preferred for expansion.
    # Basic A* baseline: minimize f = g + h.
    return -(g_cost + h_cost)


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def tie_break_priority(features):
    return 0.0


# EVOLVE-BLOCK-END


def evaluate():
    return {"status": "Use mvp.evaluator.Evaluator.evaluate(program_source)."}
