from __future__ import annotations

import heapq
import math
import random
from collections import deque
from copy import deepcopy

from mvp.types import EvaluationResult, StageResult


SCENARIOS = [
    {
        "machines": [
            {"cpu": 8.0, "mem": 16.0},
            {"cpu": 8.0, "mem": 16.0},
            {"cpu": 4.0, "mem": 8.0},
        ],
        "jobs": [
            {"cpu": 2.0, "mem": 4.0},
            {"cpu": 3.0, "mem": 2.0},
            {"cpu": 1.0, "mem": 1.0},
            {"cpu": 4.0, "mem": 8.0},
            {"cpu": 2.0, "mem": 3.0},
            {"cpu": 1.0, "mem": 2.0},
        ],
    },
    {
        "machines": [
            {"cpu": 10.0, "mem": 10.0},
            {"cpu": 6.0, "mem": 14.0},
            {"cpu": 12.0, "mem": 12.0},
        ],
        "jobs": [
            {"cpu": 5.0, "mem": 4.0},
            {"cpu": 2.0, "mem": 5.0},
            {"cpu": 3.0, "mem": 3.0},
            {"cpu": 4.0, "mem": 4.0},
            {"cpu": 2.0, "mem": 2.0},
        ],
    },
    {
        "machines": [
            {"cpu": 12.0, "mem": 24.0},
            {"cpu": 12.0, "mem": 24.0},
            {"cpu": 6.0, "mem": 12.0},
        ],
        "jobs": [
            {"cpu": 6.0, "mem": 8.0},
            {"cpu": 4.0, "mem": 8.0},
            {"cpu": 3.0, "mem": 4.0},
            {"cpu": 2.0, "mem": 6.0},
            {"cpu": 5.0, "mem": 7.0},
            {"cpu": 1.0, "mem": 1.0},
        ],
    },
    {
        "machines": [
            {"cpu": 7.0, "mem": 9.0},
            {"cpu": 7.0, "mem": 9.0},
            {"cpu": 14.0, "mem": 18.0},
        ],
        "jobs": [
            {"cpu": 2.0, "mem": 2.0},
            {"cpu": 2.0, "mem": 4.0},
            {"cpu": 3.0, "mem": 3.0},
            {"cpu": 5.0, "mem": 7.0},
            {"cpu": 2.0, "mem": 1.0},
            {"cpu": 1.0, "mem": 3.0},
            {"cpu": 4.0, "mem": 4.0},
        ],
    },
]


def _generate_astar_scenarios(seed: int = 11, count: int = 12) -> list[dict[str, object]]:
    rng = random.Random(seed)
    scenarios: list[dict[str, object]] = []

    while len(scenarios) < count:
        rows = rng.randint(20, 30)
        cols = rng.randint(20, 30)
        obstacle_prob = rng.uniform(0.18, 0.34)

        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if rng.random() < obstacle_prob:
                    grid[r][c] = 1

        terrain = [[(1.0 + 2.0 * rng.random()) for _ in range(cols)] for _ in range(rows)]
        for _ in range(rng.randint(2, 6)):
            center_r = rng.randrange(rows)
            center_c = rng.randrange(cols)
            radius = rng.randint(2, 5)
            bump = rng.uniform(0.5, 1.8)
            for r in range(max(0, center_r - radius), min(rows, center_r + radius + 1)):
                for c in range(max(0, center_c - radius), min(cols, center_c + radius + 1)):
                    if grid[r][c] == 1:
                        continue
                    if (r - center_r) ** 2 + (c - center_c) ** 2 <= radius**2:
                        terrain[r][c] = min(5.0, terrain[r][c] + bump)

        start, goal = _sample_far_points(rng, grid)
        if start is None or goal is None:
            continue
        grid[start[0]][start[1]] = 0
        grid[goal[0]][goal[1]] = 0
        terrain[start[0]][start[1]] = 1.0
        terrain[goal[0]][goal[1]] = 1.0

        opt = _dijkstra_shortest_path_cost(grid, terrain, start, goal)
        if opt is None:
            continue

        scenarios.append(
            {
                "grid": grid,
                "terrain": terrain,
                "start": start,
                "goal": goal,
                "optimal_cost": opt,
            }
        )

    return scenarios


class Evaluator:
    def __init__(self, stage1_threshold: float = 0.35) -> None:
        self.stage1_threshold = stage1_threshold
        self.astar_train_scenarios = _generate_astar_scenarios(seed=11, count=14)
        self.astar_holdout_scenarios = _generate_astar_scenarios(seed=29, count=10)
        self.astar_stage1_scenarios = self.astar_train_scenarios[:4]

    def evaluate(self, program_source: str) -> EvaluationResult:
        namespace: dict[str, object] = {}
        stage_results: list[StageResult] = []

        try:
            compiled = compile(program_source, "<candidate_program>", "exec")
            exec(compiled, namespace, namespace)
        except Exception as exc:
            return EvaluationResult(
                valid=False,
                aggregate_score=-1.0,
                failure_reasons=[f"Program compile/exec failed: {exc}"],
                stage_results=[
                    StageResult(name="compile", passed=False, message=f"compile failed: {exc}")
                ],
            )

        if "priority_score" in namespace and callable(namespace["priority_score"]):
            return self._evaluate_astar(namespace["priority_score"], namespace.get("tie_break_priority"))

        if "heuristic_score" in namespace and callable(namespace["heuristic_score"]):
            return self._evaluate_binpack(namespace["heuristic_score"], namespace.get("tie_break_score"))

        return EvaluationResult(
            valid=False,
            aggregate_score=-1.0,
            failure_reasons=[
                "Missing expected interface: provide either priority_score(features) for A* or heuristic_score(required, free) for bin-packing."
            ],
            stage_results=[
                StageResult(name="compile", passed=False, message="expected function interface not found")
            ],
        )

    def _evaluate_astar(self, priority_score, tie_break_priority) -> EvaluationResult:
        stage_results: list[StageResult] = []

        try:
            stage1_metrics, stage1_diag = _run_astar_scenarios(
                priority_score,
                tie_break_priority,
                self.astar_stage1_scenarios,
            )
            stage1_passed = stage1_metrics["solved_ratio"] >= 0.50 and stage1_metrics["path_quality"] >= 0.60
            stage_results.append(
                StageResult(
                    name="stage1_quick_filter",
                    passed=stage1_passed,
                    metrics=stage1_metrics,
                    message="A* quick gate",
                )
            )
            if not stage1_passed:
                return EvaluationResult(
                    valid=False,
                    aggregate_score=stage1_metrics["aggregate_score"],
                    metrics=stage1_metrics,
                    failure_reasons=[
                        "A* stage1 threshold not met: "
                        f"solved_ratio={stage1_metrics['solved_ratio']:.3f}, "
                        f"path_quality={stage1_metrics['path_quality']:.3f}"
                    ],
                    stage_results=stage_results,
                    diagnostics={"stage1": stage1_diag},
                )

            train_metrics, train_diag = _run_astar_scenarios(
                priority_score,
                tie_break_priority,
                self.astar_train_scenarios,
            )
            stage_results.append(
                StageResult(
                    name="stage2_train_eval",
                    passed=True,
                    metrics=train_metrics,
                    message="A* train scenario suite",
                )
            )

            holdout_metrics, holdout_diag = _run_astar_scenarios(
                priority_score,
                tie_break_priority,
                self.astar_holdout_scenarios,
            )
            stage_results.append(
                StageResult(
                    name="stage3_holdout_eval",
                    passed=True,
                    metrics=holdout_metrics,
                    message="A* holdout generalization suite",
                )
            )

            aggregate_score = 0.65 * train_metrics["aggregate_score"] + 0.35 * holdout_metrics["aggregate_score"]
            metrics = {
                "solved_ratio": 0.65 * train_metrics["solved_ratio"] + 0.35 * holdout_metrics["solved_ratio"],
                "path_quality": 0.65 * train_metrics["path_quality"] + 0.35 * holdout_metrics["path_quality"],
                "expansion_efficiency": 0.65 * train_metrics["expansion_efficiency"]
                + 0.35 * holdout_metrics["expansion_efficiency"],
                "route_smoothness": 0.65 * train_metrics["route_smoothness"] + 0.35 * holdout_metrics["route_smoothness"],
                "train_aggregate_score": train_metrics["aggregate_score"],
                "holdout_aggregate_score": holdout_metrics["aggregate_score"],
                "aggregate_score": aggregate_score,
            }

            return EvaluationResult(
                valid=True,
                aggregate_score=aggregate_score,
                metrics=metrics,
                stage_results=stage_results,
                diagnostics={
                    "stage1": stage1_diag,
                    "stage2_train": train_diag,
                    "stage3_holdout": holdout_diag,
                },
            )
        except Exception as exc:
            stage_results.append(StageResult(name="runtime", passed=False, message=f"runtime failure: {exc}"))
            return EvaluationResult(
                valid=False,
                aggregate_score=-1.0,
                failure_reasons=[f"A* runtime evaluation failed: {exc}"],
                stage_results=stage_results,
            )

    def _evaluate_binpack(self, heuristic_score, tie_break_score) -> EvaluationResult:
        stage_results: list[StageResult] = []

        try:
            stage1_metrics, stage1_diag = _run_binpack_scenarios(heuristic_score, tie_break_score, SCENARIOS[:2])
            stage1_passed = stage1_metrics["placed_jobs_ratio"] >= self.stage1_threshold
            stage_results.append(
                StageResult(
                    name="stage1_quick_filter",
                    passed=stage1_passed,
                    metrics=stage1_metrics,
                    message="quick gate on easy subset",
                )
            )
            if not stage1_passed:
                return EvaluationResult(
                    valid=False,
                    aggregate_score=stage1_metrics["aggregate_score"],
                    metrics=stage1_metrics,
                    failure_reasons=[
                        f"Stage1 threshold not met: placed_jobs_ratio={stage1_metrics['placed_jobs_ratio']:.3f} < {self.stage1_threshold:.3f}"
                    ],
                    stage_results=stage_results,
                    diagnostics={"stage1": stage1_diag},
                )

            full_metrics, full_diag = _run_binpack_scenarios(heuristic_score, tie_break_score, SCENARIOS)
            stage_results.append(
                StageResult(
                    name="stage2_full_eval",
                    passed=True,
                    metrics=full_metrics,
                    message="full dataset evaluation",
                )
            )
            return EvaluationResult(
                valid=True,
                aggregate_score=full_metrics["aggregate_score"],
                metrics=full_metrics,
                stage_results=stage_results,
                diagnostics={"stage1": stage1_diag, "stage2": full_diag},
            )
        except Exception as exc:
            stage_results.append(StageResult(name="runtime", passed=False, message=f"runtime failure: {exc}"))
            return EvaluationResult(
                valid=False,
                aggregate_score=-1.0,
                failure_reasons=[f"Runtime evaluation failed: {exc}"],
                stage_results=stage_results,
            )


def _run_astar_scenarios(priority_score, tie_break_priority, scenarios):
    solved = 0
    quality_acc = 0.0
    efficiency_acc = 0.0
    smooth_acc = 0.0
    total = len(scenarios)
    expanded_total = 0

    for sc in scenarios:
        result = _astar_route(
            sc["grid"],
            sc["terrain"],
            sc["start"],
            sc["goal"],
            priority_score,
            tie_break_priority,
        )
        expanded_total += result["expanded_nodes"]
        if result["solved"]:
            solved += 1
            quality_acc += sc["optimal_cost"] / max(1e-9, result["path_cost"])
            smooth_acc += 1.0 - min(1.0, result["turns"] / max(1, result["path_len"] - 1))

        rows = len(sc["grid"])
        cols = len(sc["grid"][0])
        free_cells = sum(1 for row in sc["grid"] for cell in row if cell == 0)
        efficiency_acc += 1.0 - min(1.0, result["expanded_nodes"] / max(1.0, free_cells * 2.5))

    solved_ratio = solved / max(1, total)
    path_quality = quality_acc / max(1, solved)
    expansion_efficiency = efficiency_acc / max(1, total)
    route_smoothness = smooth_acc / max(1, solved)

    aggregate_score = (
        0.45 * solved_ratio
        + 0.30 * path_quality
        + 0.15 * expansion_efficiency
        + 0.10 * route_smoothness
    )

    metrics = {
        "solved_ratio": solved_ratio,
        "path_quality": path_quality,
        "expansion_efficiency": expansion_efficiency,
        "route_smoothness": route_smoothness,
        "aggregate_score": aggregate_score,
    }
    diagnostics = {
        "solved": solved,
        "total": total,
        "expanded_nodes_total": expanded_total,
    }
    return metrics, diagnostics


def _astar_route(grid, terrain, start, goal, priority_score, tie_break_priority):
    rows = len(grid)
    cols = len(grid[0])
    start_h = _manhattan(start, goal)

    # Heap item: (-priority, -tie_break, g, r, c, prev_dir)
    open_heap: list[tuple[float, float, float, int, int, tuple[int, int] | None]] = []
    best_g: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    start_features = {
        "g_cost": 0.0,
        "h_cost": float(start_h),
        "turn_penalty": 0.0,
        "crowding": _crowding(grid, start),
        "progress": 0.0,
        "step_cost": 1.0,
        "escape_routes": float(_free_neighbor_count(grid, start)),
    }
    p0 = float(priority_score(start_features))
    if not math.isfinite(p0):
        raise ValueError("priority_score produced non-finite value")
    t0 = _tie_value(tie_break_priority, start_features)
    heapq.heappush(open_heap, (-p0, -t0, 0.0, start[0], start[1], None))

    expanded = 0
    max_expansions = rows * cols * 6

    while open_heap and expanded < max_expansions:
        _, _, g, r, c, prev_dir = heapq.heappop(open_heap)
        expanded += 1
        node = (r, c)
        if node == goal:
            path = _reconstruct_path(parent, goal)
            turns = _count_turns(path)
            return {
                "solved": True,
                "path_cost": g,
                "path_len": len(path) - 1,
                "turns": turns,
                "expanded_nodes": expanded,
            }

        if g > best_g.get(node, 10**12):
            continue

        for nr, nc in _neighbors(r, c, rows, cols):
            if grid[nr][nc] == 1:
                continue
            step_cost = float(terrain[nr][nc])
            ng = g + step_cost
            nxt = (nr, nc)
            if ng >= best_g.get(nxt, 10**12):
                continue

            direction = (nr - r, nc - c)
            turn_penalty = 1.0 if (prev_dir is not None and direction != prev_dir) else 0.0
            h = _manhattan(nxt, goal)
            features = {
                "g_cost": float(ng),
                "h_cost": float(h),
                "turn_penalty": turn_penalty,
                "crowding": _crowding(grid, nxt),
                "progress": 1.0 - (h / max(1.0, start_h)),
                "step_cost": step_cost,
                "escape_routes": float(_free_neighbor_count(grid, nxt)),
            }
            pr = float(priority_score(features))
            if not math.isfinite(pr):
                raise ValueError("priority_score produced non-finite value")
            tie = _tie_value(tie_break_priority, features)

            best_g[nxt] = ng
            parent[nxt] = node
            heapq.heappush(open_heap, (-pr, -tie, ng, nr, nc, direction))

    return {
        "solved": False,
        "path_cost": 10**12,
        "path_len": 10**9,
        "turns": 10**6,
        "expanded_nodes": expanded,
    }


def _tie_value(tie_break_priority, features):
    if not callable(tie_break_priority):
        return 0.0
    t = float(tie_break_priority(features))
    if not math.isfinite(t):
        return 0.0
    return t


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(r, c, rows, cols):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _sample_far_points(rng, grid):
    rows = len(grid)
    cols = len(grid[0])
    free_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 0]
    if len(free_cells) < 2:
        return None, None

    min_dist = int((rows + cols) * 0.60)
    for _ in range(48):
        start = rng.choice(free_cells)
        goal = rng.choice(free_cells)
        if start == goal:
            continue
        if _manhattan(start, goal) >= min_dist:
            return start, goal
    return None, None


def _crowding(grid, pos):
    rows = len(grid)
    cols = len(grid[0])
    r, c = pos
    blocked = 0
    total = 0
    for nr, nc in _neighbors(r, c, rows, cols):
        total += 1
        if grid[nr][nc] == 1:
            blocked += 1
    if total == 0:
        return 0.0
    return blocked / total


def _free_neighbor_count(grid, pos):
    rows = len(grid)
    cols = len(grid[0])
    r, c = pos
    return sum(1 for nr, nc in _neighbors(r, c, rows, cols) if grid[nr][nc] == 0)


def _reconstruct_path(parent, end):
    cur = end
    path = [cur]
    while parent.get(cur) is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _count_turns(path):
    if len(path) < 3:
        return 0
    turns = 0
    prev = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for a, b in zip(path[1:], path[2:]):
        cur = (b[0] - a[0], b[1] - a[1])
        if cur != prev:
            turns += 1
        prev = cur
    return turns


def _bfs_shortest_path_len(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0])
    q = deque([(start[0], start[1], 0)])
    seen = {start}

    while q:
        r, c, d = q.popleft()
        if (r, c) == goal:
            return d
        for nr, nc in _neighbors(r, c, rows, cols):
            if grid[nr][nc] == 1:
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nr, nc, d + 1))
    return None


def _dijkstra_shortest_path_cost(grid, terrain, start, goal):
    rows = len(grid)
    cols = len(grid[0])
    heap: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    dist: dict[tuple[int, int], float] = {start: 0.0}

    while heap:
        cur, r, c = heapq.heappop(heap)
        if (r, c) == goal:
            return cur
        if cur > dist.get((r, c), 10**12):
            continue

        for nr, nc in _neighbors(r, c, rows, cols):
            if grid[nr][nc] == 1:
                continue
            nxt = (nr, nc)
            nd = cur + float(terrain[nr][nc])
            if nd >= dist.get(nxt, 10**12):
                continue
            dist[nxt] = nd
            heapq.heappush(heap, (nd, nr, nc))
    return None


def _run_binpack_scenarios(heuristic_score, tie_break_score, scenarios):
    placed_total = 0
    jobs_total = 0
    balance_scores: list[float] = []
    heuristic_calls = 0
    tiny_hole_count = 0
    tiny_hole_slots = 0
    chosen_machine_indices: list[int] = []

    for scenario in scenarios:
        machines = deepcopy(scenario["machines"])
        jobs = scenario["jobs"]
        jobs_total += len(jobs)

        for job in jobs:
            best_idx = None
            best_score = -float("inf")
            best_tie = -float("inf")

            for idx, machine in enumerate(machines):
                if machine["cpu"] < job["cpu"] or machine["mem"] < job["mem"]:
                    continue

                score = float(heuristic_score(job, machine))
                heuristic_calls += 1
                if not math.isfinite(score):
                    raise ValueError("heuristic_score produced a non-finite value")

                tie = 0.0
                if callable(tie_break_score):
                    tie = float(tie_break_score(job, machine))
                    if not math.isfinite(tie):
                        tie = 0.0

                if score > best_score or (math.isclose(score, best_score) and tie > best_tie):
                    best_idx = idx
                    best_score = score
                    best_tie = tie

            if best_idx is not None:
                machines[best_idx]["cpu"] -= job["cpu"]
                machines[best_idx]["mem"] -= job["mem"]
                placed_total += 1
                chosen_machine_indices.append(best_idx)

        for machine in machines:
            cpu_denom = machine["cpu"] + 1e-9
            mem_denom = machine["mem"] + 1e-9
            balance = 1.0 - min(1.0, abs(cpu_denom - mem_denom) / max(cpu_denom, mem_denom))
            balance_scores.append(balance)
            if 0.0 < machine["cpu"] < 1.5:
                tiny_hole_count += 1
            if 0.0 < machine["mem"] < 1.5:
                tiny_hole_count += 1
            tiny_hole_slots += 2

    placed_ratio = placed_total / max(1, jobs_total)
    mean_balance = sum(balance_scores) / max(1, len(balance_scores))
    call_penalty = min(1.0, heuristic_calls / max(1.0, jobs_total * 4.0))
    fragmentation_score = 1.0 - (tiny_hole_count / max(1, tiny_hole_slots))

    switch_count = 0
    for prev_idx, next_idx in zip(chosen_machine_indices, chosen_machine_indices[1:]):
        if prev_idx != next_idx:
            switch_count += 1
    churn_penalty = switch_count / max(1, len(chosen_machine_indices) - 1)

    simplicity = 0.5
    aggregate_score = (
        0.55 * placed_ratio
        + 0.20 * mean_balance
        + 0.20 * fragmentation_score
        - 0.05 * churn_penalty
    )

    metrics = {
        "placed_jobs_ratio": placed_ratio,
        "balance_score": mean_balance,
        "fragmentation_score": fragmentation_score,
        "churn_penalty": churn_penalty,
        "simplicity_score": simplicity,
        "call_penalty": call_penalty,
        "aggregate_score": aggregate_score,
    }
    diagnostics = {
        "placed_jobs": placed_total,
        "total_jobs": jobs_total,
        "heuristic_calls": heuristic_calls,
    }
    return metrics, diagnostics
