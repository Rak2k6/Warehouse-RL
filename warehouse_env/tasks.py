"""
Task Definitions & Graders for the Warehouse Order Fulfillment Environment
===========================================================================
Defines 3 evaluation tasks with deterministic graders that return
scores STRICTLY between 0.0 and 1.0 (exclusive) as required by the
OpenEnv Phase 2 validator.

Tasks:
  1. Easy   — Minimize average fulfillment time
  2. Medium — Maximize worker utilization
  3. Hard   — Handle rush mode efficiently (composite metric)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .models import EpisodeResult


# ── Core clamp — NEVER returns exactly 0.0 or 1.0 ───────────────────────────

def _clamp(score: float) -> float:
    """Return score strictly in (0.001, 0.999). Validator rejects 0.0 and 1.0."""
    return round(float(max(0.001, min(0.999, score))), 4)


# ── Internal metric extractor — accepts EpisodeResult OR env/state dict ──────

def _metrics(source) -> dict:
    """
    Normalise inputs so graders work whether the validator passes:
      - an EpisodeResult (used by evaluate.py / run_task_grader)
      - a live env instance  (has get_full_state_dict())
      - a raw state dict
    """
    if isinstance(source, EpisodeResult):
        return {
            "orders_completed":       max(source.orders_completed, 0),
            "orders_generated":       max(source.orders_generated, 1),
            "avg_ft":                 float(source.avg_fulfillment_time),
            "utilization":            float(max(0.0, min(1.0, source.worker_utilization))),
            "steps":                  max(source.steps, 1),
        }

    # env instance or dict
    if hasattr(source, "get_full_state_dict"):
        s = source.get_full_state_dict()
    elif isinstance(source, dict):
        s = source
    else:
        # last resort: try attribute access
        s = vars(source)

    import numpy as np
    orders_completed       = max(int(s.get("orders_completed", 0)), 0)
    orders_generated       = max(int(s.get("total_orders_generated", 1)), 1)
    total_fulfillment_time = float(s.get("total_fulfillment_time", 0.0))
    worker_work_time       = s.get("worker_work_time", [0.0])
    step_count             = max(int(s.get("step_count", s.get("current_step", 1))), 1)

    avg_ft = total_fulfillment_time / max(orders_completed, 1)
    utilization = float(np.clip(np.mean(worker_work_time) / step_count, 0.0, 1.0))

    return {
        "orders_completed": orders_completed,
        "orders_generated": orders_generated,
        "avg_ft":           avg_ft,
        "utilization":      utilization,
        "steps":            step_count,
    }


# ── Grader Functions ─────────────────────────────────────────────────────────

def grade_fulfillment_time(result) -> float:
    """Task 1 (Easy): Minimize average fulfillment time.

    Score mapping (linear):
      avg_ft <= 2.0  → approaches 0.999
      avg_ft >= 20.0 → approaches 0.001
    Returns a float STRICTLY in (0.001, 0.999).
    """
    m = _metrics(result)
    if m["orders_completed"] == 0:
        return 0.001

    avg_ft = m["avg_ft"]
    BEST   = 2.0
    WORST  = 20.0

    raw = 1.0 - (avg_ft - BEST) / (WORST - BEST)
    return _clamp(raw)


def grade_worker_utilization(result) -> float:
    """Task 2 (Medium): Maximize worker utilization.

    Score = mean worker utilization (fraction of time busy).
    Returns a float STRICTLY in (0.001, 0.999).
    """
    m = _metrics(result)
    if m["steps"] == 0:
        return 0.001

    return _clamp(m["utilization"])


def grade_rush_mode(result) -> float:
    """Task 3 (Hard): Handle high-load (rush mode) efficiently.

    Composite:
      40% — completion rate  (orders_completed / orders_generated)
      30% — speed score      (same scale as task 1, rush worst=15 ticks)
      30% — utilization

    Returns a float STRICTLY in (0.001, 0.999).
    """
    import numpy as np

    m = _metrics(result)
    if m["orders_generated"] == 0:
        return 0.001

    completion_rate = m["orders_completed"] / m["orders_generated"]

    WORST_FT = 15.0   # rush mode has shorter proc times, so tighter bound
    speed_score = float(np.clip(1.0 - m["avg_ft"] / WORST_FT, 0.0, 1.0))

    utilization = m["utilization"]

    raw = 0.4 * completion_rate + 0.3 * speed_score + 0.3 * utilization
    return _clamp(raw)


# ── Task Definition ──────────────────────────────────────────────────────────

@dataclass
class Task:
    """A single evaluation task with its grader."""
    name: str
    description: str
    difficulty: str          # "easy", "medium", "hard"
    grader: Callable[[Any], float]
    env_config: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 200
    num_episodes: int = 10
    seed: int = 42


# ── Task Registry ────────────────────────────────────────────────────────────

TASKS: list[Task] = [
    Task(
        name="minimize_fulfillment_time",
        description=(
            "Minimize the average order fulfillment time. "
            "Score 0.999 = avg fulfillment ≤ 2.0 ticks; "
            "Score 0.001 = avg fulfillment ≥ 20.0 ticks."
        ),
        difficulty="easy",
        grader=grade_fulfillment_time,
        env_config={
            "num_workers": 4,
            "max_queue": 20,
            "max_orders": 50,
            "max_steps": 200,
            "mode": "normal",
        },
        max_steps=200,
        num_episodes=10,
        seed=42,
    ),
    Task(
        name="maximize_worker_utilization",
        description=(
            "Maximize average worker utilization (fraction of time workers are busy). "
            "Score = utilization ratio, strictly in (0.001, 0.999)."
        ),
        difficulty="medium",
        grader=grade_worker_utilization,
        env_config={
            "num_workers": 4,
            "max_queue": 20,
            "max_orders": 50,
            "max_steps": 200,
            "mode": "normal",
        },
        max_steps=200,
        num_episodes=10,
        seed=42,
    ),
    Task(
        name="rush_mode_efficiency",
        description=(
            "Handle high-load rush mode efficiently. "
            "Composite: 40% completion rate + 30% speed + 30% utilization. "
            "Rush mode: 70% arrival prob, shorter processing times."
        ),
        difficulty="hard",
        grader=grade_rush_mode,
        env_config={
            "num_workers": 4,
            "max_queue": 20,
            "max_orders": 80,
            "max_steps": 300,
            "mode": "rush",
        },
        max_steps=300,
        num_episodes=10,
        seed=42,
    ),
]


def get_task(name: str) -> Optional[Task]:
    """Look up a task by name."""
    for task in TASKS:
        if task.name == name:
            return task
    return None


def run_task_grader(task: Task, result) -> float:
    """Run the grader for a task and return the score."""
    return task.grader(result)