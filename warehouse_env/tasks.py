"""
Task Definitions & Graders for the Warehouse Order Fulfillment Environment
===========================================================================
Defines 3 evaluation tasks with deterministic graders that return
scores strictly between 0.0 and 1.0.

Tasks:
  1. Easy   — Minimize average fulfillment time
  2. Medium — Maximize worker utilization
  3. Hard   — Handle rush mode efficiently (composite metric)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .models import EpisodeResult


# ── Grader Functions ─────────────────────────────────────────────────

def grade_fulfillment_time(result: EpisodeResult) -> float:
    """Task 1 (Easy): Minimize average fulfillment time.

    Score mapping:
      - avg_ft <= 2.0  → 1.0  (near-optimal)
      - avg_ft >= 20.0 → 0.0  (very poor)
      - Linear interpolation between
    """
    if result.orders_completed == 0:
        return 0.0

    avg_ft = result.avg_fulfillment_time
    best = 2.0   # near-theoretical minimum
    worst = 20.0  # very poor performance

    score = 1.0 - (avg_ft - best) / (worst - best)
    return round(max(0.0, min(1.0, score)), 4)


def grade_worker_utilization(result: EpisodeResult) -> float:
    """Task 2 (Medium): Maximize worker utilization.

    Score = worker_utilization (already 0-1).
    Clamped and rounded for safety.
    """
    if result.steps == 0:
        return 0.0

    score = result.worker_utilization
    return round(max(0.0, min(1.0, score)), 4)


def grade_rush_mode(result: EpisodeResult) -> float:
    """Task 3 (Hard): Handle high-load (rush mode) efficiently.

    Composite score:
      40% — completion rate (orders_completed / orders_generated)
      30% — speed score (inverse of fulfillment time, same scale as task 1)
      30% — utilization score (same as task 2)

    Designed so that only a well-tuned policy scores > 0.7.
    """
    if result.orders_generated == 0:
        return 0.0

    # Completion rate
    completion_rate = result.orders_completed / max(result.orders_generated, 1)

    # Speed score (same as task 1)
    if result.orders_completed > 0:
        avg_ft = result.avg_fulfillment_time
        speed_score = 1.0 - (avg_ft - 2.0) / (20.0 - 2.0)
        speed_score = max(0.0, min(1.0, speed_score))
    else:
        speed_score = 0.0

    # Utilization
    utilization = max(0.0, min(1.0, result.worker_utilization))

    # Composite
    score = 0.4 * completion_rate + 0.3 * speed_score + 0.3 * utilization
    return round(max(0.0, min(1.0, score)), 4)


# ── Task Definition ──────────────────────────────────────────────────

@dataclass
class Task:
    """A single evaluation task with its grader."""
    name: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    grader: Callable[[EpisodeResult], float]
    env_config: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 200
    num_episodes: int = 10
    seed: int = 42


# ── Task Registry ────────────────────────────────────────────────────

TASKS: list[Task] = [
    Task(
        name="minimize_fulfillment_time",
        description=(
            "Minimize the average order fulfillment time. "
            "Score 1.0 = avg fulfillment ≤ 2.0 ticks; "
            "Score 0.0 = avg fulfillment ≥ 20.0 ticks."
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
            "Score = utilization ratio (0.0 to 1.0)."
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
            "Composite score: 40% completion rate + 30% speed + 30% utilization. "
            "Rush mode has 70% order arrival probability and shorter processing times."
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


def run_task_grader(task: Task, result: EpisodeResult) -> float:
    """Run the grader for a task and return the score."""
    return task.grader(result)
