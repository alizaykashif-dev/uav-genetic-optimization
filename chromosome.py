"""Hybrid chromosome encoding for routes, speed modes, and sensor modes."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class DronePlan:
    waypoints: np.ndarray  # shape (K, 2), integer (x, y)
    speeds: np.ndarray     # shape (K,), integer modes 0..2
    sensors: np.ndarray    # shape (K,), integer modes 0..2

    def copy(self) -> "DronePlan":
        return DronePlan(self.waypoints.copy(), self.speeds.copy(), self.sensors.copy())


@dataclass
class Chromosome:
    plans: list[DronePlan]
    fitness: float = float("-inf")
    metrics: dict | None = None
    feasible: bool = False

    def copy(self) -> "Chromosome":
        return Chromosome(
            plans=[p.copy() for p in self.plans],
            fitness=self.fitness,
            metrics=None if self.metrics is None else dict(self.metrics),
            feasible=self.feasible,
        )


def random_chromosome(env, waypoints_per_drone: int, rng: np.random.Generator) -> Chromosome:
    plans = []
    for drone in env.drones:
        points = []
        # Seed routes near the direct start-goal corridor, with spatial noise.
        # This gives the population useful structure while still allowing exploration.
        for index in range(waypoints_per_drone):
            fraction = (index + 1) / (waypoints_per_drone + 1)
            base_x = drone.start[0] + fraction * (drone.goal[0] - drone.start[0])
            base_y = drone.start[1] + fraction * (drone.goal[1] - drone.start[1])
            x = int(round(base_x + rng.integers(-5, 6)))
            y = int(round(base_y + rng.integers(-5, 6)))
            points.append(env.nearest_valid_cell(x, y))
        plans.append(
            DronePlan(
                waypoints=np.asarray(points, dtype=int),
                speeds=rng.integers(0, 3, size=waypoints_per_drone, dtype=int),
                sensors=rng.integers(0, 3, size=waypoints_per_drone, dtype=int),
            )
        )
    return Chromosome(plans=plans)
