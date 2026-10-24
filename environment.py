"""Synthetic grid environment with obstacles, hazards, and priority targets."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GridEnvironment:
    width: int = 50
    height: int = 50
    obstacle_map: np.ndarray | None = None
    hazard_map: np.ndarray | None = None
    target_map: np.ndarray | None = None
    target_weights: np.ndarray | None = None
    drones: list | None = None

    @classmethod
    def create_default(cls, seed: int = 42, width: int = 50, height: int = 50,
                       obstacle_fraction: float = 0.08, target_count: int = 22):
        """Create a reproducible map; obstacle rectangles avoid start/goal areas."""
        rng = np.random.default_rng(seed)
        env = cls(width=width, height=height)
        env.obstacle_map = np.zeros((height, width), dtype=bool)
        env.hazard_map = np.zeros((height, width), dtype=float)
        env.target_map = np.zeros((height, width), dtype=bool)
        env.target_weights = np.zeros((height, width), dtype=float)

        # Smooth hazard pockets, built from Gaussian-like fields.
        yy, xx = np.mgrid[0:height, 0:width]
        for _ in range(7):
            cx = rng.integers(4, width - 4)
            cy = rng.integers(4, height - 4)
            sigma = rng.uniform(2.5, 6.5)
            strength = rng.uniform(0.35, 0.95)
            field = strength * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            env.hazard_map = np.maximum(env.hazard_map, field)
        env.hazard_map = np.clip(env.hazard_map, 0.0, 1.0)

        # Place rectangular obstacles, avoiding the four corner operating zones.
        desired_obstacles = int(width * height * obstacle_fraction)
        attempts = 0
        while int(env.obstacle_map.sum()) < desired_obstacles and attempts < desired_obstacles * 12 + 100:
            attempts += 1
            x = int(rng.integers(3, width - 5))
            y = int(rng.integers(3, height - 5))
            rw = int(rng.integers(2, 5))
            rh = int(rng.integers(2, 5))
            if (x < 8 and y < 8) or (x > width - 9 and y > height - 9):
                continue
            env.obstacle_map[y:min(y + rh, height), x:min(x + rw, width)] = True

        # Fixed start and goal locations for three drones.
        from .drone import Drone
        env.drones = [
            Drone("Drone 1", (2, 2), (width - 3, height - 3)),
            Drone("Drone 2", (2, height - 3), (width - 3, 2)),
            Drone("Drone 3", (width // 2, 2), (width // 2, height - 3)),
        ]
        for drone in env.drones:
            for x, y in (drone.start, drone.goal):
                env.obstacle_map[y, x] = False

        # Random target cells weighted toward higher-priority targets.
        candidates = []
        for _ in range(target_count * 30):
            x = int(rng.integers(1, width - 1))
            y = int(rng.integers(1, height - 1))
            if not env.obstacle_map[y, x] and all(
                (x, y) not in (d.start, d.goal) for d in env.drones
            ):
                candidates.append((x, y))
            if len(candidates) >= target_count:
                break
        for x, y in candidates:
            env.target_map[y, x] = True
            env.target_weights[y, x] = float(rng.choice([1.0, 1.5, 2.0, 3.0], p=[0.35, 0.3, 0.25, 0.1]))

        return env

    def is_valid_cell(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and not self.obstacle_map[y, x]

    def nearest_valid_cell(self, x: int, y: int, max_radius: int = 12) -> tuple[int, int]:
        """Return the nearest non-obstacle cell, searching outward from (x, y)."""
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        if self.is_valid_cell(x, y):
            return x, y
        for radius in range(1, max_radius + 1):
            candidates = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    nx, ny = x + dx, y + dy
                    if self.is_valid_cell(nx, ny):
                        candidates.append((nx, ny, dx * dx + dy * dy))
            if candidates:
                nx, ny, _ = min(candidates, key=lambda item: item[2])
                return nx, ny
        # The default map always has valid cells; this is a defensive fallback.
        for ny in range(self.height):
            for nx in range(self.width):
                if self.is_valid_cell(nx, ny):
                    return nx, ny
        raise RuntimeError("Environment contains no valid cells.")
