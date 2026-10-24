"""Drone configuration and discrete operational modes."""
from dataclasses import dataclass

SPEED_VALUES = (2.0, 4.0, 6.0)  # grid units per time unit
SENSOR_POWER = (0.0, 0.5, 1.0)
SENSOR_DETECTION_PROBABILITY = (0.0, 0.55, 0.90)


@dataclass(frozen=True)
class Drone:
    """Simple synthetic drone model used by the educational simulator."""

    name: str
    start: tuple[int, int]
    goal: tuple[int, int]
    max_range: float = 115.0
    battery_budget: float = 400.0
    time_budget: float = 70.0
