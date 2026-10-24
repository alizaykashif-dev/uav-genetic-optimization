"""Custom genetic operators for the hybrid UAV chromosome."""
from __future__ import annotations
import numpy as np
from .chromosome import Chromosome
from .fitness import repair_chromosome


def tournament_selection(population, tournament_size, rng):
    size = min(max(2, int(tournament_size)), len(population))
    indices = rng.choice(len(population), size=size, replace=False)
    return max((population[int(i)] for i in indices), key=lambda item: item.fitness)


def segment_crossover(parent_a: Chromosome, parent_b: Chromosome, rng) -> tuple[Chromosome, Chromosome]:
    """Swap a contiguous waypoint segment and its matching operational settings."""
    child_a, child_b = parent_a.copy(), parent_b.copy()
    for plan_a, plan_b in zip(child_a.plans, child_b.plans):
        k = len(plan_a.waypoints)
        if k < 2:
            continue
        left = int(rng.integers(0, k - 1))
        right = int(rng.integers(left + 1, k + 1))
        for child_plan, donor_plan in ((plan_a, plan_b), (plan_b, plan_a)):
            # Temporary copies are used below to avoid overwriting donor values.
            pass
        old_a_wp, old_b_wp = plan_a.waypoints.copy(), plan_b.waypoints.copy()
        old_a_s, old_b_s = plan_a.speeds.copy(), plan_b.speeds.copy()
        old_a_p, old_b_p = plan_a.sensors.copy(), plan_b.sensors.copy()
        plan_a.waypoints[left:right] = old_b_wp[left:right]
        plan_b.waypoints[left:right] = old_a_wp[left:right]
        plan_a.speeds[left:right] = old_b_s[left:right]
        plan_b.speeds[left:right] = old_a_s[left:right]
        plan_a.sensors[left:right] = old_b_p[left:right]
        plan_b.sensors[left:right] = old_a_p[left:right]
    child_a.fitness = child_b.fitness = float("-inf")
    child_a.metrics = child_b.metrics = None
    return child_a, child_b


def mutate_chromosome(chromosome: Chromosome, env, rng, mutation_rate: float,
                      spatial_step: int = 5):
    """Mutate waypoint coordinates, speed modes, and sensor modes."""
    child = chromosome.copy()
    for plan in child.plans:
        for i in range(len(plan.waypoints)):
            if rng.random() < mutation_rate:
                if rng.random() < 0.75:
                    x, y = plan.waypoints[i]
                    # Mostly local moves; occasionally a larger random jump.
                    if rng.random() < 0.8:
                        x += int(rng.integers(-spatial_step, spatial_step + 1))
                        y += int(rng.integers(-spatial_step, spatial_step + 1))
                    else:
                        x = int(rng.integers(0, env.width))
                        y = int(rng.integers(0, env.height))
                    plan.waypoints[i] = env.nearest_valid_cell(x, y)
                else:
                    # Mutate one of the discrete speed/sensor genes.
                    if rng.random() < 0.5:
                        plan.speeds[i] = int(rng.integers(0, 3))
                    else:
                        plan.sensors[i] = int(rng.integers(0, 3))
    child.fitness = float("-inf")
    child.metrics = None
    child.feasible = False
    return repair_chromosome(child, env)
