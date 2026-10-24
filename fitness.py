"""Objective evaluation and constraint penalties for candidate drone plans."""
from __future__ import annotations
import math
import numpy as np
from .drone import SPEED_VALUES, SENSOR_POWER, SENSOR_DETECTION_PROBABILITY


# Weighted-sum objective weights; maximize the final scalar fitness.
WEIGHTS = {"coverage": 0.35, "detection": 0.30, "energy": 0.15, "time": 0.10, "hazard": 0.10}
SENSOR_RADIUS = (0.0, 2.0, 3.5)


def bresenham_cells(start, end):
    """Integer grid cells along a line segment, including both endpoints."""
    x0, y0 = map(int, start)
    x1, y1 = map(int, end)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


def repair_chromosome(chromosome, env):
    """Clamp waypoints to the grid and relocate points found inside obstacles."""
    for plan in chromosome.plans:
        for i, (x, y) in enumerate(plan.waypoints):
            x = int(np.clip(x, 0, env.width - 1))
            y = int(np.clip(y, 0, env.height - 1))
            plan.waypoints[i] = env.nearest_valid_cell(x, y)
        plan.speeds[:] = np.clip(plan.speeds, 0, 2)
        plan.sensors[:] = np.clip(plan.sensors, 0, 2)
    return chromosome


def evaluate_chromosome(chromosome, env, waypoint_count: int, coverage_radius: int = 2):
    """Evaluate route feasibility and normalized mission objectives.

    Waypoint speed/sensor settings apply to the segment ending at that waypoint.
    Start/end points are fixed by the environment.
    """
    repair_chromosome(chromosome, env)
    covered = np.zeros((env.height, env.width), dtype=bool)
    detection_probability = np.zeros((env.height, env.width), dtype=float)
    total_energy = total_time = total_hazard = total_distance = 0.0
    obstacle_violations = range_violations = time_violations = 0
    route_data = []

    for drone, plan in zip(env.drones, chromosome.plans):
        points = [drone.start] + [tuple(map(int, p)) for p in plan.waypoints] + [drone.goal]
        drone_distance = drone_energy = drone_time = 0.0
        drone_hazard = 0.0
        segments = []

        for idx in range(len(points) - 1):
            start, end = points[idx], points[idx + 1]
            mode_index = min(idx, len(plan.speeds) - 1)
            speed = SPEED_VALUES[int(plan.speeds[mode_index])]
            sensor_mode = int(plan.sensors[mode_index])
            cells = bresenham_cells(start, end)
            distance = math.dist(start, end)
            blocked = any(not env.is_valid_cell(x, y) for x, y in cells)
            if blocked:
                obstacle_violations += 1

            # Segment hazard exposure approximated by average hazard along traversed cells.
            valid_cells = [(x, y) for x, y in cells if 0 <= x < env.width and 0 <= y < env.height]
            mean_hazard = float(np.mean([env.hazard_map[y, x] for x, y in valid_cells])) if valid_cells else 1.0
            drone_distance += distance
            drone_hazard += distance * mean_hazard
            # Synthetic energy: distance term + speed-dependent term + sensor cost per unit time.
            drone_time += distance / speed
            drone_energy += distance * (0.7 + 0.08 * speed * speed) + SENSOR_POWER[sensor_mode] * (distance / speed) * 1.5

            # Coverage footprint from sensor mode around cells traversed on the segment.
            radius = SENSOR_RADIUS[sensor_mode]
            if radius > 0:
                for x, y in cells:
                    xmin, xmax = max(0, x - int(math.ceil(radius))), min(env.width - 1, x + int(math.ceil(radius)))
                    ymin, ymax = max(0, y - int(math.ceil(radius))), min(env.height - 1, y + int(math.ceil(radius)))
                    for cy in range(ymin, ymax + 1):
                        for cx in range(xmin, xmax + 1):
                            if (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2:
                                covered[cy, cx] = True
                                detection_probability[cy, cx] = max(
                                    detection_probability[cy, cx],
                                    SENSOR_DETECTION_PROBABILITY[sensor_mode]
                                )
            segments.append({"start": start, "end": end, "cells": cells, "blocked": blocked,
                             "speed": speed, "sensor_mode": sensor_mode, "distance": distance})
        if drone_distance > drone.max_range:
            range_violations += 1
        if drone_time > drone.time_budget:
            time_violations += 1
        total_distance += drone_distance
        total_energy += drone_energy
        total_time += drone_time
        total_hazard += drone_hazard
        route_data.append({"drone": drone, "points": points, "segments": segments,
                           "distance": drone_distance, "energy": drone_energy,
                           "time": drone_time, "hazard": drone_hazard})

    target_mask = env.target_map
    target_weights = env.target_weights
    total_target_weight = float(target_weights[target_mask].sum())
    if total_target_weight > 0:
        covered_weight = float(target_weights[target_mask & covered].sum())
        expected_detected_weight = float((target_weights * detection_probability)[target_mask].sum())
        coverage_score = covered_weight / total_target_weight
        detection_score = expected_detected_weight / total_target_weight
    else:
        coverage_score = detection_score = 0.0

    # Also reward broad survey coverage across the whole non-obstacle grid.
    open_mask = ~env.obstacle_map
    open_coverage = float(covered[open_mask].mean()) if open_mask.any() else 0.0
    coverage_score = 0.75 * coverage_score + 0.25 * open_coverage

    energy_norm = min(1.0, total_energy / sum(d.battery_budget for d in env.drones))
    time_norm = min(1.0, total_time / sum(d.time_budget for d in env.drones))
    hazard_norm = min(1.0, total_hazard / max(1.0, total_distance))
    constraint_penalty = (
        0.35 * obstacle_violations
        + 0.30 * range_violations
        + 0.25 * time_violations
    )
    fitness = (
        WEIGHTS["coverage"] * coverage_score
        + WEIGHTS["detection"] * detection_score
        - WEIGHTS["energy"] * energy_norm
        - WEIGHTS["time"] * time_norm
        - WEIGHTS["hazard"] * hazard_norm
        - constraint_penalty
    )
    feasible = obstacle_violations == 0 and range_violations == 0 and time_violations == 0
    metrics = {
        "fitness": float(fitness),
        "coverage": float(coverage_score),
        "open_area_coverage": float(open_coverage),
        "detection": float(detection_score),
        "energy": float(total_energy),
        "energy_normalized": float(energy_norm),
        "flight_time": float(total_time),
        "time_normalized": float(time_norm),
        "hazard_exposure": float(total_hazard),
        "hazard_normalized": float(hazard_norm),
        "distance": float(total_distance),
        "obstacle_violations": int(obstacle_violations),
        "range_violations": int(range_violations),
        "time_violations": int(time_violations),
        "feasible": bool(feasible),
        "covered_cells": int(covered.sum()),
    }
    chromosome.fitness = float(fitness)
    chromosome.metrics = metrics
    chromosome.feasible = feasible
    chromosome._evaluation = {"covered": covered, "detection_probability": detection_probability,
                              "routes": route_data}
    return fitness, metrics
