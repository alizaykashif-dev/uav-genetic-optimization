"""Matplotlib visualizations and CSV/JSON output helpers."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .fitness import evaluate_chromosome, bresenham_cells, SENSOR_RADIUS


def ensure_output_dir(path="outputs"):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def plot_convergence(history, output_dir="outputs"):
    out = ensure_output_dir(output_dir)
    generations = [row["generation"] for row in history]
    best = [row["best_fitness"] for row in history]
    average = [row["average_fitness"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(generations, best, label="Best fitness", linewidth=2)
    ax.plot(generations, average, label="Average fitness", linewidth=1.5)
    ax.set(title="GA Fitness Convergence", xlabel="Generation", ylabel="Fitness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fitness_convergence.png", dpi=160)
    plt.close(fig)


def plot_routes(env, chromosome, waypoint_count, output_dir="outputs"):
    out = ensure_output_dir(output_dir)
    _, metrics = evaluate_chromosome(chromosome, env, waypoint_count)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(env.hazard_map, origin="lower", cmap="YlOrRd", vmin=0, vmax=1, alpha=0.62)
    obstacle_y, obstacle_x = np.where(env.obstacle_map)
    ax.scatter(obstacle_x, obstacle_y, marker="s", s=12, color="black", label="No-fly / obstacle cells")
    target_y, target_x = np.where(env.target_map)
    if len(target_x):
        sizes = 25 + 30 * env.target_weights[target_y, target_x]
        ax.scatter(target_x, target_y, marker="*", s=sizes, facecolors="cyan",
                   edgecolors="black", linewidths=0.5, label="Priority targets")
    styles = ["-", "--", "-."]
    for idx, (drone, plan) in enumerate(zip(env.drones, chromosome.plans)):
        points = [drone.start] + [tuple(map(int, p)) for p in plan.waypoints] + [drone.goal]
        xs, ys = zip(*points)
        ax.plot(xs, ys, linestyle=styles[idx % len(styles)], linewidth=2.0,
                marker="o", markersize=3, label=drone.name)
        ax.scatter([drone.start[0]], [drone.start[1]], marker="s", s=65, edgecolors="black")
        ax.scatter([drone.goal[0]], [drone.goal[1]], marker="X", s=70, edgecolors="black")
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect("equal")
    ax.set(title=f"Evolved UAV Paths (fitness={metrics['fitness']:.3f})",
           xlabel="Grid x", ylabel="Grid y")
    ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "evolved_paths.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(env, chromosome, waypoint_count, output_dir="outputs"):
    out = ensure_output_dir(output_dir)
    evaluate_chromosome(chromosome, env, waypoint_count)
    coverage = np.zeros((env.height, env.width), dtype=int)
    for drone, plan in zip(env.drones, chromosome.plans):
        points = [drone.start] + [tuple(map(int, p)) for p in plan.waypoints] + [drone.goal]
        for idx, (start, end) in enumerate(zip(points[:-1], points[1:])):
            mode = int(plan.sensors[min(idx, len(plan.sensors) - 1)])
            radius = SENSOR_RADIUS[mode]
            if radius <= 0:
                continue
            for x, y in bresenham_cells(start, end):
                r = int(np.ceil(radius))
                for cy in range(max(0, y-r), min(env.height, y+r+1)):
                    for cx in range(max(0, x-r), min(env.width, x+r+1)):
                        if (cx-x)**2 + (cy-y)**2 <= radius**2:
                            coverage[cy, cx] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(coverage, origin="lower", cmap="viridis", interpolation="nearest")
    oy, ox = np.where(env.obstacle_map)
    ax.scatter(ox, oy, marker="s", s=8, color="white", edgecolors="black", linewidths=0.2, label="Obstacles")
    ty, tx = np.where(env.target_map)
    ax.scatter(tx, ty, marker="*", s=28, color="red", label="Targets")
    ax.set(title="Coverage Heatmap (number of drone passes)", xlabel="Grid x", ylabel="Grid y")
    fig.colorbar(image, ax=ax, label="Coverage count")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "coverage_heatmap.png", dpi=160)
    plt.close(fig)


def save_history(history, output_dir="outputs"):
    out = ensure_output_dir(output_dir)
    if not history:
        return
    with (out / "fitness_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_summary(chromosome, config, output_dir="outputs"):
    out = ensure_output_dir(output_dir)
    summary = {"config": config, "metrics": chromosome.metrics, "feasible": chromosome.feasible,
               "plans": []}
    for drone, plan in zip(config["environment"].drones, chromosome.plans):
        summary["plans"].append({
            "drone": drone.name,
            "start": list(drone.start),
            "goal": list(drone.goal),
            "waypoints": plan.waypoints.tolist(),
            "speed_modes": plan.speeds.tolist(),
            "sensor_modes": plan.sensors.tolist(),
        })
    summary["config"].pop("environment", None)
    with (out / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
