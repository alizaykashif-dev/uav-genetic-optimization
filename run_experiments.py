"""Run small sensitivity experiments for population, mutation, and selection pressure."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.environment import GridEnvironment
from src.genetic_algorithm import GeneticAlgorithm
from src.visualization import ensure_output_dir


def run_one(seed, population, generations, mutation, tournament, waypoints):
    env = GridEnvironment.create_default(seed=seed)
    ga = GeneticAlgorithm(env, population_size=population, generations=generations,
                          waypoints_per_drone=waypoints, mutation_rate=mutation,
                          tournament_size=tournament, seed=seed, patience=0)
    best = ga.run()
    return best.metrics["fitness"], best.metrics["coverage"], best.metrics["detection"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--waypoints", type=int, default=8)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    out = ensure_output_dir(args.output)
    # Change one factor at a time from the baseline.
    configs = []
    for value in (20, 40, 60):
        configs.append(("population_size", value, value, 0.10, 3))
    for value in (0.02, 0.10, 0.20):
        configs.append(("mutation_rate", value, 40, value, 3))
    for value in (2, 3, 5):
        configs.append(("tournament_size", value, 40, 0.10, value))

    rows = []
    for factor, value, pop, mutation, tournament in configs:
        fitnesses, coverages, detections = [], [], []
        for repeat in range(args.repeats):
            seed = 1000 + repeat
            fitness, coverage, detection = run_one(seed, pop, args.generations, mutation,
                                                    tournament, args.waypoints)
            fitnesses.append(fitness)
            coverages.append(coverage)
            detections.append(detection)
        rows.append({
            "factor": factor, "value": value, "population_size": pop,
            "mutation_rate": mutation, "tournament_size": tournament,
            "mean_best_fitness": statistics.mean(fitnesses),
            "std_best_fitness": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
            "mean_coverage": statistics.mean(coverages),
            "mean_detection": statistics.mean(detections),
            "repeats": args.repeats, "generations": args.generations,
        })
        print(f"{factor}={value}: mean fitness {rows[-1]['mean_best_fitness']:.4f}")

    csv_path = out / "sensitivity_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r['factor']}={r['value']}" for r in rows]
    means = [r["mean_best_fitness"] for r in rows]
    errors = [r["std_best_fitness"] for r in rows]
    ax.bar(range(len(rows)), means, yerr=errors, capsize=3)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean best fitness")
    ax.set_title("GA Hyperparameter Sensitivity")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "sensitivity_plot.png", dpi=160)
    plt.close(fig)
    print(f"CSV and plot saved in {out.resolve()}")


if __name__ == "__main__":
    main()
