"""Compare GA and random search under the same candidate-evaluation budget."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import csv
import numpy as np
from src.environment import GridEnvironment
from src.chromosome import random_chromosome
from src.fitness import evaluate_chromosome
from src.genetic_algorithm import GeneticAlgorithm
from src.visualization import ensure_output_dir


def compare(seed=123, population_size=40, generations=30, waypoints=8, output_dir="outputs"):
    env = GridEnvironment.create_default(seed=seed)
    ga = GeneticAlgorithm(env, population_size=population_size, generations=generations,
                          waypoints_per_drone=waypoints, seed=seed, patience=0)
    best_ga = ga.run()
    ga_evaluations = population_size + generations * population_size
    rng = np.random.default_rng(seed + 1)
    best_random = None
    for _ in range(ga_evaluations):
        candidate = random_chromosome(env, waypoints, rng)
        evaluate_chromosome(candidate, env, waypoints)
        if best_random is None or candidate.fitness > best_random.fitness:
            best_random = candidate.copy()

    rows = []
    for name, candidate in (("Genetic Algorithm", best_ga), ("Random Search", best_random)):
        row = {"method": name, **candidate.metrics}
        rows.append(row)
    out = ensure_output_dir(output_dir)
    fields = list(rows[0].keys())
    with (out / "baseline_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("\n=== GA vs Random Search ===")
    for row in rows:
        print(f"{row['method']:20s} fitness={row['fitness']:.4f}, "
              f"coverage={row['coverage']:.3f}, detection={row['detection']:.3f}, "
              f"feasible={row['feasible']}")
    print(f"Results saved to {out / 'baseline_comparison.csv'}")
    return rows


if __name__ == "__main__":
    compare()
