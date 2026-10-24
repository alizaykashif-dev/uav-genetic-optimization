"""Run the UAV Genetic Algorithm simulation."""
from pathlib import Path
import argparse
import json
from src.environment import GridEnvironment
from src.genetic_algorithm import GeneticAlgorithm
from src.visualization import plot_convergence, plot_routes, plot_coverage, save_history, save_summary, ensure_output_dir


def run_simulation(seed=42, population_size=50, generations=100, mutation_rate=0.10,
                   tournament_size=3, waypoints_per_drone=8, output_dir="outputs", patience=20):
    env = GridEnvironment.create_default(seed=seed)
    ga = GeneticAlgorithm(
        env=env, population_size=population_size, generations=generations,
        waypoints_per_drone=waypoints_per_drone, mutation_rate=mutation_rate,
        tournament_size=tournament_size, seed=seed, patience=patience,
    )
    best = ga.run()
    # Re-evaluate once to populate visualization coverage and metrics consistently.
    from src.fitness import evaluate_chromosome
    evaluate_chromosome(best, env, waypoints_per_drone)
    ensure_output_dir(output_dir)
    plot_convergence(ga.history, output_dir)
    plot_routes(env, best, waypoints_per_drone, output_dir)
    plot_coverage(env, best, waypoints_per_drone, output_dir)
    save_history(ga.history, output_dir)
    config = {
        "seed": seed, "population_size": population_size, "generations_requested": generations,
        "generations_completed": ga.history[-1]["generation"], "mutation_rate": mutation_rate,
        "tournament_size": tournament_size, "waypoints_per_drone": waypoints_per_drone,
        "patience": patience, "environment": env,
    }
    save_summary(best, config, output_dir)
    print("\n=== UAV Genetic Algorithm Results ===")
    print(f"Generations completed: {ga.history[-1]['generation']}")
    print(f"Best fitness:          {best.fitness:.4f}")
    print(f"Feasible route plan:   {best.feasible}")
    for key, value in best.metrics.items():
        if key not in ("fitness", "feasible"):
            print(f"{key:22s}: {value}")
    print(f"\nPlots and results saved in: {Path(output_dir).resolve()}")
    return env, ga, best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the UAV Genetic Algorithm simulation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--mutation", type=float, default=0.10)
    parser.add_argument("--tournament", type=int, default=3)
    parser.add_argument("--waypoints", type=int, default=8)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    run_simulation(args.seed, args.population, args.generations, args.mutation,
                   args.tournament, args.waypoints, args.output, args.patience)
