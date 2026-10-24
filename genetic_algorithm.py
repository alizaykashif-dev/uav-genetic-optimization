"""Genetic Algorithm engine with elitism and stagnation termination."""
from __future__ import annotations
import numpy as np
from .chromosome import random_chromosome
from .fitness import evaluate_chromosome, repair_chromosome
from .operators import tournament_selection, segment_crossover, mutate_chromosome


class GeneticAlgorithm:
    def __init__(self, env, population_size=50, generations=100, waypoints_per_drone=8,
                 crossover_rate=0.85, mutation_rate=0.10, tournament_size=3,
                 elite_count=2, patience=20, seed=42):
        if population_size < 4:
            raise ValueError("population_size must be at least 4.")
        self.env = env
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.waypoints_per_drone = int(waypoints_per_drone)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.tournament_size = int(tournament_size)
        self.elite_count = min(max(1, int(elite_count)), self.population_size - 2)
        self.patience = int(patience)
        self.rng = np.random.default_rng(seed)
        self.population = []
        self.history = []

    def _evaluate(self, chromosome):
        evaluate_chromosome(chromosome, self.env, self.waypoints_per_drone)
        return chromosome

    def run(self):
        self.population = [
            self._evaluate(random_chromosome(self.env, self.waypoints_per_drone, self.rng))
            for _ in range(self.population_size)
        ]
        best = max(self.population, key=lambda c: c.fitness).copy()
        best_seen = best.fitness
        stale = 0
        self._record(0)

        for generation in range(1, self.generations + 1):
            ranked = sorted(self.population, key=lambda c: c.fitness, reverse=True)
            new_population = [item.copy() for item in ranked[:self.elite_count]]
            while len(new_population) < self.population_size:
                parent_a = tournament_selection(self.population, self.tournament_size, self.rng)
                parent_b = tournament_selection(self.population, self.tournament_size, self.rng)
                if self.rng.random() < self.crossover_rate:
                    child_a, child_b = segment_crossover(parent_a, parent_b, self.rng)
                else:
                    child_a, child_b = parent_a.copy(), parent_b.copy()
                child_a = mutate_chromosome(child_a, self.env, self.rng, self.mutation_rate)
                child_b = mutate_chromosome(child_b, self.env, self.rng, self.mutation_rate)
                new_population.append(self._evaluate(child_a))
                if len(new_population) < self.population_size:
                    new_population.append(self._evaluate(child_b))
            self.population = new_population
            generation_best = max(self.population, key=lambda c: c.fitness)
            if generation_best.fitness > best_seen + 1e-10:
                best = generation_best.copy()
                best_seen = generation_best.fitness
                stale = 0
            else:
                stale += 1
            self._record(generation)
            if self.patience > 0 and stale >= self.patience:
                break

        self.best = best
        return best

    def _record(self, generation):
        fitnesses = np.asarray([c.fitness for c in self.population], dtype=float)
        best = max(self.population, key=lambda c: c.fitness)
        self.history.append({
            "generation": int(generation),
            "best_fitness": float(best.fitness),
            "average_fitness": float(fitnesses.mean()),
            "feasible_fraction": float(np.mean([c.feasible for c in self.population])),
        })
