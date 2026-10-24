# GA-Based Multi-Objective UAV Deployment and Path Planning

A self-contained educational simulation that uses a custom Genetic Algorithm (GA) to evolve routes and operating schedules for a fleet of UAVs surveying a simulated disaster grid.

> **Scope and safety:** This is a synthetic, educational simulation—not flight-control software. Coordinates, sensor probabilities, energy costs, and hazards are illustrative. Do not use its routes for real aircraft or real-world emergency response.

## Features

- Configurable 2D grid with obstacles, hazard intensity, and weighted target cells.
- Object-oriented hybrid chromosome: integer waypoint coordinates, speed modes, and sensor modes.
- Custom tournament selection, segment crossover, spatial/operational mutation, elitism, and termination on generation limit or stagnation.
- Route checking against grid obstacles, range, and mission-time constraints.
- Fitness balancing weighted coverage, expected target detection, energy, flight time, and hazard exposure.
- Best-vs-average fitness plot, evolved route map, and coverage heatmap.
- Random-search baseline with a comparable fitness-evaluation budget.
- Hyperparameter sensitivity experiment script.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib

## Setup

Open a terminal in this repository folder.

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, you can use `.\.venv\Scripts\python.exe main.py` after installing packages with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The simulation writes plots and machine-readable results to `outputs/`:
- `fitness_convergence.png`
- `evolved_paths.png`
- `coverage_heatmap.png`
- `run_summary.json`
- `fitness_history.csv`

The default run uses a fixed random seed so it is reproducible. Change `seed` in `main.py` to explore another run.

## Run the random-search comparison

```bash
python experiments/compare_random_search.py
```

This uses the same environment, fitness function, and number of candidate evaluations for GA and random search. It writes `outputs/baseline_comparison.csv` and prints a summary.

## Run sensitivity experiments

```bash
python experiments/run_experiments.py
```

The script varies population size, mutation rate, and tournament size. It runs multiple seeds for each configuration and writes:
- `outputs/sensitivity_results.csv`
- `outputs/sensitivity_plot.png`

The default experiment settings are intentionally modest so they can run on a student laptop. Increase `--repeats` or generations for more reliable analysis:

```bash
python experiments/run_experiments.py --repeats 5 --generations 60
```

## Project structure

```text
uav-genetic-optimization/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── src/
│   ├── __init__.py
│   ├── environment.py
│   ├── drone.py
│   ├── chromosome.py
│   ├── fitness.py
│   ├── operators.py
│   ├── genetic_algorithm.py
│   └── visualization.py
├── experiments/
│   ├── compare_random_search.py
│   └── run_experiments.py
├── outputs/
│   └── .gitkeep
└── report/
    └── technical_report_template.md
```

## Model and assumptions

### Chromosome

Each candidate has a fixed number of intermediate waypoints per drone. Start and end points are fixed by the environment. Every intermediate waypoint has:
- `(x, y)` integer grid coordinates
- a speed mode: `0=slow`, `1=normal`, `2=fast`
- a sensor mode: `0=off`, `1=low`, `2=high`

The chromosome is repaired after initialization, crossover, and mutation. Waypoints inside obstacles are moved to nearby valid cells. Segments are checked cell-by-cell; a route that crosses a blocked cell is infeasible and penalized. This simulation does not implement aircraft dynamics or a full low-level path planner.

### Fitness

All objective components are normalized to `[0, 1]` before combining them:

`fitness = 0.35*coverage + 0.30*detection - 0.15*energy - 0.10*time - 0.10*hazard - constraint_penalties`

Coverage is the weighted fraction of priority cells reached within the simulated sensor footprint. Detection is the expected weighted detection rate, using illustrative sensor-mode probabilities. Energy, time, and hazard exposure are normalized against configured budgets. The weights are configurable in `src/fitness.py`; they represent a design choice, not a universal optimum.

Hard restrictions (obstacles, flight range, mission time) are checked separately and incur penalties if violated. The repair function tries to correct waypoint locations, but a repaired waypoint sequence is not guaranteed to make every route feasible.

### Multi-objective note

The implementation uses a **weighted-sum scalar fitness** so a standard GA can rank candidate solutions. This returns one trade-off solution for the selected weights; it does not calculate a Pareto front. A future extension could use NSGA-II or another Pareto-based method.

## Interpreting results

- A rising best-fitness curve suggests the search is finding better candidates. The average may fluctuate.
- The evolved route figure overlays paths, obstacles, hazards, targets, and start/end markers.
- The coverage heatmap shows how many drones' simulated sensor footprints cover each cell in the best solution.
- Compare GA and random search using the same number of candidate evaluations. Do not claim the GA always wins; report the actual measurements from your run.
- Sensitivity results are experimental. Run multiple seeds and report mean and standard deviation rather than relying on one run.

## Team workflow suggestion

1. One member owns `environment.py` and `drone.py`.
2. One member owns `chromosome.py`, `fitness.py`, `operators.py`, and `genetic_algorithm.py`.
3. One member owns visualizations, experiments, and the report.
4. Agree on interfaces first, integrate with Git, and all team members should understand the complete implementation.

## GitHub upload

Create an empty repository on GitHub, then run:

```bash
git init
git add .
git commit -m "Add UAV genetic optimization simulation"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_REPOSITORY_URL` with the URL of your own repository. Do not commit a virtual environment.

