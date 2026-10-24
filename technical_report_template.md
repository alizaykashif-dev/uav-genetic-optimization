# GA-Based Multi-Objective UAV Deployment and Path Planning

**Team members:** [Member 1], [Member 2], [Member 3]  
**Course / section:** [Course and section]  
**Date:** [Submission date]

> Replace every bracketed item with your team's information. Insert the actual figures and measured values from your runs. Do not report example or fabricated results.

## 1. Introduction and problem formulation

This project models a fleet of three unmanned aerial vehicles (UAVs) surveying a synthetic disaster region represented by a 2D grid. The environment contains blocked cells, spatially varying hazard intensity, and weighted target locations. The task is to choose waypoint trajectories and per-segment speed and sensor settings that balance survey coverage and target detection against energy consumption, mission time, and hazard exposure.

A single-objective shortest-path approach would not capture all operational trade-offs. A shorter route may miss priority targets, while high-power sensing may improve detection but consume additional energy. We therefore use a Genetic Algorithm (GA), which evolves candidate fleet plans using selection, crossover, and mutation.

The simulator is educational and uses illustrative values rather than validated aircraft or sensor models.

## 2. Chromosome design and genetic operators

A chromosome contains one plan for each drone. Each plan stores a fixed sequence of intermediate integer `(x, y)` waypoints, a speed mode for each waypoint, and a sensor mode for each waypoint. The drone's start and destination are fixed by the environment. Speed modes are `0=slow`, `1=normal`, `2=fast`; sensor modes are `0=off`, `1=low`, `2=high`.

This hybrid encoding was selected because coordinates describe spatial decisions while discrete integers describe operational modes. The fixed-length representation makes segment crossover straightforward. Tournament selection chooses parents. Segment crossover swaps a contiguous section of waypoint coordinates and the corresponding speed and sensor genes. Mutation can perturb a waypoint locally, make a larger spatial jump, or change a speed/sensor mode. Elitism retains the best candidates, and termination occurs after the generation limit or a configured number of generations without improvement.

After variation, waypoints are clipped to the grid and moved away from obstacle cells. Route segments are checked cell-by-cell. Range and time violations are recorded and penalized. A waypoint repair does not guarantee that every segment is obstacle-free, so feasibility is measured explicitly.

## 3. Fitness function

The scalar fitness function is

\[
F = 0.35C + 0.30D - 0.15E - 0.10T - 0.10H - P
\]

where `C` is weighted coverage, `D` is expected weighted target detection, `E` is normalized energy, `T` is normalized flight time, `H` is normalized hazard exposure, and `P` is the sum of constraint penalties. Coverage and detection are normalized to `[0, 1]`; energy and time are normalized against fleet budgets; hazard is normalized by total distance. The weights are design choices that emphasize coverage and detection while retaining costs for energy, time, and risk. A weighted sum produces one compromise solution for the chosen weights; it does not represent the full Pareto front.

The sensor model uses illustrative detection probabilities and sensing radii for off, low, and high modes. Energy combines distance, speed-dependent cost, and sensor-on time. Hazard exposure accumulates distance multiplied by the hazard intensity along a route segment. These simplified models are useful for comparing candidate solutions within the simulation but should not be interpreted as physical performance predictions.

## 4. Experimental setup and results

**Environment:** [grid size], [obstacle percentage], [target count], [number of drones].  
**GA settings:** population [value], generations [value], mutation rate [value], tournament size [value], waypoint count [value], random seed(s) [values].  
**Machine / runtime environment:** [brief details].

Insert `outputs/fitness_convergence.png`, `outputs/evolved_paths.png`, and `outputs/coverage_heatmap.png` here. Describe whether best fitness improved, how average fitness changed, where the paths travelled, and which priority areas were covered. Report the best plan's feasibility and its measured coverage, detection, energy, time, and hazard metrics.

### Hyperparameter sensitivity

Vary one parameter at a time while keeping other settings and the environment fixed. Use multiple random seeds and report means and standard deviations.

| Factor | Values tested | Mean best fitness (and standard deviation) | Interpretation |
|---|---|---|---|
| Population size | [20, 40, 60] | [measured values] | [interpretation] |
| Mutation rate | [0.02, 0.10, 0.20] | [measured values] | [interpretation] |
| Tournament size | [2, 3, 5] | [measured values] | [interpretation] |

Discuss the observed exploration/runtime trade-off. A larger population can explore more candidates but costs more evaluations. Very low mutation may reduce exploration; very high mutation can disrupt useful building blocks. Larger tournaments increase selection pressure and may cause faster convergence but reduce diversity. Treat these as hypotheses and base conclusions on the measured results.

## 5. Random-search baseline and discussion

Random search samples and evaluates candidate chromosomes without using evolutionary selection or recombination. The provided comparison script uses the same environment, chromosome representation, fitness function, and approximately the same candidate-evaluation budget for both methods.

| Metric | Genetic Algorithm | Random Search |
|---|---:|---:|
| Best fitness | [measured] | [measured] |
| Coverage | [measured] | [measured] |
| Detection | [measured] | [measured] |
| Energy | [measured] | [measured] |
| Flight time | [measured] | [measured] |
| Hazard exposure | [measured] | [measured] |
| Feasible solution | [measured] | [measured] |

Explain whether the GA outperformed random search in your runs and whether the difference was consistent across seeds. Do not assume the GA always wins; the outcome depends on the fitness design, constraints, hyperparameters, and evaluation budget.

## 6. Limitations and conclusion

The environment and energy, sensor, and hazard models are simplified. The current implementation uses a weighted-sum fitness rather than a Pareto-based multi-objective algorithm. Routes are represented as waypoint-to-waypoint straight segments for evaluation; the simulator does not provide certified collision avoidance, aircraft dynamics, wind response, communication constraints, or real-world flight control. Future work could use A* to connect strategic waypoints with obstacle-avoiding paths, model battery state over time, and use NSGA-II to produce a Pareto front of trade-off solutions.

In conclusion, this project demonstrates a complete evolutionary optimization workflow for a synthetic UAV deployment problem. The main finding from our experiments was [insert measured conclusion].
