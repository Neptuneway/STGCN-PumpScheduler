# Pump Scheduling Optimization

This project provides a framework for optimizing pump scheduling in water distribution networks using a multi-objective genetic algorithm (NSGA-II). The primary goals of the optimization are to minimize both energy consumption costs and water leakage, while satisfying hydraulic constraints such as maintaining adequate pressure at critical nodes.

## Project Structure

```
Pump_Schedul_Optimization/
│
├──                          # EPANET water network model file
├──                          # Excel file with model constraints and operational data
├──                          # Excel file with water demand data
├──                          # Text file listing nodes with pressure constraints
├──                          # Text file listing pipes excluded from leakage calculations
│
├── config.py                 # Central configuration file for all parameters and file paths
├── ga_contraint.py           # Main script to run the genetic algorithm optimization
└──  chooseopti.py             # Module for calculating objective functions and hydraulic constraints

```

## How It Works

The optimization process is driven by the `ga_contraint.py` script, which performs the following steps:

1.  **Initialization**: The water network model (`.inp` file) is loaded, and pump patterns are configured based on settings in `config.py`.
2.  **Problem Definition**: An optimization problem is defined using the `pymoo` library. This includes:
    -   **Decision Variables**: The operational status (on/off) or speed of each pump.
    -   **Objective Functions**: Minimizing energy cost and water leakage, calculated in `chooseopti.py`.
    -   **Constraints**: Ensuring that hydraulic conditions, such as minimum pressure at critical nodes and maximum head at water plants, are met.
3.  **Genetic Algorithm**: The NSGA-II algorithm is run to evolve a population of pump schedules over a set number of generations.
4.  **Simulation and Evaluation**: For each candidate solution, a hydraulic simulation is performed using `wntr`. The results are then used to evaluate the objective functions and constraints.
5.  **Results**: The final Pareto front of non-dominated solutions is saved to CSV files, allowing for post-analysis to select the best trade-off solution.

