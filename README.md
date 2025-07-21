# STOPS: Spatio-Temporal Optimization for Pump Scheduling

## Introduction

This is the official source code repository for the paper "Intelligent Real-Time Pump Scheduling in Water Distribution Networks Based on Graph Convolutional Networks". This research aims to address two core challenges faced by traditional pump scheduling methods in real-world applications:

1.  **Scheduling Lag due to Computational Delays**: The inherent time lag between data sensing and schedule execution means that network demand can shift during the optimization window. This renders the optimized schedule suboptimal by the time it is deployed, leading to resource wastage.
2.  **Mismatch between Model Constraints and Reality**: Existing optimization models often fail to capture the full spectrum of physical and operational constraints of a real-world system. This can result in schedules that are operationally infeasible, potentially causing low-pressure events and other risks upon deployment.

To address these challenges, we propose **STOPS (Spatio-Temporal Optimization for Pump Scheduling)**, an innovative framework based on a data-model co-design. STOPS integrates a **Spatio-Temporal Graph Convolutional Network (STGCN)** with a **Multi-Objective Optimization Algorithm (NSGA-II)** to achieve more timely and cost-effective pump scheduling while ensuring system operational safety.

## Framework Overview

The STOPS framework consists of the following key components:

1.  **WDN Graph Construction**: We first model the complex Water Distribution Network (WDN) as a topological graph, which accurately represents the physical connections between network components such as pipes, junctions, and pumping stations.

2.  **Spatio-Temporal Hydraulic State Forecasting**: We leverage a **Spatio-Temporal Graph Convolutional Network (STGCN)** to process real-time pressure and flow data from SCADA systems.
    -   The **temporal module** captures the time-series dependencies and dynamic evolution of hydraulic states (e.g., flow, pressure).
    -   The **spatial module** encodes the topological relationships and spatial correlations of hydraulic states across the network graph.
    -   By using the STGCN, we can accurately forecast future water treatment plant (WTP) outflow and the **head loss** between WTPs and critical monitoring points. This predictive capability effectively compensates for the sensing-to-execution time lag.

3.  **Multi-Objective Schedule Optimization**:
    -   The forecasted future flows from the STGCN are used to inform the selection of the hydraulic model, providing accurate inputs for the optimization solver.
    -   The forecasted future head losses are translated into **pressure constraints** at critical nodes within the optimization model. This ensures that the resulting schedules are physically feasible and compliant with operational requirements.
    -   We employ the **Non-dominated Sorting Genetic Algorithm II (NSGA-II)** to solve this multi-objective problem, generating adaptive control strategies with hourly resolution that balance trade-offs between objectives like **energy consumption** and **operational costs**.

This framework enables real-time pump scheduling by using STGCN-predicted future network states to secure a sufficient computational window for optimization. Furthermore, by incorporating predicted head loss as a core model constraint, we significantly improve the operational compliance and practical deployability of the schedules.

## Code Structure

This repository contains the core modules for implementing the STOPS framework:

-   `Graph_Construct/`: Code for constructing the Water Distribution Network (WDN) topological graph.
-   `STGCN/`: Implementation of the Spatio-Temporal Graph Convolutional Network (STGCN) for forecasting flow and head loss.
-   `GCN+LSTM/` & `LSTM/`: Other forecasting models used for performance comparison during our research.
-   `Pump_Schedul_Optimization/`: The pump scheduling optimization module based on the NSGA-II algorithm, including constraint definitions, objective functions, and the solver implementation.

## Usage

Each subdirectory contains the detailed code for the corresponding module. To run a specific module, please refer to the README file within that directory for specific instructions and guidelines.

## Citation

If you use the code or ideas from this project in your research, please consider citing our paper.


