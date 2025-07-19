# Water Distribution Network Graph Extraction for STGCN

This project implements a hydraulic-model-based graph extraction methodology for water distribution network (WDN) monitoring systems, supporting spatiotemporal graph convolutional networks (STGCN) and other data-driven hydraulic analysis.

## Features

- Automatic extraction of WDN graph structure based on EPANET hydraulic simulation
- Dynamic flow direction and path analysis for critical nodes (monitoring points, sources, pump stations)
- Aggregation and symmetrization of adjacency matrices over multiple time steps
- Final graph visualization for direct use in STGCN or other GNN models
- All sensitive data and model files are externalized for data security

## Directory Structure

```
code/
  config.py                  # All parameters, file paths, and node lists (sensitive data and should be provided by user)
  wdn_graph_builder.py       # Build and save adjacency matrices for each time step
  wdn_graph_aggregate.py     # Aggregate and symmetrize adjacency matrices
  wdn_graph_analysis.py      # Visualize the final aggregated graph
  adjacency_matrices/        # Output folder for all generated adjacency matrices and results
  water_network.inp          # (Not included) Your EPANET .inp file (should be provided by user)
```

## Usage

1. **Prepare your data**  
   Place your EPANET `.inp` file in the `code/` directory.  
   **Do not upload or share your actual `.inp` file or any sensitive data to public repositories.**

2. **Configure parameters**  
   Edit `config.py` to set the correct file path, node list, and simulation parameters.

3. **Generate adjacency matrices**  
   ```bash
   python wdn_graph_builder.py
   ```
   This will create 216 adjacency matrices in `adjacency_matrices/`.

4. **Aggregate and process the matrices**  
   ```bash
   python wdn_graph_aggregate.py
   ```
   This will produce the final aggregated and symmetrized adjacency matrix.

5. **Visualize the final graph**  
   ```bash
   python wdn_graph_analysis.py
   ```
   The final graph image will be saved as `adjacency_matrices/final_graph.png`.

## Data Security

- **No sensitive or proprietary data is included in this repository.**
- All data, node lists, and model files are referenced via `config.py` and should be provided by the user locally.


## Requirements

- Python 3.7+
- [WNTR](https://github.com/USEPA/WNTR)
- numpy
- pandas
- networkx
- matplotlib
- scipy

Install dependencies:
```bash
pip install wntr numpy pandas networkx matplotlib scipy
```
 