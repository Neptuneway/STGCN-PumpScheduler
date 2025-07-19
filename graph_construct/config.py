# config.py

# Path to the EPANET INP file
INP_FILE_PATH = "water_network.inp"

# List of pressure sensor node IDs
PRESSURE_SENSORS = [
]

# Simulation parameters
SIM_DURATION_HOURS = 
TIME_STEP =  # hours

# Aggregation threshold (e.g., 75% of total steps)
AGGREGATION_THRESHOLD = int(SIM_DURATION_HOURS * 0.75)

# Number of critical nodes (for adjacency matrix size)
NUM_CRITICAL_NODES = 