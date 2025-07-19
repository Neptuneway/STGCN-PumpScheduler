import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from config import SIM_DURATION_HOURS, AGGREGATION_THRESHOLD, NUM_CRITICAL_NODES

INPUT_DIR = "adjacency_matrices"

# Aggregate adjacency matrices over all time steps
def aggregate_adjacency_matrices():
    n = NUM_CRITICAL_NODES
    m = AGGREGATION_THRESHOLD
    num = np.zeros((n, n))
    for i in range(SIM_DURATION_HOURS):
        a = np.loadtxt(os.path.join(INPUT_DIR, f'{i}_monitor_adjacency_weighted.csv'), delimiter=',', skiprows=1)[:, 1:]
        num = num + a
    num2 = np.where(num > m, 1, 0)
    np.savetxt(os.path.join(INPUT_DIR, f'adjacency_edge_count_{m}.csv'), num, delimiter=',')
    sum_matrix = np.zeros((n, n))
    for i in range(SIM_DURATION_HOURS):
        b = pd.read_csv(os.path.join(INPUT_DIR, f'{i}_monitor_adjacency_weighted.csv'), index_col=0)
        sum_matrix = sum_matrix + b.values
    num_masked = np.where(num > m, num, 1)
    result_avg = sum_matrix / num_masked
    result_avg = result_avg * num2
    result2 = result_avg * 0.0001
    result2 = np.where(result2 == 0, 0, np.exp(-result2 ** 2))
    row, col = np.diag_indices_from(result2)
    result2[row, col] = 1
    pd.DataFrame(result2).to_csv(os.path.join(INPUT_DIR, f'final_adjacency_exp_{m}_directed.csv'))
    # Symmetrize the adjacency matrix
    row, col = np.nonzero(result2)
    values = result2[row, col]
    result2_sparse = csr_matrix((values, (row, col)), shape=(n, n))
    sym_result2 = result2_sparse + result2_sparse.T.multiply(result2_sparse.T > result2_sparse) - result2_sparse.multiply(result2_sparse.T > result2_sparse)
    sym_result2 = sym_result2.toarray()
    pd.DataFrame(sym_result2).to_csv(os.path.join(INPUT_DIR, f'final_adjacency_exp_{m}_undirected.csv'))

if __name__ == '__main__':
    aggregate_adjacency_matrices() 