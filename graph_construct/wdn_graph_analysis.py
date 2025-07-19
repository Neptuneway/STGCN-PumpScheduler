import os
import wntr
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from config import INP_FILE_PATH, PRESSURE_SENSORS, AGGREGATION_THRESHOLD

INPUT_DIR = "adjacency_matrices"

class WaterNetworkGraphAnalyzer:
    """
    Analyze and visualize water distribution network graphs.
    """
    def __init__(self, inp_file, time_step):
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.time_step = time_step

    def get_node_positions(self, monitors):
        g = self.wn.to_graph(link_weight=self.wn.query_link_attribute('length'))
        pos = nx.get_node_attributes(g, 'pos')
        ps = [pos[node] for node in monitors]
        return {i: ps[i] for i in range(len(ps))}

    def plot_graph(self, adj_matrix, positions, output_file):
        G = nx.from_numpy_array(adj_matrix)
        G.remove_edges_from(nx.selfloop_edges(G))
        nx.draw(G, positions, node_size=10, arrowsize=8, linewidths=0.8)
        plt.savefig(output_file, dpi=1000)
        plt.close()

if __name__ == '__main__':
    # Read the final aggregated adjacency matrix
    agg_file = os.path.join(INPUT_DIR, f'final_adjacency_exp_{AGGREGATION_THRESHOLD}_undirected.csv')
    adj = pd.read_csv(agg_file, index_col=0).values
    # Get node positions from the original network (using time_step=0)
    analyzer = WaterNetworkGraphAnalyzer(INP_FILE_PATH, 0)
    pos_dict = analyzer.get_node_positions(PRESSURE_SENSORS)
    analyzer.plot_graph(adj, pos_dict, os.path.join(INPUT_DIR, 'final_graph.png')) 