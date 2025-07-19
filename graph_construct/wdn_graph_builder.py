import os
import wntr
import numpy as np
import pandas as pd
import networkx as nx
from config import INP_FILE_PATH, PRESSURE_SENSORS, SIM_DURATION_HOURS, TIME_STEP

OUTPUT_DIR = "adjacency_matrices"

class WaterNetworkGraphBuilder:
    """
    Build and process water distribution network graphs based on hydraulic simulations.
    """
    def __init__(self, inp_file, time_step):
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.time_step = time_step

    def get_reverse_pipes(self):
        """
        Identify pipes with reverse flow at the current time step.
        """
        original_duration = self.wn.options.time.duration
        self.wn.options.time.duration = 3600 * SIM_DURATION_HOURS
        sim = wntr.sim.EpanetSimulator(self.wn)
        result = sim.run_sim()
        self.wn.options.time.duration = original_duration
        self.wn.reset_initial_values()
        pipes = result.link['flowrate'].iloc[self.time_step, :][result.link['flowrate'].iloc[self.time_step, :] < 0].keys()
        return pipes

    def build_directed_graph(self):
        """
        Build a directed graph based on the hydraulic model and adjust edge directions according to flow.
        """
        pipes = self.get_reverse_pipes()
        length = self.wn.query_link_attribute('length')
        g = self.wn.to_graph(link_weight=length)
        g_copy = self.wn.to_graph(link_weight=length)
        for edge in g_copy.edges:
            for pipe in pipes:
                if pipe == edge[2]:
                    length_origin = length.loc[edge[2]]
                    g.remove_edge(edge[0], edge[1])
                    g.add_edge(edge[1], edge[0], weight=length_origin)
        return g

    def build_monitor_adjacency(self, monitors):
        """
        Build the adjacency matrix for the set of monitoring nodes.
        """
        g = self.build_directed_graph()
        pos = nx.get_node_attributes(g, 'pos')
        ps = [pos[node] for node in monitors]
        graph = np.zeros((len(monitors), len(monitors)))
        for i, s in enumerate(monitors):
            for j, t in enumerate(monitors):
                if nx.has_path(g, s, t):
                    path = nx.dijkstra_path(g, s, t)
                    if len(list(set(monitors) & set(path))) <= 2:
                        graph[i, j] = round(nx.shortest_path_length(g, source=s, target=t, weight='weight', method="dijkstra"), 2)
        return graph, ps

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i in range(0, SIM_DURATION_HOURS, TIME_STEP):
        builder = WaterNetworkGraphBuilder(INP_FILE_PATH, i)
        adj, ps = builder.build_monitor_adjacency(PRESSURE_SENSORS)
        adj = np.asarray(adj)
        pd.DataFrame(adj).to_csv(os.path.join(OUTPUT_DIR, f'{i}_monitor_adjacency_weighted.csv')) 