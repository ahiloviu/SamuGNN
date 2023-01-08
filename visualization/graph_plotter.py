import pandas as pd
import configparser
import networkx as nx
import matplotlib.pyplot as plt

from config.configuration import GlobalConfig


class GraphPlotter:
    def __init__(self, config: GlobalConfig):
        self.global_config = config

    def plot_graph(self, g, colors):
        #pos = nx.get_node_attributes(g, 'pos')
        nx.draw_networkx(g, node_color=colors, arrows=True, pos=None)
        plt.margins(0.1)
        plt.figure(figsize=(200, 200))
        plt.axis('equal')
        plt.show()
