# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import random

from config.configuration import GlobalConfig
from normalization.generator import DataGenerator
from graph.graph import GraphGenerator
from visualization.graph_plotter import GraphPlotter

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config = GlobalConfig('config/global_config.yaml', 'config/normalization_parameters.ini')
    generator = DataGenerator(config)
    generator.gen_data()
    df = generator.train_data_set(sample_rows=10)
    graph_dataset = GraphGenerator(df, config)
    graph_dataset.create_graph()
    g, color_map = graph_dataset.generate_plot_graph()
    ploter = GraphPlotter(g)
    ploter.plot_graph(g, color_map)

