"""
    Basado en la investigación de Gillermo Cobo (guillermo.cobo1998@gmail.com)
    Generacion del Grafo usando HeteroData y la siguinte estructura
    Source IP <=> IP:PORT <=> Flow <=> IP:PORT <=> Destination IP
"""

import math
import torch
import networkx as nx
from config.configuration import GlobalConfig
from torch_geometric.data import HeteroData
from pandas import DataFrame
from torch_geometric.data import HeteroData


class GraphGenerator:

    def __init__(self, data: DataFrame, config: GlobalConfig):

        self.data = data
        self.config = config
        self.all_labels = config.get("data.labels")
        self.chosen_features = config.get("data.chosen_features")
        self.graph_data = HeteroData()

        indices = range(len(self.all_labels))
        zip_iterator = zip(self.all_labels, indices)
        self.label_dict = dict(zip_iterator)

        indices = range(len(self.chosen_features))
        zip_iterator = zip(self.chosen_features, indices)
        self.chosen_features_dict = dict(zip_iterator)

    def generate_family_nodes(self):

        unique_index = set()
        idx = 0
        node_map = {}

        # get node names
        family_nd_name = self.config.get("graph.family_id")
        # group_nd_name = self.config.get("graph.group")

        # extraer direcciones Source y Destination
        for index, row in self.data.iterrows():
            family_nd = row[family_nd_name]
            # group_nd = row[group_nd_name]
            # indexar
            if family_nd not in unique_index:
                unique_index.add(family_nd)
                node_map[family_nd] = idx
                idx = idx + 1
            # if group_nd not in unique_index:
            #     unique_index.add(group_nd)
            #     node_map[group_nd] = idx
            #     idx = idx + 1

        return node_map

    def generate_group_nodes(self):

        unique_index = set()
        idx = 0
        node_map = {}

        group_nd_name = self.config.get("graph.group")

        # extraer combinaciones SourceIp:port y DestinationIp:Port
        for index, row in self.data.iterrows():
            group_nd = row[group_nd_name]
            # group_nd = row[group_nd_name]
            # indexar
            if group_nd not in unique_index:
                unique_index.add(group_nd)
                node_map[group_nd] = idx
                idx = idx + 1

        return node_map

    def get_chosen_features(self, row):
        # normalize features
        result_values = []
        for feature in self.chosen_features:
            value = row[feature]
            try:
                f_value = float(value)
                if not math.isinf(f_value):
                    result_values.append(f_value)
                else:
                    print(f'get_chosen_features  feature-> {feature} value-> {f_value}')
                    result_values.append(0)
            except Exception:
                print(f'get_chosen_features Exception row {feature} feature-> {feature}')
                result_values.append(0)

        return result_values

    def get_labels_values(self, row):
        impact_label_node = self.config.get("graph.label_node")
        return row[impact_label_node]
        # if attack.lower() == 'benign':
        #     return 1
        # else:
        #     return 0

        # result = self.label_dict.get(attack.lower())
        # if result is None:
        #    return 0  # Other
        # return result

    def generate_flow_nodes(self):

        unique_index = set()
        idx = 0
        node_map = {}
        features, labels = [], []
        flow_node_name = self.config.get("graph.flow_node")

        # indexar
        for index, row in self.data.iterrows():
            flow_id = row[flow_node_name]
            if flow_id not in unique_index:
                unique_index.add(flow_id)
                node_map[flow_id] = idx
                idx = idx + 1
                features.append(self.get_chosen_features(row))
                labels.append(self.get_labels_values(row))

        return torch.FloatTensor(features), torch.LongTensor(labels), node_map

    def generate_adjacencie(self, family_map, group_map, dates_map):

        family_nd_name = self.config.get("graph.family_id")
        group_nd_name = self.config.get("graph.group")
        date_node_name = self.config.get("graph.flow_node")

        src_0, src_1, src_2, src_3 = [], [], [], []
        dest_0, dest_1, dest_2, dest_3 = [], [], [], []

        for _, row in self.data.iterrows():
            family_id = row[family_nd_name]
            # source_ip_port = (row[src_ip_nd_name], row[src_ip_port_nd_name])
            group = row[group_nd_name]
            # destination_ip_port = (row[dst_ip_nd_name], row[dst_ip_port_nd_name])
            date_connection = row[date_node_name]

            # Source IP <=> IP:PORT <=> Flow <=> IP:PORT <=> Destination IP
            # conectar nodos IP => nodos IP:PORT
            # conectar family id => group os fingerprint
            src_0.append(family_map[family_id])
            dest_0.append(group_map[group])
            # src_0.append(ip_map[destination_ip])
            # dest_0.append(ip_port_map[destination_ip_port])

            # conectar nodos IP:PORT =>  nodos FLOW
            src_1.append(family_map[family_id])
            dest_1.append(dates_map[date_connection])
            # src_1.append(ip_port_map[destination_ip_port])
            # dest_1.append(dates_map[date_connection])

            # conectar  nodos FLOW => IP:PORT
            src_2.append(dates_map[date_connection])
            dest_2.append(group_map[group])
            # src_2.append(flows_map[flow_connection])
            # dest_2.append(ip_port_map[destination_ip_port])

            # conectar nodos IP:PORT => IP
            # src_3.append(ip_port_map[source_ip_port])
            # dest_3.append(ip_map[source_ip])
            # src_3.append(ip_port_map[destination_ip_port])
            # dest_3.append(ip_map[destination_ip])

        return torch.LongTensor([src_0, dest_0]), torch.LongTensor([src_1, dest_1]), torch.LongTensor(
            [src_2, dest_2])  # , torch.LongTensor([src_3, dest_3])

    def create_graph(self) -> HeteroData:
        """
        :return: El grafo encapsulado en un objeto heterogeneo
        """
        finger_nodes = self.generate_family_nodes()
        group_nodes = self.generate_group_nodes()
        features_flows, labels_flows, flow_nodes = self.generate_flow_nodes()
        family_to_group, family_to_flow, flow_to_group = self.generate_adjacencie(finger_nodes, group_nodes,
                                                                                  flow_nodes)

        result_data = HeteroData()

        # crear tensores basados en el numero de ip address, initial state dimesion
        family_x = torch.ones(len(finger_nodes.keys()), 128)
        group_x = torch.ones(len(group_nodes.keys()), 128)

        # node types
        result_data["family_nodes"].x = family_x
        result_data["group_nodes"].x = group_x
        result_data["flow_nodes"].x = features_flows
        result_data["flow_nodes"].y = labels_flows

        # adjacencie
        result_data["family", "to", "group"].edge_index = family_to_group
        result_data["family", "to", "flow"].edge_index = family_to_flow
        result_data["flow", "to", "group"].edge_index = flow_to_group
        # result_data["ip_port", "to", "ip"].edge_index = port_to_ip

        self.graph_data = result_data
        return result_data

    def generate_plot_graph(self):
        g = nx.DiGraph()
        family_nodes = self.generate_family_nodes()
        group_nodes = self.generate_group_nodes()
        _, _, flow_nodes = self.generate_flow_nodes()
        colors = []

        family_nd_name = self.config.get("graph.family_id")
        group_nd_name = self.config.get("graph.group")
        # src_ip_port_nd_name = self.config.get("graph.source_port_node")
        # dst_ip_port_nd_name = self.config.get("graph.dest_port_node")
        flow_node_name = self.config.get("graph.flow_node")

        for n in family_nodes.keys():
            g.add_node(family_nodes[n])
            colors.append('blue')

        for n in group_nodes.keys():
            g.add_node(n)
            colors.append('green')

        for n in flow_nodes.keys():
            g.add_node(n)
            colors.append('yellow')

        for _, row in self.data.iterrows():
            family = family_nodes[row[family_nd_name]]
            group = group_nodes[row[group_nd_name]]
            # destination_ip = ip_nodes[row[dst_ip_nd_name]]
            # destination_ip_port = (row[dst_ip_nd_name], row["L4_DST_PORT"])
            flow_connection = row[flow_node_name]

            g.add_edge(family, group)
            g.add_edge(family, flow_connection)
            g.add_edge(flow_connection, group)
            # g.add_edge(destination_ip_port, destination_ip)

        return g, colors
