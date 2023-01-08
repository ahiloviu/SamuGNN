import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from config.configuration import GlobalConfig

# mean function
from torch_scatter import scatter


class GNNModel(nn.Module):
    """
        Implemnetacion del modelo
        Basado en MPNN ->  Basado en la investigación de Gillermo Cobo (guillermo.cobo1998@gmail.com)
        Parameters
        ----------

        input_chanels_size : int
            Size of the hidden states (the input will be padded with 0's to this size).

        chanels_hidden_size : int
            Message function output vector size.

        output_chanels_size : int
            Size of the output.

        config : GlobalConfig
            general config


    """

    def __init__(self, input_chanels_size, chanels_hidden_size, output_chanels_size, config: GlobalConfig):
        super().__init__()

        # store variables
        self.input_chanels = input_chanels_size
        self.output_chanels = output_chanels_size
        self.hidden_chanels_size = chanels_hidden_size
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

        # Define message functions ( sequential container )
        # valor dropout de ebtrada
        self.message_func_ip_nodes = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(self.hidden_chanels_size * 2, self.hidden_chanels_size),
        )
        self.message_func_flow_nodes = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(self.hidden_chanels_size * 2, self.hidden_chanels_size)
        )

        # update
        self.update_ip_nodes = nn.GRU(self.hidden_chanels_size, self.hidden_chanels_size)
        self.update_flow_nodes = nn.GRU(self.hidden_chanels_size, self.hidden_chanels_size)

        # readout
        self.readout_func = nn.Sequential(
            nn.Linear(self.hidden_chanels_size, self.hidden_chanels_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_chanels_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, self.output_chanels),
            nn.Softmax(dim=1)
        )

    def forward(self, data: HeteroData):
        # Iterations
        n_iteration = self.config.get('model.number_iterations')

        # Get flow data
        n_flow_nodes = data['flow_nodes'].x.shape[0]

        # node connections
        source_ip_to_ip_port, dest_ip_to_ip_port = data["ip", "to", "ip_port"].edge_index
        source_ip_port_to_flow, dest_ip_port_to_flow = data["ip_port", "to", "flow"].edge_index
        source_flow_to_ip_port, dest_flow_to_ip_port = data["flow", "to", "ip_port"].edge_index
        source_ip_port_to_ip, dest_ip_port_to_ip = data["ip_port", "to", "ip"].edge_index

        # IP node initialization
        h_ip = data["ip_nodes"].x
        h_ip_port = data["ip_port_nodes"].x

        # Flows node initialization
        h_flows = torch.cat((data["flow_nodes"].x,
                             torch.zeros(n_flow_nodes,
                                         self.hidden_chanels_size - self.input_chanels).to(self.device)), dim=1)

        for _ in range(n_iteration):
            # IP -> IP:Port
            ip_data = h_ip[source_ip_to_ip_port]
            ip_port_data = h_ip_port[dest_ip_to_ip_port]

            # De tf a pt
            # apply the message function on the ip nodes
            nn_input = torch.cat((ip_data, ip_port_data), dim=1).float()
            ip_to_ip_port_message = self.message_func_ip_nodes(nn_input)
            ip_to_ip_port_mean = scatter(ip_to_ip_port_message, dest_ip_to_ip_port, dim=0, reduce="mean")

            # IP:Port -> IP
            ip_port_data = h_ip_port[source_ip_port_to_ip]
            ip_data = h_ip[dest_ip_port_to_ip]

            nn_input = torch.cat((ip_port_data, ip_data), dim=1).float()
            ip_port_to_ip_message = self.message_func_ip_nodes(nn_input)
            ip_port_to_ip_mean = scatter(ip_port_to_ip_message, dest_ip_port_to_ip, dim=0, reduce="mean")

            # IP:Port -> Flow
            ip_port_data = h_ip_port[source_ip_port_to_flow]
            flow_data = h_flows[dest_ip_port_to_flow]
            nn_input = torch.cat((ip_port_data, flow_data), dim=1).float()

            ip_port_to_flow_message = self.message_func_flow_nodes(nn_input)
            ip_port_to_flow_mean = scatter(ip_port_to_flow_message, dest_ip_port_to_flow, dim=0, reduce="mean")

            # Flow to IP:Port
            flow_data = h_flows[source_flow_to_ip_port]
            ip_port_data = h_ip_port[dest_flow_to_ip_port]
            nn_input = torch.cat((flow_data, ip_port_data), dim=1).float()

            flow_to_ip_port_message = self.message_func_flow_nodes(nn_input)
            flow_to_ip_port_mean = scatter(flow_to_ip_port_message, dest_flow_to_ip_port, dim=0, reduce="mean")

            # Upate nodes
            n_result = self.update_ip_nodes(ip_port_to_ip_mean.unsqueeze(0), h_ip.unsqueeze(0))
            _, new_h_ip = n_result
            h_ip = new_h_ip[0]

            _, new_h_flows = self.update_flow_nodes(ip_port_to_flow_mean.unsqueeze(0), h_flows.unsqueeze(0))
            h_flows = new_h_flows[0]

            _, new_h_ip_port = self.update_ip_nodes(flow_to_ip_port_mean.unsqueeze(0), h_ip_port.unsqueeze(0))
            h_ip_port = new_h_ip_port[0]

            _, new_h_ip_port = self.update_ip_nodes(ip_to_ip_port_mean.unsqueeze(0), h_ip_port.unsqueeze(0))
            h_ip_port = new_h_ip_port[0]

        return self.readout_func(h_flows)
