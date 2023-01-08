# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from os.path import exists

import numpy as np
import torch
import torch.nn
import torch_geometric.transforms as t

from config.configuration import GlobalConfig
from graph.graph import GraphGenerator
from model.GNN import GNNModel
from normalization.generator import DataGenerator
import model.metrics as m
from sklearn.utils import class_weight

import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


def optimizer_to(optim):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def val_model(gnn_model, data, labels):
    with torch.no_grad():
        gnn_model.eval()
        result = gnn_model(data)
        _, pred = torch.max(result, 1)
        label = data["flow_nodes"].y

    return m.accuracy(label, pred)


def test_model(gnn_model, data):
    with torch.no_grad():
        gnn_model.eval()
        mask = data["flow_nodes"].test_mask
        result = gnn_model(data)
        _, pred = torch.max(result[mask], 1)
        label = data["flow_nodes"].y[mask]
    return m.accuracy(label.cpu(), pred.cpu())


def train_model(gnn_model, data, optim):
    gnn_model.train()
    optim.zero_grad()
    mask = data["flow_nodes"].train_mask
    out = gnn_model(data)
    loss = crit(out[mask], data["flow_nodes"].y[mask])
    loss.backward()
    optim.step()

    return float(loss)


def run_model_epochs(gnn_model, data, optim, epochs):
    best_accuracy = 0.0
    losses_arr = []
    gnn_model.to(device)

    for epoch in range(0, epochs):
        # if epoch == 0:
        #      for param_group in optimizer.param_groups:
        #          param_group['lr'] = 0.001
        #
        # if epoch == 400:
        #      for param_group in optimizer.param_groups:
        #          param_group['lr'] = 0.0001

        loss = train_model(gnn_model, data, optim)
        losses_arr.append(loss)
        metrics = test_model(gnn_model, data)
        print(str(datetime.datetime.now()) + ' Epoch ' + str(epoch) + ':{ Loss: ' + str(loss) + ' }  ' + str(metrics))

    return losses_arr


def run_model_validation(model, val_data, str_classes):
    pass


if __name__ == '__main__':

    config = GlobalConfig('config/nf_config.yaml', 'config/normalization_parameters.ini')
    str_classes = config.get('data.labels')
    choosen_features = config.get('data.chosen_features')
    train_epochs = config.get('model.train_epochs')

    n_classes = len(str_classes)
    n_features = len(choosen_features)

    # generate data
    generator = DataGenerator(config)

    # df = generator.train_data_set_parquet() , 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

    for i in [1]:
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            df = generator.train_data_set(f'./data/NF-DataSet/NF_DATA_DoS_normalized_{str(i)}.csv')
            # create graph
            graph_dataset = GraphGenerator(df, config)
            train_data = graph_dataset.create_graph()

            classes = np.unique(train_data["flow_nodes"].y.numpy())
            classes_data = train_data["flow_nodes"].y.numpy()
            class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=classes_data)
            crit = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

            # Split data
            # Ramdom split devuelve una tupla de [train_mask, val_mask, test_mask] por cada nodo
            split_t = t.Compose([t.RandomNodeSplit(num_val=0.0, num_test=0.15)])
            train_data = split_t(train_data)
            train_data.to(device)

            model = GNNModel(input_chanels_size=n_features, chanels_hidden_size=128, output_chanels_size=n_classes,
                             config=config)

            # TODO
            if exists('./model_check_point.dat'):
                checkpoint = torch.load('model_check_point.dat', map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # , weight_decay=1e-5
            if exists('./model_check_point.dat'):
                checkpoint = torch.load('model_check_point.dat', map_location=device)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # optimizer_to(optimizer)
            print(f'Training start data -> {str(i)} ')
            losses = run_model_epochs(model, train_data, optimizer, train_epochs)
            print("Training end data -> {str(i)}")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'model_check_point.dat')

    # print("Validation start ...")
    # df = generator.val_data_set()
    # val_dataset = GraphGenerator(df, config)
    # val_data = val_dataset.create_graph()
    # run_model_validation(model, val_data, str_classes)
    # print("Validation end")
