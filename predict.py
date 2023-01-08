from os.path import exists

import pandas as pd
import torch
import torch.nn
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

from config.configuration import GlobalConfig
from graph.graph import GraphGenerator
from model.GNN import GNNModel
from normalization.generator import DataGenerator
from joblib import dump, load
from model.metrics import f1
from visualization.graph_plotter import GraphPlotter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crit = torch.nn.CrossEntropyLoss()


def val_model(gnn_model, data, classes):
    with torch.no_grad():
        gnn_model.eval()
        result = gnn_model(data)
        _, prediction = torch.max(result, 1)
        y = data["flow_nodes"].y
        # f1_result = f1(y, prediction)
        print(prediction)

        # cf_matrix = confusion_matrix(y.cpu().numpy(), prediction.cpu().numpy())
        # df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
        #                     columns=[i for i in classes])
        # plt.figure(figsize=(12, 7))
        # sn.heatmap(df_cm, annot=True)
        # plt.show()

    return


def extract_attack(df_all: pd.DataFrame, label):
    index_result = df_all['Attack'] == label
    return df_all[index_result]


if __name__ == '__main__':

    config = GlobalConfig('config/nf_config.yaml', 'config/normalization_parameters.ini')
    str_classes = config.get('data.labels')
    choosen_features = config.get('data.chosen_features')
    train_epochs = config.get('model.train_epochs')
    normalize_features = config.get('data.normalize_features')
    all_features = config.get('data.all_features')

    n_classes = len(str_classes)
    n_features = len(choosen_features)

    model = GNNModel(input_chanels_size=n_features, chanels_hidden_size=128, output_chanels_size=n_classes,
                     config=config)

    if exists('./model_check_point.dat'):
        checkpoint = torch.load('model_check_point.dat')
        model.load_state_dict(checkpoint['model_state_dict'])

    # load data
    generator = DataGenerator(config)
    df = generator.val_data_set(1900000)

    print(set(list(df.loc[df["Attack"] == "dos"]["IPV4_DST_ADDR"])))
    print(set(list(df.loc[df["Attack"] == "Benign"]["IPV4_DST_ADDR"])))
    print("192.168.1.30" in set(list(df.loc[df["Attack"] == "Benign"]["IPV4_DST_ADDR"])))

    scaler = MinMaxScaler()
    only_transform = False
    if exists('./std_scaler2.bin'):
        scaler = load('./std_scaler2.bin')
        only_transform = True

    if only_transform:
        df[normalize_features] = scaler.transform(df[normalize_features])
    else:
        df[normalize_features] = scaler.fit_transform(df[normalize_features])

    df_Benign = extract_attack(df, 'Benign')
    df_Dos = extract_attack(df, 'dos')
    df_Benign = df_Benign.sample(n=20)
    df_Dos = df_Dos.sample(n=20)
    df_result = shuffle(pd.concat([df_Benign, df_Dos]))

    print(f'Muestras Dos {df_Dos.shape[0]}')
    print(f'Muestras Benign {df_Benign.shape[0]}')

    df_result = df_result.reset_index()
    df_result.drop(df_result.columns[[0]], axis=1, inplace=True)
    df_result.insert(0, 'Flow ID', 1000000 + df_result.index)
    df_result.columns = all_features
    # df_result = df_result.iloc[5:9, ]
    df_result.to_csv('./muestra.csv')
    # generate data
    graph_dataset = GraphGenerator(df_result, config)
    g, color_map = graph_dataset.generate_plot_graph()
    eval_data = graph_dataset.create_graph()

    model.to(device)
    eval_data.to(device)
    val_model(model, eval_data, str_classes)

    ploter1 = GraphPlotter(g)
    ploter1.plot_graph(g, color_map)
