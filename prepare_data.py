from os.path import exists

import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

from config.configuration import GlobalConfig


def read_csv(path):
    df_all = pd.read_csv(path, header=0)
    return df_all


def extract_attack(df_all: pd.DataFrame):
    index_result = df_all['Attack'] != 'Benign'
    return df_all[index_result]


def extract_Benign(df_all: pd.DataFrame):
    index_result = df_all['Attack'] == 'Benign'
    return df_all[index_result]


if __name__ == '__main__':
    config = GlobalConfig('config/nf_config.yaml', 'config/normalization_parameters.ini')
    columns = config.get("data.all_features")
    normalize_features = config.get('data.normalize_features')
    df = read_csv('./data/NF-DataSet/NF-UNSW-NB15-v2.csv')

    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    only_transform = False
    if exists('./std_scaler2.bin'):
        scaler = load('./std_scaler2.bin')
        only_transform = True

    if only_transform:
        df[normalize_features] = scaler.transform(df[normalize_features])
    else:
        df[normalize_features] = scaler.fit_transform(df[normalize_features])

    df_Benign = extract_Benign(df)
    l_b = len(df_Benign)
    df_Dos = extract_attack(df)

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        # get samples
        df_Benign_result = df_Benign.sample(n=20000)
        df_Dos_result = df_Dos.sample(n=10000)
        df_result = shuffle(pd.concat([df_Benign_result, df_Dos_result]))
        df_result = df_result.reset_index()
        df_result.drop(df_result.columns[[0]], axis=1, inplace=True)

        # generate data
        df_result.insert(0, 'Flow ID', 1000000 + df_result.index)
        df_result.to_csv(f'./data/NF-DataSet/NF_DATA_DoS_normalized_{str(i)}.csv', index=False, header=False)

    dump(scaler, './std_scaler.bin', compress=True)
