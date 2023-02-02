import datetime
import random
import numpy as np

import pandas as pd
import configparser
from config.configuration import GlobalConfig

__author__ = "NetworkSecurityLab"


def read_csv(path, column_names, sample):
    n_rows = sample
    if n_rows == 0:
        n_rows = None

    df = pd.read_csv(path, sep=',', header=None, nrows=n_rows)
    df.columns = column_names

    return df


class DataGeneratorEco:

    def __init__(self, config: GlobalConfig):
        self.global_config = config
        self.normalization_params = self.global_config.normalization_params
        self.chosen_features_dict = self.global_config.get('data.chosen_features')
        self.all_features = self.global_config.get('data.all_features')
        self.groups = self.global_config.get('data.groups')

    def normalization_df(self, df: pd.DataFrame):
        param = self.normalization_params
        for _, row in df.iterrows():
            features = row[["Flow ID", "Source IP"]]
            print(len(features.values))

        return df

    def train_data_set(self, sample_rows=0):
        path = self.global_config.get('data.training_path')
        df = read_csv(path, self.all_features, sample_rows)
        return df

    def train_data_set_parquet(self):
        path = self.global_config.get('data.training_parquet_path')
        df = pd.read_parquet(path)
        df.columns = self.all_features
        return df

    def val_data_set(self, sample_rows=0):
        path = self.global_config.get('data.validation_path')
        df = pd.read_csv(path, header=0, nrows=sample_rows)
        return df

    def gen_data(self, eval=0):
        ndays: int = 30
        family: int = 20 if eval == 0 else 1
        famimembers: int = np.random.choice(range(2, 6), family) if eval == 0 else [4]  # random.sample(range(2, 4), 20)
        result = []
        for day in range(1, ndays):
            for j in range(family):
                for i in range(1, 4):
                    item: int = np.random.randint(1, 9) \
                        if i == 1 else np.random.randint(1, 4) \
                        if i == 2 else 1 \
                        if i == 3 else np.random.randint(1, 3)
                    self.gen_grupo_data(result, j+1, famimembers[j], i, day, item)
        path = self.global_config.get('data.training_path' if eval == 0 else 'data.validation_path')
        df = pd.DataFrame(result).to_csv(path, header=False, index=False)
        return df

    def gen_grupo_data(self, result, famid, nfami, grupo, day, nitems):
        for i in range(nitems):
            a = []
            dt = datetime.datetime(2022, 10, day, np.random.randint(24), np.random.randint(60), np.random.randint(60))
            a.extend(['Family '+str(famid), self.groups[grupo-1], np.random.randint(3), dt,
                           np.random.randint(1, 4)])
            if grupo == 1:
                a.append(np.random.randint(1, nfami+1))
            else:
                a.append(1)
            a.append(self.calculate_label(grupo-1, a[2], a[4]-1))
            result.append(a)
        return result

    def calculate_label(self, group, typetrans, value):
        wgroup: list[int] = [100, 60, 40, 20]
        wtrans: list[int] = [0, 50, 100]
        wvalue: list[int] = [10, 50, 100]
        n = wgroup[group] + wtrans[typetrans] + wvalue[value]
        if n <= 70:
            return 0
        elif 70 < n <= 150:
            return 1
        return 2
