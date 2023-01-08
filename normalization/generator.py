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


class DataGenerator:

    def __init__(self, config: GlobalConfig):
        self.global_config = config
        self.normalization_params = self.global_config.normalization_params
        self.chosen_features_dict = self.global_config.get('data.chosen_features')
        self.all_features = self.global_config.get('data.all_features')

    def normalization_df(self, df: pd.DataFrame):
        param = self.normalization_params
        for _, row in df.iterrows():
            features = row[["Flow ID", "Source IP"]]
            print(len(features.values))

        return df

    def train_data_set(self, path, sample_rows=0):
        #path = self.global_config.get('data.training_path')
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
