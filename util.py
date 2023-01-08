import pandas as pd


def read_csv(path):
    df = pd.read_csv(path, header=None)
    return df
