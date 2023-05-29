from random import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


RANDOM_SEED = 41


def split_data(dataset):
    x = dataset.iloc[:, 2:].values
    y = dataset.iloc[:, 1].values.reshape(-1, 1)
    return train_test_split(x, y, test_size=.2, random_state=RANDOM_SEED)


def main():
    random.seed(RANDOM_SEED)
    df = pd.read_csv('Datasets/london_weekends.csv')
    x_train, x_test, y_train, y_test = split_data(df)


if __name__ == '__main__':
    main()

