from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd


RANDOM_SEED = 52

# random forest parameters
MAX_DEPTH = None
MIN_IMPURITY_DECREASE = 0.05


def split_data(dataset):
    x = dataset.iloc[:, 2:].values
    y = dataset.iloc[:, 1].values

    # preprocess data
    numerical_columns = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    non_numerical_columns = [0, 1, 2, 4]

    numerical_transformer = StandardScaler()
    non_numerical_transformer = OrdinalEncoder()
    preprocressor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', non_numerical_transformer, non_numerical_columns)
    ])
    preprocessed_x = preprocressor.fit_transform(x)

    return train_test_split(preprocessed_x, y, test_size=.2, random_state=RANDOM_SEED)


def main():
    random.seed(RANDOM_SEED)
    df = pd.read_csv('Datasets/london_weekends.csv')
    x_train, x_test, y_train, y_test = split_data(df)

    # random forest regression
    random_forest_regression = RandomForestRegressor(
        max_depth=MAX_DEPTH, min_impurity_decrease=MIN_IMPURITY_DECREASE, random_state=RANDOM_SEED)
    random_forest_regression.fit(x_train, y_train)
    print(f'{random_forest_regression.score(x_test, y_test)}')


if __name__ == '__main__':
    main()

