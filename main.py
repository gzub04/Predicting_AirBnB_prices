import sys
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import glob



# RANDOM_SEED = 52

# random forest parameters
MAX_DEPTH = None
MIN_IMPURITY_DECREASE = 0.05


def split_data(dataset):
    x = dataset.iloc[:, 2:].values
    y = dataset.iloc[:, 1].values

    # preprocess data
    numerical_columns = [3]
    non_numerical_columns = [0, 1, 2, 4]

    numerical_transformer = StandardScaler()
    non_numerical_transformer = OrdinalEncoder()
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', non_numerical_transformer, non_numerical_columns)
    ])
    preprocessed_x = preprocessor.fit_transform(x)

    return train_test_split(preprocessed_x, y, test_size=.2)  # random_state=RANDOM_SEED)


def root_squared_mean_error(regressor, x_test, y_test):
    y_pred = regressor.predict(x_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


def load_files():
    file_list = glob.glob("Datasets/*.csv")
    dataframes = []
    for file in file_list:
        df = pd.read_csv(file)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def run_algorithms(df):

    x_train, x_test, y_train, y_test = split_data(df)

    # random forest regression
    random_forest_regression = RandomForestRegressor(
        max_depth=MAX_DEPTH, min_impurity_decrease=MIN_IMPURITY_DECREASE)  # , random_state=RANDOM_SEED)
    random_forest_regression.fit(x_train, y_train)

    # linear regression
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)

    # print(f'Random forest score from scikit-learn = {random_forest_regression.score(x_test, y_test)}')
    # print(f'Linear regression score from scikit-learn = {linear_regression.score(x_test, y_test)}')
    # print(f'Random forest RMSE = {root_squared_mean_error(random_forest_regression, x_test, y_test)}')
    # print(f'Linear regression RMSE = {root_squared_mean_error(linear_regression, x_test, y_test)}')
    return [random_forest_regression.score(x_test, y_test), linear_regression.score(x_test, y_test)]


def main():
    df = load_files()
    sum_random_forest = 0
    sum_linear = 0
    num_of_iterations = 10
    for i in range(num_of_iterations):
        progress = f"Progress: {i}/{num_of_iterations}"
        sys.stdout.write('\r' + progress)
        sys.stdout.flush()
        results = run_algorithms(df)
        sum_random_forest += results[0]
        sum_linear += results[1]
    sys.stdout.write('\r' + f'Progress: {num_of_iterations}/{num_of_iterations}\n')
    average_random_forest = sum_random_forest / num_of_iterations
    average_linear = sum_linear / num_of_iterations
    print(f'Average of scores from scikit-learn for Random Forest = {average_random_forest}')
    print(f'Average of scores from scikit-learn for Linear Regression = {average_linear}')


if __name__ == '__main__':
    main()
