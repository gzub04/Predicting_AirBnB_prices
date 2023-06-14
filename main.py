from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd



# RANDOM_SEED = 52

# random forest parameters
MAX_DEPTH = None
MIN_IMPURITY_DECREASE = 0.05

# gradient boositng parameters
N_ESTIMATORS_GRADIENT = 100
MAX_DEPTH_GRADIENT = 5
MIN_SAMPLES_SPLIT_GRADIENT = 10
LEARNING_RATE_GRADIENT=  0.1
LOSS_GRADIENT = "squared_error"


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


def main():

    df = pd.read_csv('Datasets/london_weekends.csv')
    x_train, x_test, y_train, y_test = split_data(df)

    # random forest regression
    random_forest_regression = RandomForestRegressor(
        max_depth=MAX_DEPTH, min_impurity_decrease=MIN_IMPURITY_DECREASE)  # , random_state=RANDOM_SEED)
    random_forest_regression.fit(x_train, y_train)

    # linear regression
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)

    #gradient boosting regression
    gradient_boositing_regressor = GradientBoostingRegressor(max_depth=MAX_DEPTH_GRADIENT, n_estimators=N_ESTIMATORS_GRADIENT, learning_rate=LEARNING_RATE_GRADIENT,loss=LOSS_GRADIENT,min_samples_split=MIN_SAMPLES_SPLIT_GRADIENT)
    gradient_boositing_regressor.fit(x_train, y_train)

    print(f'Random forest score from scikit-learn = {random_forest_regression.score(x_test, y_test)}')
    print(f'Linear regression score from scikit-learn = {linear_regression.score(x_test, y_test)}')
    print(f'Gradient boosting score from scikit-learn = {gradient_boositing_regressor.score(x_test, y_test)}')
    print(f'Random forest RMSE = {root_squared_mean_error(random_forest_regression, x_test, y_test)}')
    print(f'Linear regression RMSE = {root_squared_mean_error(linear_regression, x_test, y_test)}')
    print(f'Gradient boosting RMSE = {root_squared_mean_error(gradient_boositing_regressor, x_test, y_test)}')
    return random_forest_regression.score(x_test, y_test)


if __name__ == '__main__':
    sum_of_scores = 0
    num_of_iterations = 40
    for i in range(num_of_iterations):
        sum_of_scores += main()
    average = sum_of_scores/num_of_iterations
    print(average)
