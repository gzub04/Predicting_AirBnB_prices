import sys
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
import glob
import matplotlib.pyplot as plt


NUM_OF_ITERATIONS = 20  # how many times should each ML algorithm be run

# random forest parameters
MAX_DEPTH = None
MIN_IMPURITY_DECREASE = 0.05

# gradient boosting parameters
N_ESTIMATORS_GRADIENT = 100
MAX_DEPTH_GRADIENT = 5
MIN_SAMPLES_SPLIT_GRADIENT = 10
LEARNING_RATE_GRADIENT = 0.1
LOSS_GRADIENT = "squared_error"

dataset_names = []

forest_scores = []
forest_scores_rms = []

linear_scores = []
linear_scores_rms = []

gradient_scores = []
gradient_scores_rms = []

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
        dataset_names.append(file.title())
        df = pd.read_csv(file)
        dataframes.append(df)
    return dataframes


def run_algorithms(df):

    x_train, x_test, y_train, y_test = split_data(df)

    # random forest regression
    random_forest_regression = RandomForestRegressor(
        max_depth=MAX_DEPTH, min_impurity_decrease=MIN_IMPURITY_DECREASE)  # , random_state=RANDOM_SEED)
    random_forest_regression.fit(x_train, y_train)

    # linear regression
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)

    gradient_boosting_regressor = GradientBoostingRegressor(max_depth=MAX_DEPTH_GRADIENT,
                                                            n_estimators=N_ESTIMATORS_GRADIENT,
                                                            learning_rate=LEARNING_RATE_GRADIENT, loss=LOSS_GRADIENT,
                                                            min_samples_split=MIN_SAMPLES_SPLIT_GRADIENT)
    gradient_boosting_regressor.fit(x_train, y_train)

    # print(f'Random forest score from scikit-learn = {random_forest_regression.score(x_test, y_test)}')
    # print(f'Linear regression score from scikit-learn = {linear_regression.score(x_test, y_test)}')
    # print(f'Random forest RMSE = {root_squared_mean_error(random_forest_regression, x_test, y_test)}')
    # print(f'Linear regression RMSE = {root_squared_mean_error(linear_regression, x_test, y_test)}')

    forest_scores.append(random_forest_regression.score(x_test, y_test))
    linear_scores.append(linear_regression.score(x_test, y_test))
    gradient_scores.append(gradient_boosting_regressor.score(x_test, y_test))

    forest_scores_rms.append(root_squared_mean_error(random_forest_regression, x_test, y_test))
    linear_scores_rms.append(root_squared_mean_error(linear_regression, x_test, y_test))
    gradient_scores_rms.append(root_squared_mean_error(gradient_boosting_regressor, x_test, y_test))

    return [random_forest_regression.score(x_test, y_test), linear_regression.score(x_test, y_test),
            gradient_boosting_regressor.score(x_test, y_test)]

def plotScores():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    dataset_indexes = np.arange(len(dataset_names))
    
    #plt.xticks(dataset_indexes, dataset_names)

    #ax = fig.add_subplot(211)

    ax.plot(dataset_indexes,forest_scores, "ob", label = "Random Forest")
    ax.plot( dataset_indexes,linear_scores, "or", label = "Linear Regression")
    ax.plot( dataset_indexes, gradient_scores, "og", label = "Gradient Boosting")
    ax.set_xticks(dataset_indexes)
    ax.set_xticklabels(dataset_names, rotation='vertical')
    ax.set_title("Score")
    ax.set_ylim(top = 1.0, bottom = 0)
    ax.margins(0.2)
    #plt.xticks(dataset_indexes, dataset_names)

    #ax = fig.add_subplot(211)

    ax2.plot(dataset_indexes,forest_scores_rms, "ob", label = "Random Forest")
    ax2.plot( dataset_indexes,linear_scores_rms, "or", label = "Linear Regression")
    ax2.plot( dataset_indexes, gradient_scores_rms, "og", label = "Gradient Boosting")
    ax2.set_xticks(dataset_indexes)
    ax2.set_xticklabels(dataset_names, rotation='vertical')
    ax2.set_title("RMSE")
    ax2.margins(0.2)

    #plt.ylim(bottom = 0)
    #plt.xticks(dataset_indexes,dataset_names,rotation='vertical')
    #plt.legend(loc="upper right")
    #plt.margins(0.2)
    #plt.subplots_adjust(bottom=0.5)

    plt.legend(loc="upper right")
    #plt.margins(0.2)
    plt.subplots_adjust(bottom=0.5)
    plt.show()


def main():
    df = load_files()
    sum_random_forest = 0
    sum_linear = 0
    sum_gradient_boosting = 0
    for i in range(NUM_OF_ITERATIONS):
        progress = f"Progress: {i}/{NUM_OF_ITERATIONS}"
        sys.stdout.write('\r' + progress)
        sys.stdout.flush()
        results = run_algorithms(df[i])
        sum_random_forest += results[0]
        sum_linear += results[1]
        sum_gradient_boosting += results[2]
    sys.stdout.write('\r' + f'Progress: {NUM_OF_ITERATIONS}/{NUM_OF_ITERATIONS}\n')
    average_random_forest = sum_random_forest / NUM_OF_ITERATIONS
    average_linear = sum_linear / NUM_OF_ITERATIONS
    average_gradient_boosting = sum_gradient_boosting / NUM_OF_ITERATIONS
    print(f'Average of scores from scikit-learn for Random Forest = {average_random_forest}')
    print(f'Average of scores from scikit-learn for Linear Regression = {average_linear}')
    print(f'Average of scores from scikit-learn for Gradient Boosting = {average_gradient_boosting}')
    plotScores()


if __name__ == '__main__':
    main()


