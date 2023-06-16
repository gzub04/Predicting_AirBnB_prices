import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

NUM_OF_ITERATIONS = 50  # how many times should each ML algorithm be run

# random forest parameters
MAX_DEPTH = None
MIN_IMPURITY_DECREASE = 0.05

# gradient boosting parameters
N_ESTIMATORS_GRADIENT = 100
MAX_DEPTH_GRADIENT = 5
MIN_SAMPLES_SPLIT_GRADIENT = 10
LEARNING_RATE_GRADIENT = 0.1
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


def rmse(regressor, x_test, y_test):
    # Returns root squared mean error
    y_pred = regressor.predict(x_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


def mape(regressor, x_test, y_test):
    # Returns mean absolute percentage error
    y_pred = regressor.predict(x_test)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


def load_files():
    file_list = glob.glob("Datasets/*.csv")
    dataframes = []
    for file in file_list:
        df = pd.read_csv(file)
        df.name = os.path.basename(file).split(".")[0]
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

    return [random_forest_regression.score(x_test, y_test), rmse(random_forest_regression, x_test, y_test),
            mape(random_forest_regression, x_test, y_test),
            linear_regression.score(x_test, y_test), rmse(linear_regression, x_test, y_test),
            mape(linear_regression, x_test, y_test),
            gradient_boosting_regressor.score(x_test, y_test), rmse(gradient_boosting_regressor, x_test, y_test),
            mape(gradient_boosting_regressor, x_test, y_test)]


def plot_scores(dataset_names, forest_r_squared, forest_scores_rms, forest_mape,
                linear_r_squared, linear_scores_rms, linear_mape,
                gradient_r_squared, gradient_scores_rms, gradient_mape):

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 8))
    dataset_indexes = np.arange(len(dataset_names))

    ax.plot(dataset_indexes, forest_r_squared, "ob", label="Random Forest")
    ax.plot(dataset_indexes, linear_r_squared, "or", label="Linear Regression")
    ax.plot(dataset_indexes, gradient_r_squared, "og", label="Gradient Boosting")
    ax.set_xticks(dataset_indexes)
    ax.set_xticklabels(dataset_names, rotation='vertical', fontsize=8)
    ax.set_title("R^2 score")
    ax.set_ylim(top=1.0, bottom=0)
    ax.margins(0.2)
    ax.grid()

    ax2.plot(dataset_indexes, forest_scores_rms, "ob", label="Random Forest")
    ax2.plot(dataset_indexes, linear_scores_rms, "or", label="Linear Regression")
    ax2.plot(dataset_indexes, gradient_scores_rms, "og", label="Gradient Boosting")
    ax2.set_xticks(dataset_indexes)
    ax2.set_xticklabels(dataset_names, rotation='vertical', fontsize=8)
    ax2.set_title("RMSE")
    ax2.margins(0.2)
    ax2.grid()

    ax3.plot(dataset_indexes, forest_mape, "ob", label="Random Forest")
    ax3.plot(dataset_indexes, linear_mape, "or", label="Linear Regression")
    ax3.plot(dataset_indexes, gradient_mape, "og", label="Gradient Boosting")
    ax3.set_xticks(dataset_indexes)
    ax3.set_xticklabels(dataset_names, rotation='vertical', fontsize=8)
    ax3.set_title("MAPE")
    ax3.margins(0.2)
    ax3.grid()

    ax.set_xlim(0, len(dataset_indexes)-1)
    ax2.set_xlim(0, len(dataset_indexes)-1)
    ax3.set_xlim(0, len(dataset_indexes)-1)

    fig.legend(labels=['Random Forest', 'Linear Regression', 'Gradient Boosting'], loc='upper center', ncol=3)

    plt.subplots_adjust(bottom=0.5)

    plt.show()


def print_results(scores):
    """
    Takes in a list of scores and prints out its average and standard variation, best and worst score
    :param scores: List of metric scores
    :return: None
    """
    print(f'Average: {np.mean(scores)}')
    print(f'Standard variation: {np.std(scores)}')
    print(f'Highest and lowest score: {max(scores), min(scores)}')


def main():
    dfs = load_files()

    algorithm_names = ['Random forest', 'Linear', 'Gradient boosting']
    metric_names = ['r_squared', 'rmse', 'mape']
    general_scores = {
        'Random forest': {'r_squared': [], 'rmse': [], 'mape': []},
        'Linear': {'r_squared': [], 'rmse': [], 'mape': []},
        'Gradient boosting': {'r_squared': [], 'rmse': [], 'mape': []}
    }
    dataset_names = []

    for i, dataset in enumerate(dfs):
        dataset_names.append(dataset.name)
        scores_for_one_dataset = {
            'Random forest': {'r_squared': [], 'rmse': [], 'mape': []},
            'Linear': {'r_squared': [], 'rmse': [], 'mape': []},
            'Gradient boosting': {'r_squared': [], 'rmse': [], 'mape': []}
        }

        for j in range(NUM_OF_ITERATIONS):
            progress = f"Analysing {dataset.name} ({i+1}/{len(dfs)}): Progress: {j+1}/{NUM_OF_ITERATIONS}"
            sys.stdout.write('\r' + progress)
            sys.stdout.flush()

            results = run_algorithms(dataset)
            scores_for_one_dataset['Random forest']['r_squared'].append(results[0])
            scores_for_one_dataset['Random forest']['rmse'].append(results[1])
            scores_for_one_dataset['Random forest']['mape'].append(results[2])

            scores_for_one_dataset['Linear']['r_squared'].append(results[3])
            scores_for_one_dataset['Linear']['rmse'].append(results[4])
            scores_for_one_dataset['Linear']['mape'].append(results[5])

            scores_for_one_dataset['Gradient boosting']['r_squared'].append(results[6])
            scores_for_one_dataset['Gradient boosting']['rmse'].append(results[7])
            scores_for_one_dataset['Gradient boosting']['mape'].append(results[8])

        for algorithm_name in algorithm_names:
            for metric_name in metric_names:
                general_scores[algorithm_name][metric_name].append(
                    np.mean(scores_for_one_dataset[algorithm_name][metric_name]))

    sys.stdout.write('\r')
    for metric_name in metric_names:
        print(f'{metric_name}:')
        for algorithm_name in algorithm_names:
            print(f'{algorithm_name}:')
            print_results(general_scores[algorithm_name][metric_name])

    plot_scores(dataset_names, general_scores['Random forest']['r_squared'], general_scores['Random forest']['rmse'],
                general_scores['Random forest']['mape'],
                general_scores['Linear']['r_squared'], general_scores['Linear']['rmse'],
                general_scores['Linear']['mape'],
                general_scores['Gradient boosting']['r_squared'], general_scores['Gradient boosting']['rmse'],
                general_scores['Gradient boosting']['mape'])


if __name__ == '__main__':
    main()
