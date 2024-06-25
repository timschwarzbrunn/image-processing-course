from argparse import ArgumentParser
import json
import math
import matplotlib
from matplotlib import pyplot as plt
import os
import pandas as pd
from keras._tf_keras.keras.datasets import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# Ignore some warnings about the convergence.
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

# https://github.com/jktr/matplotlib-backend-kitty
try:
    matplotlib.use("module://matplotlib-backend-kitty")
except ImportError:
    pass


FILEPATH_RESULTS = "benchmark_results.json"


def task_4_2(recalculate_benchmark=False):
    """
    This function calculates or loads the benchmark results and visualizes them.
    """
    if os.path.isfile(FILEPATH_RESULTS) and recalculate_benchmark == False:
        print(f"Loading results ({FILEPATH_RESULTS}).")
        with open(FILEPATH_RESULTS, "r") as file:
            results = json.load(file)
    else:
        print("Calculate benchmark results ...")
        results = benchmark()
    # Reshape the results into a dictionary. We could not save it that way because json.dumps cannot handle tuples as keys.
    benchmark_results = dict()
    for result in results:
        benchmark_results[
            (
                result["max_iteration"],
                result["learning_rate"],
                result["number_of_layers"],
                result["solver"],
            )
        ] = {
            "f1": result["f1"],
            "accuracy": result["accuracy"],
            "recall": result["recall"],
            "precision": result["precision"],
        }
    df = pd.DataFrame(
        benchmark_results.values(),
        index=pd.MultiIndex.from_tuples(
            benchmark_results.keys(),
            names=["max_iteration", "learning_rate", "number_of_layers", "solver"],
        ),
    )
    df.sort_index()
    print(df)
    analyze_benchmark_results(df)


@ignore_warnings(category=ConvergenceWarning)
def benchmark():
    """
    This function executes a benchmark on the MNIST dataset with varying parameters.
    The returned result containes the parameters as well as the metrics.
    """
    # Import the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    # Reshape.
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    # Parameters.
    max_iterations = [5, 10, 15, 20]
    learning_rates = [0.1, 0.01, 0.005]
    hidden_layer_size = 20
    numbers_of_layers = [1, 2, 3, 4]
    solvers = ["adam", "sgd", "lbfgs"]
    n_max = (
        len(max_iterations)
        * len(learning_rates)
        * len(numbers_of_layers)
        * len(solvers)
    )

    # Results are saved in a dictionary and contain the parameters for the network
    # as well as the metrics.
    results = []
    n_current = 0
    for max_iteration in max_iterations:
        for learning_rate in learning_rates:
            for number_of_layers in numbers_of_layers:
                for solver in solvers:
                    n_current += 1
                    print(
                        f"[{n_current} / {n_max}] {max_iteration} iterations, learning rate of {learning_rate}, {number_of_layers} layers and solver '{solver}'.  ",
                        end="\r",
                    )
                    mlp_classifier = MLPClassifier(
                        max_iter=max_iteration,
                        learning_rate_init=learning_rate,
                        hidden_layer_sizes=[
                            hidden_layer_size for _ in range(number_of_layers)
                        ],
                        solver=solver,
                    )
                    mlp_classifier.fit(x_train_flat, y_train)
                    predict = mlp_classifier.predict(x_test_flat)
                    results.append(
                        {
                            "max_iteration": max_iteration,
                            "learning_rate": learning_rate,
                            "number_of_layers": number_of_layers,
                            "solver": solver,
                            "f1": f1_score(y_test, predict, average="macro"),
                            "accuracy": accuracy_score(y_test, predict),
                            "recall": recall_score(y_test, predict, average="macro"),
                            "precision": precision_score(
                                y_test, predict, average="macro", zero_division=0.0
                            ),
                        }
                    )
    with open(FILEPATH_RESULTS, "w") as file:
        json.dump(results, file)
        print(f"Results written to '{FILEPATH_RESULTS}'.")
    return results


def analyze_benchmark_results(df):
    """
    This function analyzes the effect of changing different parameters on the metrics.
    f1 = harmonic mean of precision and recall
    f1 = 2 / (1 / precision + 1 / recall) = (2 * precision * recall) / (precision + recall) = (2 * TP) / (2 * TP + FP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    """
    fig, axs = plt.subplots(math.ceil(len(df.index.names) / 2), 2)
    axs = axs.flatten()
    for idx, index_name in enumerate(df.index.names):
        df_inspection = df.groupby(level=index_name).mean()
        print(df_inspection)
        for col_name, col in df_inspection.items():
            axs[idx].plot(col, label=col_name)
        axs[idx].set_title(index_name)
        axs[idx].legend()
        axs[idx].grid(alpha=0.25)

    if len(df.index.names) % 2 == 1:
        fig.delaxes(axs[-1])
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--recalculate-benchmark",
        dest="recalculate_benchmark",
        help="recalculate the benchmark instead of loading it",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    task_4_2(args.recalculate_benchmark)
