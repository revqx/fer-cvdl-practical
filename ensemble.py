import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from inference import apply_model


def get_model_results(model_ids, data_path=os.getenv("DATASET_TEST_PATH")):
    results = []
    paths = []
    columns = None
    for model_id in model_ids:
        _, result_df = apply_model(model_id, data_path)
        results.append(result_df.drop(columns=["file"]).values.tolist())
        paths = result_df["file"].tolist()
        if columns is None:
            columns = result_df.columns.drop("file")

    # Inner lists contain the results for one image over all models
    results = list(map(list, zip(*results)))

    return results, paths, columns


def ensemble_results(model_ids, data_path=os.getenv("DATASET_TEST_PATH")):
    results, paths, columns = get_model_results(model_ids, data_path)

    # We have no softmax layer in our models, so we have to use averaging method to create the ensemble
    averaged_results = [[sum(values) / len(values) for values in zip(*row)] for row in results]

    # Accuracies and confusionmatrix expect a dataframe
    ensemble_results_df = pd.DataFrame(averaged_results, columns=columns)
    ensemble_results_df.insert(0, "path", paths)

    return ensemble_results_df


def save_confusion_matrix_as_heatmap(conf_matrix, labels, filename):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, ax=ax)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    fig.suptitle("Top-1: 82,16% & Top-3: 95.33%", y=0.1, weight='bold')

    plt.savefig(filename)
