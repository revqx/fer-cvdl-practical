import os

import numpy as np
import pandas as pd
import wandb

from inference import apply_model
from utils import LABEL_TO_STR, label_from_path


def accuracies(inference_results: pd.DataFrame, best=3) -> dict:
    """Return top1 and top3 accuracies."""
    top_n = [0] * best

    for path, *values in inference_results.values:
        label = label_from_path(path)
        if label is None:
            raise ValueError(f"Could not find label in path {path}.")
        predictions = np.array(values).argsort()[::-1]
        for i in range(len(top_n)):
            if label in predictions[:i + 1]:
                top_n[i] += 1
    return {f'top{i + 1}_acc': top_n[i] / len(inference_results) for i in range(len(top_n))}


def confusion_matrix(inference_result: pd.DataFrame) -> pd.DataFrame:
    """Return the confusion matrix.
       Rows are the true labels, columns are the predictions."""
    matrix = np.zeros((len(LABEL_TO_STR), len(LABEL_TO_STR)))
    for path, *values in inference_result.values:
        label = label_from_path(path)
        if label is None:
            raise ValueError(f"Could not find label in path {path}.")
        predictions = np.array(values).argsort()[::-1]
        matrix[label, predictions[0]] += 1
    return pd.DataFrame(matrix, index=LABEL_TO_STR.values(), columns=LABEL_TO_STR.values())


def predictions_and_true_labels(inference_result: pd.DataFrame) -> (list, list):
    """Return the predictions and true labels."""
    predictions = []
    true_labels = []
    for path, *values in inference_result.values:
        label = label_from_path(path)
        if label is None:
            raise ValueError(f"Could not find label in path {path}.")
        predictions.append(np.array(values).argsort()[::-1][0])
        true_labels.append(label)
    return predictions, true_labels


def analyze_run_and_upload(model_name: str):
    validation_data_path = os.getenv("DATASET_VALIDATION_PATH")
    model_id, results = apply_model(model_name, validation_data_path)
    top_n = accuracies(results, best=3)
    wandb.log(top_n)
    preds, y_true = predictions_and_true_labels(results)
    # upload confusion matrix to wandb
    conf_matrix_plot = wandb.plot.confusion_matrix(y_true=y_true,
                                                   preds=preds,
                                                   class_names=list(LABEL_TO_STR.values()),
                                                   title="Confusion Matrix")
    wandb.log({"confusion_matrix": conf_matrix_plot})
    print(f"Finished analysis of model {model_id}.")
