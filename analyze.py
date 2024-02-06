import os

import numpy as np
import pandas as pd
import wandb

from inference import apply_model
from utils import LABEL_TO_STR, label_from_path


def accuracies(inference_results: pd.DataFrame) -> dict:
    """Return top1, top3 and avg accuracies."""
    top1 = 0
    top3 = 0
    label_counts = [0, 0, 0, 0, 0, 0]
    correct_counts = [0, 0, 0, 0, 0, 0]

    for path, *values in inference_results.values:
        label = label_from_path(path)
        if label is None:
            raise ValueError(f"Could not find label in path {path}.")
        predictions = np.array(values).argsort()[::-1]
        label_counts[label] += 1
        if label in predictions[:3]:
            top3 += 1
        if label == predictions[0]:
            top1 += 1
            correct_counts[label] += 1

    top1_acc = top1 / len(inference_results) if len(inference_results) > 0 else 0
    top3_acc = top3 / len(inference_results) if len(inference_results) > 0 else 0
    avg = [correct_counts[i] / label_counts[i] if label_counts[i] > 0 else 0 for i in range(6)]
    avg_acc = np.mean(avg)

    return {"top1_acc": top1_acc, "top3_acc": top3_acc, "avg_acc": avg_acc}


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

    # Normalize
    matrix = matrix / np.sum(matrix, axis=1, keepdims=True)

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
    validation_data_path = os.getenv("DATASET_TEST_PATH")
    model_id, results = apply_model(model_name, validation_data_path)
    top_n = accuracies(results)
    wandb.log(top_n)

    preds, y_true = predictions_and_true_labels(results)
    # upload confusion matrix to wandb
    conf_matrix_plot = wandb.plot.confusion_matrix(y_true=y_true,
                                                   preds=preds,
                                                   class_names=list(LABEL_TO_STR.values()),
                                                   title="Confusion Matrix")

    wandb.log({"confusion_matrix": conf_matrix_plot})
    print(f"Finished analysis of model {model_id}.")
