import pandas as pd
import numpy as np

from utils import LABEL_TO_NUM, LABEL_TO_STR


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
    return {f'top{i + 1}': top_n[i] / len(inference_results) for i in range(len(top_n))}


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


def label_from_path(path) -> int | None:
    """Try to guess the label from a given path.
       Returns None if no label is found."""
    for label, num in LABEL_TO_NUM.items():
        if label in path:
            return num
    return None
