import os

import pandas as pd

from inference import apply_model


def get_model_results(model_ids, data_path=os.getenv("DATASET_TEST_PATH")):
    results = []
    paths = []
    columns = None
    for model_id in model_ids:
        _, result_df = apply_model(model_id, data_path)
        results.append(result_df.drop(columns=['file']).values.tolist())
        paths = result_df['file'].tolist()
        if columns is None:
            columns = result_df.columns.drop('file')

    # Inner lists contain the results for one image over all models
    results = list(map(list, zip(*results)))

    return results, paths, columns


def ensemble_results(model_ids, data_path=os.getenv("DATASET_TEST_PATH")):
    results, paths, columns = get_model_results(model_ids, data_path)

    # We have no softmax layer in our models, so we have to use averaging method to create the ensemble
    averaged_results = [[sum(values) / len(values) for values in zip(*row)] for row in results]

    # Accuracies and confusionmatrix expect a dataframe
    ensemble_results_df = pd.DataFrame(averaged_results, columns=columns)
    ensemble_results_df.insert(0, 'path', paths)

    return ensemble_results_df
