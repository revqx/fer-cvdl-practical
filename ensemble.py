import os
import pandas as pd

from inference import apply_model
from utils import LABEL_TO_STR, load_images, label_from_path
from analyze import accuracies, confusion_matrix


def get_model_results(model_ids, data_path=os.getenv("DATASET_VALIDATION_PATH")):
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


def ensemble_results(model_ids, data_path=os.getenv("DATASET_VALIDATION_PATH")):
    results, paths, columns = get_model_results(model_ids, data_path)

    # We have no softmax layer in our models, so we have to use Avaraging method to create the ensemble
    averaged_results = [[sum(values) / len(values) for values in zip(*row)] for row in results]

    # Accuracies and confusionmatrix expect a dataframe
    ensemble_results_df = pd.DataFrame(averaged_results, columns=columns)
    ensemble_results_df.insert(0, 'path', paths)

    return ensemble_results_df



#### Legacy code #####

def ensemble_old(model1: str, model2: str, model3:str, data_path=os.getenv("DATASET_VALIDATION_PATH")):
    results = []
    paths = []
    model_ids = [model1, model2, model3]
    for model_id in model_ids:
        _, result_df = apply_model(model_id, data_path)
        results.append(result_df.drop(columns=['file']).values.tolist())
        paths = result_df['file'].tolist()

    # Transpose the results so that each inner list contains the results for one image from all models
    results = list(map(list, zip(*results)))

    # We have no softmax layer in our models, so we have to use Avaraging method to create the ensemble
    averaged_results = [[sum(values) / len(values) for values in zip(*row)] for row in results]

    # Accuracies and confusionmatrix expect a dataframe
    ensemble_results_df = pd.DataFrame(averaged_results, columns=result_df.columns.drop('file'))
    ensemble_results_df.insert(0, 'path', paths)

    top_n = accuracies(ensemble_results_df, best=3)
    conf_matrix = confusion_matrix(ensemble_results_df)
    print(conf_matrix)
    print(top_n)