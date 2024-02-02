import pandas as pd
import torch
from tqdm import tqdm

from utils import LABEL_TO_STR, load_images, load_model_and_preprocessing


def apply_model(model_name: str, data_path: str):
    """Uses the model to infer on all images in the data_path.
    Args:
        model_name: Most recent with model_name or WandB id of the model.
        data_path: Path to the folder containing the images.
                    Each image path should contain the label.
    Returns:
        model_id: The id of the model.
        results_df: A dataframe containing the results.
    """

    model_id, model, preprocessing = load_model_and_preprocessing(model_name)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    columns = ["file"] + [LABEL_TO_STR[i] for i in range(len(LABEL_TO_STR))]
    results_df = pd.DataFrame(columns=columns)

    path_tensor_pairs = load_images([data_path])
    for path, image in tqdm(path_tensor_pairs, desc=f"Inference of {data_path}"):
        image = image.to(device)
        if preprocessing:
            image = preprocessing(image)
        image = image.unsqueeze(0)
        output = model(image).tolist()[0]
        results_df.loc[len(results_df)] = [path] + output

    return model_id, results_df
