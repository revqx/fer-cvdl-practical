import glob
import os

import pandas as pd
import torch
import torchvision.transforms.v2
from tqdm import tqdm

from model import get_model
from utils import LABEL_TO_STR, load_images


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
    return use_model(model_id, model, data_path, preprocessing=preprocessing)


def use_model(model_id, model: torch.nn.Module, data_path: str, preprocessing=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    columns = ['file'] + [LABEL_TO_STR[i] for i in range(len(LABEL_TO_STR))]
    results_df = pd.DataFrame(columns=columns)

    images = load_images(data_path)
    for file_name, image in tqdm(images, desc=f"Inference of {data_path}"):
        image = image.to(device)
        if preprocessing:
            image = preprocessing(image)
        image = image.unsqueeze(0)
        output = model(image).tolist()[0]
        results_df.loc[len(results_df)] = [file_name] + output

    return model_id, results_df


def load_model_and_preprocessing(model_name: str) -> (str, torch.nn.Module, torchvision.transforms.v2.Compose):
    """Chooses most recent model with name or choose model by id."""

    # Find all files in the specified directory that match the model class name
    model_files = get_available_models()
    if not model_files:
        raise ValueError(f"No models found in path.")

    possible = []
    selected_model_path = None

    for model_path in model_files:
        model_time_id = os.path.basename(model_path)
        if model_time_id.startswith(model_name):
            possible.append(model_path)
        if model_time_id.endswith(f"{model_name}.pth"):
            possible.append(model_path)
            selected_model_path = model_path

    if not possible:
        raise ValueError(f"No model found with identifier {model_name}")

    # found exact matching id
    if not selected_model_path:
        # Extract timestamps from the model filenames
        timestamps = [os.path.basename(file).split('-')[-2] for file in possible]

        # Find the index of the most recent timestamp
        latest_index = timestamps.index(max(timestamps))

        # Load the most recent model
        selected_model_path = possible[latest_index]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded_model_dict = torch.load(selected_model_path, map_location=device)

    # Reconstruct the model
    model_type = os.path.basename(selected_model_path).split('-')[0]
    model_id = os.path.basename(selected_model_path).split('-')[-1][:-4]
    loaded_model = get_model(model_type)
    loaded_model.load_state_dict(loaded_model_dict['model'])
    loaded_preprocessing = loaded_model_dict['preprocessing']

    print(f"Loaded model from {selected_model_path}")
    return model_id, loaded_model, loaded_preprocessing


def get_available_models():
    path = os.getenv('MODEL_SAVE_PATH')
    if not os.path.exists(path):
        raise EnvironmentError('Path for saved models not specified!')
    # all files in folder at path
    pattern = os.path.join(path, f"*-*-*.pth")
    model_files = glob.glob(pattern)
    return model_files
