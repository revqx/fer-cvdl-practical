import glob
import os
import random as rnd

import torch
import torchvision
from PIL import Image
from torchvision.transforms import v2

from model import get_model

LABEL_TO_NUM = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "happiness": 3,
    "sad": 4,
    "sadness": 4,
    "surprise": 5
}

LABEL_TO_STR = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise"
}

# Default transformation for images
transform = v2.Compose([
    v2.Resize((64, 64), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])


def load_images(paths: list[str]) -> list[tuple[str, torch.Tensor]]:
    """
    Load all images from the given paths.

    Args:
        paths: A list of paths to the images.

    Returns:
        A list of tuples containing the image paths and image tensors.
    """

    img_paths = []
    path_tensor_pairs = []

    # collect all img_paths from the given directories
    for path in paths:
        img_paths += glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png"))

    for img_path in img_paths:
        with Image.open(img_path) as image:
            path_tensor_pairs.append((img_path, transform(image)))

    return path_tensor_pairs


def label_from_path(path: str):
    """Try to guess the label from a given path.
       Returns None if no label is found."""
    for label, num in LABEL_TO_NUM.items():
        if label in path:
            return num
    return None


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
        timestamps = [os.path.basename(file).split("-")[-2] for file in possible]

        # Find the index of the most recent timestamp
        latest_idx = timestamps.index(max(timestamps))

        # Load the most recent model
        selected_model_path = possible[latest_idx]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded_model_dict = torch.load(selected_model_path, map_location=device)

    # Reconstruct the model
    model_name = os.path.basename(selected_model_path).split("-")[0]
    model_id = os.path.basename(selected_model_path).split("-")[-1][:-4]
    loaded_model = get_model(model_name)
    loaded_model.load_state_dict(loaded_model_dict["model"])
    loaded_preprocessing = loaded_model_dict["preprocessing"]

    # count number of parameter in flattened loaded_model_dict['model']
    num_params = sum(p.numel() for p in loaded_model.parameters())

    print(f"Loaded model from {selected_model_path} with {num_params} parameters.")
    return model_id, loaded_model, loaded_preprocessing


def get_available_models():
    path = os.getenv("MODEL_SAVE_PATH")
    if path is None or not os.path.exists(path):
        raise EnvironmentError("Path for saved models not specified! Make sure to copy the .env.dist file to a local .env file.")
    # all files in folder at path
    pattern = os.path.join(path, f"*-*-*.pth")
    model_files = glob.glob(pattern)
    return model_files


def get_images_and_labels(path: str, limit=None, random=False, path_contains=None) -> ([torch.Tensor], [int]):
    """Load all images and labels from the given path.
       Returns a tuple of lists containing the images and labels."""
    images = load_images([path])
    labels = [label_from_path(path) for path, _ in images]
    paths = [path for path, _ in images]
    if path_contains:
        images, labels, paths = zip(*[(i, l, p) for i, l, p in zip(images, labels, paths) if path_contains in p])
    limit = min(limit, len(images))
    if limit and random:
        images, labels, paths = zip(*rnd.sample(list(zip(images, labels, paths)), limit))
    if limit:
        images = images[:limit]
        labels = labels[:limit]
        paths = paths[:limit]
    return [tensor for _, tensor in images], labels, paths


def get_available_models_by_type():
    model_files = get_available_models()
    models = {name: [
        os.path.basename(file).split("-")[-1][:-4] for file in model_files if name in os.path.basename(file)
    ] for name in set([os.path.basename(file).split("-")[0] for file in model_files])}
    return models
