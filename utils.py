import glob
import os

import torch
from PIL import Image
from torchvision.transforms import v2

LABEL_TO_NUM = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'happiness': 3,
    'sad': 4,
    'sadness': 4,
    'surprise': 5
}

LABEL_TO_STR = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise'
}

# Default transformation for images
transform = v2.Compose([
    v2.Resize((64, 64), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])


# load images from list of paths
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


def label_from_path(path) -> int | None:
    """Try to guess the label from a given path.
       Returns None if no label is found."""
    for label, num in LABEL_TO_NUM.items():
        if label in path:
            return num
    return None
