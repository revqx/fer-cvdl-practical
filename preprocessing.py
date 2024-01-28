import torchvision.transforms as v2
from typing import Optional

from utils import AVAILABLE_PREPROCESSINGS


def select_preprocessing(preprocessing_str: str) -> Optional[v2.Compose]:
    """Select the appropriate preprocessings for the dataset.

    Parameters:
    preprocessing_str (str): A comma-separated string of preprocessing steps.

    Returns:
    torchvision.transforms.Compose: A Compose object with the specified preprocessing steps.

    Raises:
    ValueError: If an unsupported transform is specified.
    """
    if not preprocessing_str:
        return v2.Compose([])  # Return an identity transform or default preprocessing

    preprocessing_strings = [t.strip() for t in preprocessing_str.split(",")]
    preprocessing_list = []

    for preprocessing in preprocessing_strings:
        if preprocessing not in AVAILABLE_PREPROCESSINGS:
            supported = ", ".join(AVAILABLE_PREPROCESSINGS.keys())
            raise ValueError(f"Unsupported transform: {preprocessing}. Supported transforms are: {supported}")
        preprocessing_list.append(AVAILABLE_PREPROCESSINGS[preprocessing]())

    return v2.Compose(preprocessing_list)
