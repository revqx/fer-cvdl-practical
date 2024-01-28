import torch
from torchvision.transforms import v2


def select_preprocessing(transforms_str):
    # TODO: delete current prepr and add Standardization and grayscale
    """Select the appropriate transforms for the dataset."""
    if not transforms_str:
        return None

    # Split the preprocessing string into individual transforms and strip whitespace
    transforms_strings = [t.strip() for t in transforms_str.split(",")]

    # Create a list of transforms
    transform_list = []

    transforms_available = {
        "TrivialAugmentWide()": v2.TrivialAugmentWide(),
        "RandAugment()": v2.RandAugment(),
        "AutoAugment()": v2.AutoAugment(),
        "RandomHorizontalFlip()": v2.RandomHorizontalFlip()
    }

    # Loop over the transforms and add them to the list
    for transform in transforms_strings:
        if transform not in transforms_available:
            raise ValueError(f"Unsupported transform: {transform}")
        transform_list.append(transforms_available[transform])

    # Create a torchvision.transforms.Compose object from the list of transforms
    return v2.Compose(transform_list)
