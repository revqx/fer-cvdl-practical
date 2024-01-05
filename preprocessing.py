import torch
from torchvision.transforms import v2


def select_transform(transforms_str):
    """Select the appropriate transforms for the dataset."""
    if not transforms_str:
        return None

    # Split the preprocessing string into individual transforms and strip whitespace
    transforms_strings = transforms_str.split(", ")

    # Create a list of transforms
    transform_list = []

    transforms_available = {
        "Resize()": v2.Resize((64, 64)),
        "ToTensor()": v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        "StandardizeGray()": v2.Normalize((0.5,), (0.5,)),
        "StandardizeRGB()": v2.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
    }

    # Loop over the transforms and add them to the list
    for transform in transforms_strings:
        if transform not in transforms_available:
            raise ValueError(f"Unsupported transform: {transform}")
        transform_list.append(transforms_available[transform])

    # Create a torchvision.transforms.Compose object from the list of transforms
    return v2.Compose(transform_list)
