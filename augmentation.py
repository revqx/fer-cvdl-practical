import os

from torchvision.transforms import v2

from utils import load_images

AVAILABLE_AUGMENTATIONS = {
    "HorizontalFlip": v2.RandomHorizontalFlip(p=1),  # Always apply horizontal flip
    "RandomRotation": v2.RandomRotation(degrees=(-10, 10)),
    "RandomCrop": v2.RandomAffine(degrees=0, translate=(0, 0), scale=(1.0, 1.3), shear=0),
    "TrivialAugmentWide": v2.TrivialAugmentWide(),
    "RandAugment": v2.RandAugment()
}


def select_augmentations(augmentations_str: str) -> list[v2.Compose]:
    if not augmentations_str:
        return []  # Return an empty Compose object

    augmentation_strings = [aug.strip() for aug in augmentations_str.split(",")]
    augmentation_list = []

    for aug in augmentation_strings:
        if aug not in AVAILABLE_AUGMENTATIONS:
            supported = ", ".join(AVAILABLE_AUGMENTATIONS.keys())
            raise ValueError(f"Unsupported augmentation: {aug}. Supported augmentations are: {supported}")
        augmentation_list.append(AVAILABLE_AUGMENTATIONS[aug])

    return augmentation_list


def tensor_to_image(tensor):
    return v2.ToPILImage()(tensor)


def augment_images(input_dir, output_dir, augmentations):
    images = load_images([input_dir])

    for img_path, tensor in images:
        filename = os.path.basename(img_path)
        filename_without_extension, extension = os.path.splitext(filename)

        for i, augmentation in enumerate(augmentations):
            augmented_tensor = augmentation(tensor)
            augmented_image = tensor_to_image(augmented_tensor)
            augmented_image.save(os.path.join(output_dir, f"{filename_without_extension}_aug{i}.jpg"))
