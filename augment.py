import os
import random
import torch
from torchvision import transforms
from utils import load_images


def horizontal_flip(tensor):
    # Assuming tensor is in CxHxW format
    return torch.flip(tensor, [2])


def random_rotation(tensor):
    angle = random.uniform(-10, 10)
    return transforms.functional.rotate(tensor, angle)


def tensor_to_image(tensor):
    # Convert tensor to PIL Image
    return transforms.ToPILImage()(tensor)


def random_scale(tensor):
    scale = random.uniform(1.0, 1.3)
    return transforms.functional.affine(tensor, angle=0, translate=[0, 0], scale=scale, shear=0)


def small_gaussian_blur(tensor):
    return transforms.functional.gaussian_blur(tensor, kernel_size=5, sigma=1.0)


def augment_images(input_dir, output_dir, augmentations):
    images = load_images([input_dir])

    for img_path, tensor in images:
        filename = os.path.basename(img_path)
        filename_without_extension, extension = os.path.splitext(filename)

        for i, augmentation in enumerate(augmentations):
            augmented_tensor = augmentation(tensor)
            augmented_image = tensor_to_image(augmented_tensor)
            augmented_image.save(os.path.join(output_dir, f"{filename_without_extension}_aug{i}.jpg"))


if __name__ == "__main__":
    input_dir = "/Users/marius/github/fer-cvdl-practical/data/augmentations"
    output_dir = "/Users/marius/github/fer-cvdl-practical/data/augmentations"
    augmentations = [horizontal_flip, random_rotation, random_scale]

    augment_images(input_dir, output_dir, augmentations)
