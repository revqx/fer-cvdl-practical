import os
from functools import lru_cache

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import LABEL_TO_NUM, load_images, label_from_path, transform


def get_dataset(name: str):
    """Return the dataset object given the name."""
    datasets = {
        "AffectNet": AffectNet,
        "RAF-DB": RafDb
    }
    if name in datasets:
        return datasets[name]()

    raise ValueError("Invalid dataset name.")


class AffectNet(Dataset):
    def __init__(self):
        csv_path = os.path.join(os.getenv("DATASET_AFFECT_NET_PATH"), "labels.csv")
        root_dir = os.getenv("DATASET_AFFECT_NET_PATH")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {csv_path} not found. Could not load AffectNet dataset.")
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found. Could not load AffectNet dataset.")

        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_path)
        self.annotations = self.annotations[~self.annotations["label"].isin(["neutral", "contempt"])]
        self.annotations.reset_index(drop=True, inplace=True)
        self.annotations["label"] = self.annotations["label"].apply(lambda x: LABEL_TO_NUM[x])

    def __getitem__(self, item):
        # needed for sweep to work
        if item >= len(self.annotations):
            raise IndexError(f"Index {item} out of range")

        img_path = os.path.join(self.root_dir, str(self.annotations.loc[item, "pth"]))
        label = torch.tensor(int(self.annotations.loc[item, "label"]))

        with Image.open(img_path) as img:
            img = transform(img)

        return img, label, img_path

    def __len__(self):
        return len(self.annotations)


class RafDb(Dataset):
    def __init__(self):
        root_dir = os.getenv("DATASET_RAF_DB_PATH")
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found. Could not load RAF-DB dataset.")

        path_tensor_pairs = load_images([root_dir])
        self.labels = [label_from_path(path) for path, _ in path_tensor_pairs]
        self.img_paths = [path for path, _ in path_tensor_pairs]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # needed for sweep to work
        if idx >= len(self.img_paths):
            raise IndexError(f"Index {idx} out of range")

        img_path = self.img_paths[idx]
        label = torch.tensor(int(self.labels[idx]))
        with Image.open(img_path) as img:
            img = transform(img)

        return img, label, img_path


class DatasetWrapper(Dataset):
    def __init__(self, images, labels, img_paths, preprocessing=None, augmentations=[]):
        self.images = images
        self.labels = labels
        self.img_paths = img_paths
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Adjusting the length based on the presence or absence of augmentations
        self.augmentation_factor = 1 if augmentations is None else (1 + len(self.augmentations))

    def __len__(self):
        return len(self.images) * self.augmentation_factor

    @lru_cache(maxsize=100_000)
    def __getitem__(self, idx):
        original_idx = idx // self.augmentation_factor

        if original_idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range")

        image = self.images[original_idx]
        label = self.labels[original_idx]
        img_path = self.img_paths[original_idx]

        if self.preprocessing:
            image = self.preprocessing(image)

        # Apply augmentation if it's not the original image and augmentations are provided
        if idx != original_idx and self.augmentations:
            image = self.augmentations[(idx % self.augmentation_factor) - 1](image)

        image = image.to(self.device)

        return image, label, img_path
