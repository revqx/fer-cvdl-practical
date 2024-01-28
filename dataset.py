import glob
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from augment import horizontal_flip, random_rotation, random_scale

from utils import LABEL_TO_NUM, load_images, label_from_path, transform

"""
It is important to note that the dataset class is responsible 
for transforming the images to [1,3|1,64,64] tensors with values 
between -1 and 1 depending on the grayscale parameter.
The preprocessing is everything that happens after that.

Every dataset class should only return the labels 0-5.
"""


def get_dataset(name: str):
    """Return the dataset object given the name."""
    datasets = {
        "AffectNet": AffectNet,
        "FER2013": Fer2013,
        "RAF-DB": RafDb
    }
    if name in datasets:
        return datasets[name]()

    raise ValueError("Invalid dataset name.")


class AffectNet(Dataset):
    def __init__(self):
        csv_path = os.path.join(os.getenv('DATASET_AFFECT_NET_PATH'), 'labels.csv')
        root_dir = os.getenv('DATASET_AFFECT_NET_PATH')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {csv_path} not found. Could not load AffectNet dataset.")
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found. Could not load AffectNet dataset.")

        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_path)
        self.annotations = self.annotations[~self.annotations['label'].isin(['neutral', 'contempt'])]
        self.annotations.reset_idx(drop=True, inplace=True)
        self.annotations['label'] = self.annotations['label'].apply(lambda x: LABEL_TO_NUM[x])

    def __getitem__(self, item):
        # needed for sweep to work (don't ask why)
        if item >= len(self.annotations):
            raise IndexError(f"Index {item} out of range")

        img_path = os.path.join(self.root_dir, str(self.annotations.loc[item, 'pth']))
        label = torch.tensor(int(self.annotations.loc[item, 'label']))

        return img_path, label

    def __len__(self):
        return len(self.annotations)


class Fer2013(Dataset):
    def __init__(self):
        raise NotImplemented()

    def __getitem__(self, item):
        raise NotImplemented()

    def __len__(self):
        raise NotImplemented()


class RafDb(Dataset):
    def __init__(self):
        """
        Initialize the RAF_DB dataset.

        Args:
            preprocessing: A torchvision transform that is applied to the images.
            grayscale: If True, the images will be converted to grayscale.

        returns:
            The training data of the RAF-DB dataset since the validation data is already used for testing.
        """

        root_dir = os.getenv('DATASET_RAF_DB_PATH')

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found. Could not load RAF-DB dataset.")

        path_tensor_pairs = load_images([root_dir])
        labels = [label_from_path(path) for path, _ in path_tensor_pairs]

        self.root_dir = root_dir
        self.annotations = pd.DataFrame({'pth': [path for path, _ in path_tensor_pairs], 'label': labels})

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # needed for sweep to work
        if idx >= len(self.annotations):
            raise IndexError(f"Index {idx} out of range")

        img_path = self.annotations.loc[idx, 'pth']
        label = torch.tensor(int(self.annotations.loc[idx, 'label']))

        return img_path, label


class DatasetWrapper(Dataset):
    # TODO: support list of preprocessings and augmentations instead of fixed ones
    def __init__(self, img_paths, labels, preprocessing=None, augment=True):
        self.img_paths = img_paths
        self.labels = labels
        self.augment = augment
        self.preprocessing = preprocessing
        self.annotations = pd.DataFrame({'pth': img_paths, 'label': labels})

    def __len__(self):
        return len(self.img_paths) * (4 if self.augment else 1)

    def __getitem__(self, idx):
        # Determine the original index of the image and the augmentation type
        original_idx = idx // 4 if self.augment else idx
        augmentation_type = idx % 4 if self.augment else 0

        # needed for sweep to work
        if original_idx >= len(self.annotations):
            raise IndexError(f"Index {idx} out of range")

        img_path = self.annotations.loc[original_idx, 'pth']
        image = Image.open(img_path)
        label = self.annotations.loc[original_idx, 'label']

        # Apply transformations
        image = transform(image)

        # TODO: support list of preprocessings and augmentations instead of fixed ones
        # Preprocess the image
        if self.preprocessing:
            image = self.preprocessing(image)

        # Apply augmentations
        if self.augment:
            if augmentation_type == 1:
                image = horizontal_flip(image)
            elif augmentation_type == 2:
                image = random_rotation(image)
            elif augmentation_type == 3:
                image = random_scale(image)

        return image, label





