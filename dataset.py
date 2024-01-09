import os

from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

from utils import LABEL_TO_NUM

"""
It is important to note that the dataset class is responsible 
for transforming the images to [1,3|1,64,64] tensors with values 
between -1 and 1 depending on the black_and_white parameter.
The preprocessing is everything that happens after that.

Every dataset class should only return the labels 0-5.
"""


def get_dataset(name: str, preprocessing=None, black_and_white=False):
    """Return the dataset object given the name."""
    datasets = {
        "AffectNet": AffectNet,
        "FER2013": Fer2013,
        "RAF-DB": RAF_DB
    }
    if name in datasets:
        return datasets[name](preprocessing=preprocessing,
                              black_and_white=black_and_white)

    raise ValueError("Invalid dataset name.")


class AffectNet(Dataset):
    def __init__(self, preprocessing=None, black_and_white=False):
        if black_and_white:
            raise NotImplementedError("Black and white images are not yet supported for the AffectNet dataset.")

        csv_path = os.path.join(os.getenv('DATASET_AFFECT_NET_PATH'), 'labels.csv')
        root_dir = os.getenv('DATASET_AFFECT_NET_PATH')
        self.root_dir = root_dir
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {csv_path} not found. Could not load AffectNet dataset.")
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found. Could not load AffectNet dataset.")

        self.annotations = pd.read_csv(csv_path)
        # remove neutral and contempt images from dataset
        self.annotations = self.annotations[~self.annotations['label'].isin(['neutral', 'contempt'])]
        # reset index
        self.annotations = self.annotations.reset_index(drop=True)
        # Encode the 'label' column using the label_encode dictionary
        self.annotations['label'] = self.annotations['label'].apply(lambda x: LABEL_TO_NUM[x])
        self.preprocessing = preprocessing
        self.transform = v2.Compose(
            [v2.Resize((64, 64)),
             v2.ToImage(),
             v2.ToDtype(torch.float32, scale=True),
             v2.Lambda(lambda x: x * 2 - 1)])

    def __getitem__(self, item):
        img_path = os.path.join(self.root_dir, str(self.annotations.loc[item, 'pth']))
        img = Image.open(img_path)
        tensor = self.transform(img)
        label = torch.tensor(int(self.annotations.loc[item, 'label']))
        if self.preprocessing:
            tensor = self.preprocessing(tensor)
        return tensor, label

    def __len__(self):
        return len(self.annotations)


class Fer2013(Dataset):
    def __init__(self, preprocessing=None, black_and_white=False):
        if not black_and_white:
            raise ValueError("Images with color are not supported for the FER2013 dataset.")
        self.preprocessing = preprocessing

    def __getitem__(self, item):
        raise NotImplemented()

    def __len__(self):
        raise NotImplemented()


class RAF_DB(Dataset):
    def __init__(self, preprocessing=None, black_and_white=False):
        """
        Initialize the RAF_DB dataset.

        Args:
            preprocessing: A torchvision transform that is applied to the images.
            black_and_white: If True, the images will be converted to grayscale.

        returns:
            The training data of the RAFDB dataset since the validation data is already used for testing.
        """

        csv_path = os.path.join(os.getenv('DATASET_RAF_DB_PATH'), 'train_labels.csv')
        root_dir = os.getenv('DATASET_RAF_DB_PATH')
        self.root_dir = root_dir

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {csv_path} not found. Could not load RAF-DB dataset.")
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found. Could not load RAF-DB dataset.")

        self.annotations = pd.read_csv(csv_path)
        self.annotations['pth'] = self.annotations.apply(lambda x: f"DATASET/train/{x['label']}/{x['image']}", axis=1)

        rafdb_reorder_num_labels = {
            1: LABEL_TO_NUM['surprise'],
            2: LABEL_TO_NUM['fear'],
            3: LABEL_TO_NUM['disgust'],
            4: LABEL_TO_NUM['happiness'],
            5: LABEL_TO_NUM['sadness'],
            6: LABEL_TO_NUM['anger']
        }

        # remove 'neutral' images from dataset
        self.annotations = self.annotations[self.annotations['label'] != 7]
        # Encode the 'label' column using the label_encode dictionary
        self.annotations['label'] = self.annotations['label'].apply(lambda x: rafdb_reorder_num_labels[x])
        # reset index
        self.annotations = self.annotations.reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = v2.Compose(
            [v2.Resize((64, 64)),
             v2.ToImage(),
             v2.ToDtype(torch.float32, scale=True),
             v2.Lambda(lambda x: x * 2 - 1)])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, str(self.annotations.loc[index, 'pth']))
        y_label = torch.tensor(int(self.annotations.loc[index, 'label']))

        with Image.open(img_path) as img:
            if self.transform:
                img = self.transform(img)

        return img, y_label
