import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from utils import LABEL_TO_STR, LABEL_TO_NUM, load_images, label_from_path

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
        "RAF-DB": RafDb
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
             v2.Normalize(mean=[0.5], std=[0.5])])

    def __getitem__(self, item):
        # needed for sweep to work (don't ask why)
        if item >= len(self.annotations):
            raise IndexError(f"Index {item} out of range")

        img_path = os.path.join(self.root_dir, str(self.annotations.loc[item, 'pth']))
        # close the image after reading it
        with Image.open(img_path) as img:
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


class RafDb(Dataset):
    def __init__(self, preprocessing=None, black_and_white=False):
        """
        Initialize the RAF_DB dataset.

        Args:
            preprocessing: A torchvision transform that is applied to the images.
            black_and_white: If True, the images will be converted to grayscale.

        returns:
            The training data of the RAF-DB dataset since the validation data is already used for testing.
        """

        root_dir = os.getenv('DATASET_RAF_DB_PATH')
        self.root_dir = root_dir

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found. Could not load RAF-DB dataset.")

        if black_and_white:
            raise NotImplementedError("Black and white images are not yet supported for the RAF-DB dataset.")

        images = load_images(root_dir)
        train_labels = [label_from_path(path) for path, _ in images]

        self.annotations = pd.DataFrame({'pth': [path for path, _ in images], 'label': train_labels})

        self.root_dir = root_dir
        self.preprocessing = preprocessing
        self.transform = v2.Compose(
            [v2.Resize((64, 64)),
             v2.ToImage(),
             v2.ToDtype(torch.float32, scale=True),
             v2.Normalize(mean=[0.5], std=[0.5])])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # needed for sweep to work (don't ask why)
        if index >= len(self.annotations):
            raise IndexError(f"Index {index} out of range")

        img_path = self.annotations.loc[index, 'pth']
        y_label = torch.tensor(int(self.annotations.loc[index, 'label']))

        image = None
        with Image.open(img_path) as img:
            if self.transform:
                img = self.transform(img)
        image = img
        if self.preprocessing:
            image = self.preprocessing(image)
        return image, y_label


def compare_images(img1, img2, win_size=5):
    """Compare two images using SSIM."""
    img1 = img1.numpy().transpose(1, 2, 0)
    img2 = img2.numpy().transpose(1, 2, 0)
    score, _ = ssim(img1, img2, multichannel=True, full=True, win_size=win_size, channel_axis=2, data_range=2.0)
    return score


if __name__ == '__main__':
    dataset = get_dataset("RAF-DB")
    test_path = "/Users/marius/github/fer-cvdl-practical/data/test"
    test_images = load_images(test_path)

    # start_id = number of images in the data/train folder
    train_path = "/Users/marius/github/fer-cvdl-practical/data/train"
    start_id = len([file for file in os.listdir(train_path) if file.endswith(".jpg")]) + 1
    print(f"Starting id: {start_id}")

    for i in tqdm(range(len(dataset))):
        same = False
        for j in range(len(test_images)):
            similarity = compare_images(dataset[i][0], test_images[j][1])
            if similarity > 0.9:  # You can adjust this threshold
                print(f"High similarity found between images: dataset index {i} and test image index {j}")
                # plot the images
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(dataset[i][0].numpy().transpose(1, 2, 0))
                ax[1].imshow(test_images[j][1].numpy().transpose(1, 2, 0))
                plt.show()
                same = True
                break
        if not same:
            img = dataset[i][0].numpy().transpose(1, 2, 0)
            img = (img + 1) / 2  # Rescale to [0, 1]
            img = (img * 255).astype(np.uint8)  # Convert to uint8
            img = Image.fromarray(img)  # Convert to PIL Image

            # get the label using the LABEL_TO_STR dictionary
            label = LABEL_TO_STR[dataset[i][1].item()]
            # get the id using the start_id and the index
            id = start_id + i
            # save the image
            img.save(f"{train_path}/img{id}_{label}.jpg")
