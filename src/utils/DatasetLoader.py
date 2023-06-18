
import gc
from typing import Optional, Tuple
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset


__datasets__ = (
    'CUB_200_2011',
)


def get_best_batch_size(
    model: nn.Module,
    dataset: datasets.VisionDataset,
    device: torch.device,
) -> int:

    model.eval()
    model.to(device)
    # Get the batch size that fits in memory
    batch_size = 1
    while True:
        try:
            batch_size *= 2
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=8,
            )
            for _, batch in dataloader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                model(images)
                break
        except RuntimeError:
            batch_size = batch_size // 2
            break

    model.to('cpu')
    del dataloader

    torch.cuda.empty_cache()
    gc.collect()
    return batch_size


def load_dataset(
    dataset_name: str,
    resize: Optional[Tuple[int, int]] = None,
    **kwargs
) -> datasets.VisionDataset:
    """
    Load a dataset from torchvision.datasets

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load
    resize : Optional[Tuple[int, int]], optional
        An optinal value that resizes all returned images, by default None

    Returns
    -------
    _return_type_
        The loaded dataset

    Raises
    ------
    ValueError
        If the dataset name is not found in torchvision.datasets
    """

    if (dataset_name not in datasets.__dict__ and
            dataset_name not in __datasets__):

        raise ValueError(
            f"""Dataset {dataset_name} not found in torchvision.datasets and
            is not in the list of custom datasets: {__datasets__}."""
        )

    # Define the transforms
    if resize is not None:
        transform = T.Compose([
            T.Resize(resize, antialias=True),
            T.ToTensor(),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
        ])

    # Load the dataset
    root = Path(f"./data/{dataset_name}")
    if dataset_name == 'CUB_200_2011':
        dataset = Cub2011(
            root=root,
            transform=transform,
            download=True,
            **kwargs
        )
        return IndexedDataset(dataset)

    dataset = datasets.__dict__[dataset_name](
        root=root,
        download=True,
        transform=transform,
        **kwargs
    )

    return IndexedDataset(dataset)


class IndexedDataset(datasets.VisionDataset):

    def __init__(self, dataset: datasets.VisionDataset):
        super().__init__(root=dataset.root, transform=dataset.transform)
        self.dataset = dataset

    def __getitem__(self, index):
        return index, self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class Cub2011(Dataset):
    """
    Class for CUB_200_2011 dataset.
    Written by https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
    """

    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
