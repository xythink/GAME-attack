from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch, torchvision
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from typing import Any, Tuple, Callable, Optional
from torchvision.datasets.utils import check_integrity, verify_str_arg, download_and_extract_archive, download_url
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
import PIL.Image

import os
import pandas as pd
import numpy as np
from PIL import Image

datasets_dict = {
    "MNIST": {
        "location": "/home/data/mnist",
        "img_size": 32,
        "n_channels": 1,
        "n_outputs": 10,
    },
    "FashionMNIST": {
        "location": "/home/data",
        "img_size": 32,
        "n_channels": 1,
        "n_outputs": 10,
    },
    "EMNIST-letters":{
        "location": "/home/data",
        "img_size": 32,
        "n_channels": 1,
        "n_outputs": 26,
    },
    "EMNIST-digits":{
        "location": "/home/data",
        "img_size": 32,
        "n_channels": 1,
        "n_outputs": 10,
    },
    "CIFAR10":{
        "location":"/home/data",
        "img_size": 32,
        "n_channels": 3,
        "n_outputs": 10,
    },
    "CIFAR100":{
        "location":"/home/data",
        "img_size": 32,
        "n_channels": 3,
        "n_outputs": 100,
    },
    "GTSRB":{
        "location": "/home/data",
        "img_size": 32,
        "n_channels": 3,
        "n_outputs": 43,
    },
    "BelgiumTSC":{
        "location": "/home/data",
        "img_size": 32,
        "n_channels": 3,
        "n_outputs": 62,
    },
    "Flowers102":{
        "location": "/home/data",
        "img_size": 32,
        "n_channels": 3,
        "n_outputs": 102,
    },
    "Food101":{
        "location": "/home/data",
        "img_size": 32,
        "n_channels": 3,
        "n_outputs": 101,
    },
    "SVHN":{
        "location": "/home/data",
        "img_size": 32,
        "n_channels": 3,
        "n_outputs": 10,
    },
    "STL10":{
        "location": "/home/data",
        "img_size": 32,
        "n_channels": 3,
        "n_outputs": 10,
    }
}


def getdataset(name: str, train=True):
    if name == 'STL10':
        dataset = torchvision.datasets.__dict__[name](
            datasets_dict[name]["location"],
            split='train' if train else 'test',
            download=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(datasets_dict[name]["img_size"]), 
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                    )
                    ]
            ),
        )
    elif name == "GTSRB":
        dataset = GTSRB(
            root_dir='/home/data',
            train= train,
            transform=transforms.Compose(
                [
                    transforms.Resize((datasets_dict[name]["img_size"],datasets_dict[name]["img_size"])),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                        )
                ]
            ) if train else 
            transforms.Compose(
                [
                    transforms.Resize((datasets_dict[name]["img_size"], datasets_dict[name]["img_size"])),
                    transforms.ToTensor(),
                    transforms.Normalize(torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5),
                ]
                )
            )
    elif name == "BelgiumTSC":
        dataset = BelgiumTSC(
            root_dir='/home/data',
            train= train,
            transform=transforms.Compose(
                [
                    transforms.Resize((datasets_dict[name]["img_size"],datasets_dict[name]["img_size"])), 
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                    )
                    ]
            )
        )
    elif name == "CelebA":
        image_path = datasets_dict[name]["location"]
        label_path = datasets_dict[name]["label"]
        trans = transforms.Compose(
                [
                    transforms.Resize((datasets_dict[name]["img_size"],datasets_dict[name]["img_size"])), 
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                    )
                    ]
            )
        dataset = CelebA(image_path, label_path, transform=trans)
        train_size = int(len(dataset) * 0.7)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        if train:
            return train_dataset
        else:
            return test_dataset
    elif name == "EMNIST-letters":
        dataset = torchvision.datasets.EMNIST(
            datasets_dict[name]["location"],
            train=train,
            split="letters",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(datasets_dict[name]["img_size"]), 
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                    )
                    ]
            ),           
        )
    elif name == "EMNIST-digits":
        dataset = torchvision.datasets.EMNIST(
            datasets_dict[name]["location"],
            train=train,
            split="digits",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(datasets_dict[name]["img_size"]), 
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                    )
                    ]
            ),           
        )
    elif name == "Flowers102":
        dataset = Flowers102(
            datasets_dict[name]["location"],
            split='train' if train else 'test',
            download=True,
            transform=
            transforms.Compose(
                [
                    # transforms.Resize((datasets_dict[name]["img_size"],datasets_dict[name]["img_size"])), 
                    transforms.RandomResizedCrop(datasets_dict[name]["img_size"]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                    )
                    ]
            ) if train else 
            transforms.Compose(
                [
                    transforms.Resize((datasets_dict[name]["img_size"],datasets_dict[name]["img_size"])), 
                    # transforms.RandomResizedCrop(datasets_dict[name]["img_size"]),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                    )
                    ]
            ),
        )
    elif name == "SVHN":
        dataset = torchvision.datasets.SVHN(
            datasets_dict[name]["location"],
            split='train' if train else 'test',
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(datasets_dict[name]["img_size"]), 
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                    )
                    ]
            ),
        )
    elif name == "Food101":
        dataset = torchvision.datasets.__dict__[name](
            datasets_dict[name]["location"],
            split='train' if train else 'test',
            download=True,
            transform=
            transforms.Compose(
                [
                    # transforms.Resize((datasets_dict[name]["img_size"],datasets_dict[name]["img_size"])), 
                    transforms.RandomResizedCrop(datasets_dict[name]["img_size"]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                    )
                    ]
            ) if train else 
            transforms.Compose(
                [
                    transforms.Resize((datasets_dict[name]["img_size"],datasets_dict[name]["img_size"])), 
                    # transforms.RandomResizedCrop(datasets_dict[name]["img_size"]),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                    )
                    ]
            ),
        )


    else:
        dataset = torchvision.datasets.__dict__[name](
            datasets_dict[name]["location"],
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(datasets_dict[name]["img_size"]), 
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        torch.ones(datasets_dict[name]["n_channels"])*0.5, torch.ones(datasets_dict[name]["n_channels"])*0.5
                    )
                    ]
            ),
        )



    return dataset


def getloader(dataname:str, train=True, batch_size=64, shuffle=True, n_works=4)->DataLoader:
    dataloader = DataLoader(
        getdataset(dataname, train=train),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_works,
        pin_memory=True,
    )
    return dataloader


class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.data = pd.read_csv(csv_file_path)

        self.transform = transform
        self.classes = range(43)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId


class BelgiumTSC(Dataset):
    base_folder = 'BelgiumTSC'

    def __init__(self, root_dir, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.sub_directory = 'Training' if train else 'Testing'
        self.csv_file_name = 'train_data.csv' if train else 'test_data.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId


class CelebA(Dataset):
    def __init__(self, image_path, label_path, transform = None):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.n_images = 0
        self.cls_name = []
        self.label = self._get_celeba_label_()
        self.classes = range(43)

    def __getitem__(self, index):
        file_name = "%06d.jpg"%(index+1)
        image = Image.open(os.path.join(self.image_path,file_name))
        if self.transform:
            image = self.transform(image)
        label = self.label[index]
        # label = torch.index_select(label,0,torch.Tensor(datasets_dict['CelebA']["label_index"]).type(torch.int64))
        label = label[datasets_dict['CelebA']["label_index"]]
        return image, label.long()
    
    def __len__(self):
        
        return self.n_images
    
    def _get_celeba_label_(self):

        with open(self.label_path) as F:
            num = int(F.readline())
            self.n_images = num
            self.cls_name = F.readline().split()
            res = torch.ones([num,40])* (-100)
            for i in range(num):
                line_list = F.readline().split()
                assert len(line_list) == 41
                img_index = int(line_list[0].split('.')[0])
                assert i+1 == img_index
                res[i] = torch.Tensor([int(line_list[j]) for j in range(1,41)])
        res[res==-1]=0
        return res

# class Flowers102(torchvision.datasets.Flowers102):
#     def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
#         super().__init__(root, split, transform, target_transform, download)

#     def __len__(self) -> int:
#         return super().__len__()
    
#     def __getitem__(self, idx) -> Tuple[Any, Any]:
#         image, label = super().__getitem__(idx)
#         return image, label - 1


class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label -1 

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)