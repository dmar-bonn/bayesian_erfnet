import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from agri_semantics.utils import resize


class UAVDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings
        pass

    def setup(self, stage: str = None):
        """Perform train, val, test split.

        Args:
            stage (str, optional): setup logic for trainer.{fit,validate,test}
                                   if setup is called with stage = None, we assume all stages have been set-up
                                   Defaults to None.
        """

        path_to_split_info: str = os.path.join(self.cfg["data"]["path_to_dataset"], "split.yaml")
        with open(path_to_split_info) as istream:
            split_info = yaml.safe_load(istream)

        train_filenames = split_info["train"]
        val_filenames = split_info["valid"]
        test_filenames = split_info["test"]

        path_to_dataset = self.cfg["data"]["path_to_dataset"]

        img_width = self.cfg["data"]["img_prop"]["width"]
        img_height = self.cfg["data"]["img_prop"]["height"]
        keep_ap = self.cfg["data"]["img_prop"]["keep_aspect_ratio"]

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self._uav_train = UAVDataset(
                path_to_dataset, train_filenames, img_width=img_width, img_height=img_height, keep_aspect_ratio=keep_ap
            )
            self._uav_val = UAVDataset(
                path_to_dataset, val_filenames, img_width=img_width, img_height=img_height, keep_aspect_ratio=keep_ap
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self._uav_test = UAVDataset(
                path_to_dataset, test_filenames, img_width=img_width, img_height=img_height, keep_aspect_ratio=keep_ap
            )

    def train_dataloader(self):
        shuffle: bool = self.cfg["data"]["train_shuffle"]
        batch_size: int = self.cfg["data"]["batch_size"]
        n_workers: int = self.cfg["data"]["num_workers"]

        loader = DataLoader(self._uav_train, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)

        return loader

    def val_dataloader(self):
        batch_size: int = self.cfg["data"]["batch_size"]
        n_workers: int = self.cfg["data"]["num_workers"]

        loader = DataLoader(self._uav_val, batch_size=batch_size, num_workers=n_workers, shuffle=False)

        return loader

    def test_dataloader(self):
        batch_size: int = self.cfg["data"]["batch_size"]
        n_workers: int = self.cfg["data"]["num_workers"]

        loader = DataLoader(self._uav_test, batch_size=batch_size, num_workers=n_workers, shuffle=False)

        return loader


def is_image(filename: str) -> bool:
    """Check whether or not a given file is an image based on its format.

    Args:
        filename (str): filename

    Returns:
        bool: whether or not a given file is an image
    """
    return any(filename.endswith(ext) for ext in [".jpg", ".png"])


class UAVDataset(Dataset):
    """Represents the UAV dataset.

    The directory structure is as following:
    ├── annotations
    │   └── dlp
    │       └── iMapDa
    └── images
        └── rgb
    └── split.yaml
    """

    def __init__(
        self,
        path_to_dataset: str,
        filenames: List[str],
        img_width: Optional[int],
        img_height: Optional[int],
        keep_aspect_ratio: bool,
    ):
        """Get the path to all images and its corresponding annotations.

        Args:
            path_to_dataset (str): path to dir that contains the images and annotations
            filenames (List[str]): list of filenames which are considered to be part of this dataset, e.g, [filename01.png, filename02.png, ...]
            img_width (Optional[int]): specify image width - if this value is equal to 'None' we keep the original width
            img_height (Optional[int]): specify image height - if this value is equal to 'None' we keep the original height
            keep_aspect_ratio (bool): specify if aspect ratio should stay the same
        """
        if not os.path.exists(path_to_dataset):
            raise FileNotFoundError

        assert filenames, "Cannot create an empty dataset"

        super().__init__()

        # get path to all RGB images
        self.path_to_images: str = os.path.join(path_to_dataset, "images", "rgb")
        if not os.path.exists(self.path_to_images):
            raise FileNotFoundError

        self.image_files: List[str] = []
        for fname in os.listdir(self.path_to_images):
            if is_image(fname):
                if fname in filenames:
                    self.image_files.append(fname)
        self.image_files.sort()

        # get path to all ground-truth semantic masks
        self.path_to_annos: str = os.path.join(path_to_dataset, "annotations", "dlp", "iMapDa")
        if not os.path.exists(self.path_to_annos):
            raise FileNotFoundError

        self.anno_files: List[str] = []
        for fname in os.listdir(self.path_to_annos):
            if is_image(fname):
                if fname in filenames:
                    self.anno_files.append(fname)
        self.anno_files.sort()

        assert len(self.image_files) == len(self.anno_files), "Number of images and annos does not match."

        # specify image transformations
        self.img_to_tensor = transforms.ToTensor()
        self.img_width = img_width
        self.img_height = img_height
        self.keep_aspect_ratio = keep_aspect_ratio

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample of the dataset.

        Args:
            idx (int): index of sample in the dataset

        Returns:
            Dict[str, torch.Tensor]:
             - 'data' is an image of shape [C x H x W] in range(0.0, 1.0)
             - 'anno' is the corresponding annotation in {0, 1, 2}
        """
        path_to_current_img: str = os.path.join(self.path_to_images, self.image_files[idx])
        img = Image.open(path_to_current_img)  # PIL.PngImagePlugin.PngImageFile
        img = self.img_to_tensor(img)
        img = resize(
            img, width=self.img_width, height=self.img_height, interpolation=1, keep_aspect_ratio=self.keep_aspect_ratio
        )

        path_to_current_anno: str = os.path.join(self.path_to_annos, self.anno_files[idx])
        anno = cv2.imread(path_to_current_anno, cv2.IMREAD_UNCHANGED)  # dtype: uint16
        anno = anno.astype(np.int64)  # torch does not support conversion of uint16
        anno = UAVDataset.remap_annotation(anno)
        anno = torch.Tensor(anno).type(torch.int64)
        anno = resize(
            anno,
            width=self.img_width,
            height=self.img_height,
            interpolation=0,
            keep_aspect_ratio=self.keep_aspect_ratio,
        )

        return {"data": img, "anno": anno}

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def remap_annotation(anno: np.ndarray) -> np.ndarray:
        """Remap the values in the annotation.

        We need successive labels for training.

        However, the input annotations have the labels:
        -     0 := background,
        -     2 := weed,
        - 10000 := crop

        After remapping the annotations have the labels:
        - 0 := background,
        - 1 := crop,
        - 2 := weed

        Args:
            anno (np.ndarray): input annotations

        Returns:
            np.ndarray: remapped annotation
        """
        remapped = anno.copy()

        mask = anno == 10000
        remapped[mask] = 1

        return remapped
