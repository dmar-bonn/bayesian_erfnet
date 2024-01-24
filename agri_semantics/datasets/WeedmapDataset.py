import os
from typing import Dict, List

import cv2
import numpy as np
import torch
from agri_semantics.transformations import Transformation, get_transformations
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class WeedmapDataModule(LightningDataModule):
    def __init__(self, cfg: Dict):
        super().__init__()

        self.cfg = cfg
        self.active_learning = False  # "active_learning" in self.cfg
        self.max_collected_images = cfg["active_learning"]["max_collected_images"] if self.active_learning else -1
        self.data_indices = np.array([0])
        self.all_indices = None

    def setup(self, stage: str = None):
        path_to_training_dataset = os.path.join(self.cfg["data"]["path_to_dataset"], "training_set")
        path_to_validation_dataset = os.path.join(self.cfg["data"]["path_to_dataset"], "validation_set")
        path_to_test_dataset = os.path.join(self.cfg["data"]["path_to_dataset"], "test_set")

        if stage == "fit" or stage is None:
            self._train = WeedmapDataset(
                path_to_training_dataset, transformations=get_transformations(self.cfg, "train")
            )
            self._val = WeedmapDataset(path_to_validation_dataset, transformations=get_transformations(self.cfg, "val"))

            if self.active_learning and self.all_indices is None:
                train_indices = np.arange(len(self._train))
                max_num_indices = min(len(self._train), self.max_collected_images)
                self.all_indices = np.sort(np.random.choice(train_indices, size=max_num_indices, replace=False))

        if stage == "test" or stage is None:
            self._test = WeedmapDataset(path_to_test_dataset, transformations=get_transformations(self.cfg, "test"))

    def append_data_indices(self, indices):
        self.data_indices = np.unique(np.append(self.data_indices, indices))

    def get_data_indices(self):
        return self.data_indices

    def get_unlabeled_data_indices(self):
        msk = ~np.in1d(self.all_indices, self.data_indices)

        return self.all_indices[msk]

    def unlabeled_dataloader(self):
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        train_dataset = self._train
        unlabeled_data = Subset(train_dataset, self.get_unlabeled_data_indices())

        loader = DataLoader(unlabeled_data, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        return loader

    def train_dataloader(self):
        shuffle = self.cfg["data"]["train_shuffle"]
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        train_dataset = self._train
        if self.active_learning:
            train_dataset = Subset(train_dataset, self.data_indices)

        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)

        return loader

    def val_dataloader(self):
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(self._val, batch_size=batch_size, num_workers=n_workers, shuffle=False)

        return loader

    def test_dataloader(self):
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(self._test, batch_size=batch_size, num_workers=n_workers, shuffle=False)

        return loader


def is_image(filename):
    return any(filename.endswith(ext) for ext in [".jpg", ".png", ".npy"])


class WeedmapDataset(Dataset):
    def __init__(self, path_to_dataset: str, transformations: List[Transformation]):
        super().__init__()

        if not os.path.exists(path_to_dataset):
            raise FileNotFoundError(f"Weedmap dataset path '{path_to_dataset}' not found")

        self.path_to_images = os.path.join(path_to_dataset, "image")
        if not os.path.exists(self.path_to_images):
            raise FileNotFoundError(f"Weedmap images path '{self.path_to_images}' not found")

        self.image_files = []
        for fname in os.listdir(self.path_to_images):
            if is_image(fname):
                self.image_files.append(fname)
        self.image_files.sort()

        self.path_to_annos = os.path.join(path_to_dataset, "anno")
        if not os.path.exists(self.path_to_annos):
            raise FileNotFoundError(f"Weedmap annotations path '{self.path_to_annos}' not found")

        self.anno_files = []
        for fname in os.listdir(self.path_to_annos):
            if is_image(fname):
                self.anno_files.append(fname)
        self.anno_files.sort()

        self.img_to_tensor = transforms.ToTensor()
        self.transformations = transformations

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample of the dataset.

        Args:
            idx (int): index of sample in the dataset

        Returns:
            Dict[str, torch.Tensor]:
             - 'data' is an image of shape [C x H x W] in range(0.0, 1.0)
             - 'anno' is the corresponding annotation in {0, 1, 2}
        """
        path_to_current_img = os.path.join(self.path_to_images, self.image_files[idx])
        img_raw = np.load(path_to_current_img)
        img = self.img_to_tensor(img_raw)

        path_to_current_anno = os.path.join(self.path_to_annos, self.anno_files[idx])
        anno = self.get_anno(path_to_current_anno)

        for transformer in self.transformations:
            img_raw, img, anno = transformer(img_raw, img, anno)

        anno = self.remap_annotation(anno.numpy())

        return {"data": img, "image": img, "anno": anno, "index": idx}

    @staticmethod
    def get_anno(path_to_current_anno: str) -> torch.Tensor:
        anno = cv2.imread(path_to_current_anno)
        anno = anno.astype(np.int64)  # torch does not support conversion of uint16
        anno = np.moveaxis(anno, -1, 0)  # now in CHW mode
        return torch.from_numpy(anno).long()

    @staticmethod
    def remap_annotation(anno: np.array) -> torch.Tensor:
        """
        After remapping the annotations have the labels:

        potsdam:
            - 0 := soil/background,
            - 1 := crop,
            - 2 := weed,
        """
        dims = anno.shape
        assert len(dims) == 3, "wrong matrix dimension!!!"
        assert dims[0] == 3, "annotation must have 3 channels!!!"
        remapped = np.zeros((dims[1], dims[2]))

        crop_mask = anno[1, :, :] == 255  # G = 255
        weed_mask = anno[2, :, :] == 255  # R = 255

        remapped[crop_mask] = 1
        remapped[weed_mask] = 2

        return torch.from_numpy(remapped).long()

    def __len__(self):
        return len(self.image_files)
