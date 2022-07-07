import os
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from agri_semantics.transformations import get_transformations, Transformation


class PotsdamDataModule(LightningDataModule):
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

        # Assign datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self._train = PotsdamDataset(
                path_to_training_dataset, transformations=get_transformations(self.cfg, "train")
            )
            self._val = PotsdamDataset(path_to_validation_dataset, transformations=get_transformations(self.cfg, "val"))

            if self.active_learning and self.all_indices is None:
                train_indices = np.arange(len(self._train))
                max_num_indices = min(len(self._train), self.max_collected_images)
                self.all_indices = np.sort(np.random.choice(train_indices, size=max_num_indices, replace=False))

        if stage == "test" or stage is None:
            self._test = PotsdamDataset(path_to_test_dataset, transformations=get_transformations(self.cfg, "test"))

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
    return any(filename.endswith(ext) for ext in [".jpg", ".png"])


class PotsdamDataset(Dataset):
    def __init__(self, path_to_dataset, transformations: List[Transformation]):
        super().__init__()

        if not os.path.exists(path_to_dataset):
            raise FileNotFoundError

        # get path to all RGB images
        self.path_to_images = os.path.join(path_to_dataset, "image")
        if not os.path.exists(self.path_to_images):
            raise FileNotFoundError

        self.image_files = []
        for fname in os.listdir(self.path_to_images):
            if is_image(fname):
                self.image_files.append(fname)
        self.image_files.sort()

        # get path to all ground-truth semantic annotation
        self.path_to_annos = os.path.join(path_to_dataset, "anno")
        if not os.path.exists(self.path_to_annos):
            raise FileNotFoundError

        self.anno_files = []
        for fname in os.listdir(self.path_to_annos):
            if is_image(fname):
                self.anno_files.append(fname)
        self.anno_files.sort()

        # specify image transformations
        self.img_to_tensor = transforms.ToTensor()
        self.transformations = transformations

    def __getitem__(self, idx):
        path_to_current_img = os.path.join(self.path_to_images, self.image_files[idx])
        img_pil = Image.open(path_to_current_img)
        img = self.img_to_tensor(img_pil)

        path_to_current_anno = os.path.join(self.path_to_annos, self.anno_files[idx])
        anno = cv2.imread(path_to_current_anno)
        anno = anno.astype(np.int64)  # torch does not support conversion of uint16
        anno = np.moveaxis(anno, -1, 0)  # now in CHW mode
        anno = torch.Tensor(anno).type(torch.int64)

        # apply a set of transformations to the raw_image, image and anno
        for transformer in self.transformations:
            img_pil, img, anno = transformer(img_pil, img, anno)

        anno = self.remap_annotation(anno.numpy())

        return {"data": img, "image": img, "anno": anno, "index": idx}

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def remap_annotation(anno):
        """
        After remapping the annotations have the labels:

        potsdam:
            - 0 := boundary line,
            - 1 := imprevious surfaces,
            - 2 := building,
            - 3 := low vegetation,
            - 4 := tree,
            - 5 := car,
            - 6 := clutter/background
        """

        dims = anno.shape
        assert len(dims) == 3, "wrong matrix dimension!!!"
        assert dims[0] == 3, "annotation must have 3 channels!!!"
        remapped = np.zeros((dims[1], dims[2]))

        mask1 = anno[0, :, :] == 255  # B = 255
        mask2 = anno[1, :, :] == 255  # G = 255
        mask3 = anno[2, :, :] == 255  # R = 255
        mask4 = anno[0, :, :] == 0  # B = 0
        mask5 = anno[1, :, :] == 0  # G = 0
        mask6 = anno[2, :, :] == 0  # R = 0

        mask_surface = mask1 * mask2 * mask3
        remapped[mask_surface] = 1

        mask_building = mask1 * mask5 * mask6
        remapped[mask_building] = 2

        mask_veg = mask1 * mask2 * mask6
        remapped[mask_veg] = 3

        mask_tree = mask2 * mask4 * mask6
        remapped[mask_tree] = 4

        mask_car = mask2 * mask3 * mask4
        remapped[mask_car] = 5

        mask_bg = mask3 * mask4 * mask5
        remapped[mask_bg] = 6

        return torch.Tensor(remapped).type(torch.int64)
