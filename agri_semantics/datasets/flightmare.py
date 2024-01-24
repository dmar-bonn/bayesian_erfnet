import os
from typing import Dict, List, Union

import cv2
import numpy as np
import torch
from PIL import Image
from agri_semantics.transformations import get_transformations, Transformation
from agri_semantics.utils.utils import LABELS
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class FlightmareDataModule(LightningDataModule):
    def __init__(self, cfg: Dict, human_data: bool, pseudo_data: bool):
        super().__init__()

        self.cfg = cfg
        self.task = cfg["model"]["task"]
        self.active_learning = False  # "active_learning" in self.cfg
        self.max_collected_images = cfg["active_learning"]["max_collected_images"] if self.active_learning else -1
        self.data_indices = np.array([0])
        self.all_indices = None

        self.human_data = human_data
        self.pseudo_data = pseudo_data

    def setup(self, stage: str = None):
        path_to_training_human_dataset = os.path.join(self.cfg["data"]["path_to_dataset"], "training_set", "human")
        path_to_training_pseudo_dataset = os.path.join(self.cfg["data"]["path_to_dataset"], "training_set", "pseudo")

        path_to_validation_dataset = os.path.join(self.cfg["data"]["path_to_dataset"], "validation_set")
        path_to_test_dataset = os.path.join(self.cfg["data"]["path_to_dataset"], "test_set")

        if stage == "fit" or stage is None:
            self._train_human, self._train_pseudo = None, None
            if self.human_data:
                self._train_human = FlightmareDataset(
                    path_to_training_human_dataset, self.cfg, transformations=get_transformations(self.cfg, "train")
                )
            if self.pseudo_data:
                try:
                    self._train_pseudo = FlightmareDataset(
                        path_to_training_pseudo_dataset,
                        self.cfg,
                        transformations=get_transformations(self.cfg, "train"),
                    )
                except:
                    pass

            self._val = FlightmareDataset(
                path_to_validation_dataset, self.cfg, transformations=get_transformations(self.cfg, "val")
            )

            if self.active_learning and self.all_indices is None:
                train_indices = np.arange(len(self._train_human))
                max_num_indices = min(len(self._train_human), self.max_collected_images)
                self.all_indices = np.sort(np.random.choice(train_indices, size=max_num_indices, replace=False))

        if stage == "test" or stage is None:
            self._test = FlightmareDataset(
                path_to_test_dataset, self.cfg, transformations=get_transformations(self.cfg, "test")
            )

    def append_data_indices(self, indices: np.array):
        self.data_indices = np.unique(np.append(self.data_indices, indices))

    def get_data_indices(self) -> np.array:
        return self.data_indices

    def get_unlabeled_data_indices(self) -> np.array:
        msk = ~np.in1d(self.all_indices, self.data_indices)

        return self.all_indices[msk]

    def unlabeled_dataloader(self) -> DataLoader:
        batch_size = self.cfg["data"]["batch_size"]["total"]
        n_workers = self.cfg["data"]["num_workers"]

        train_dataset = self._train_human
        unlabeled_data = Subset(train_dataset, self.get_unlabeled_data_indices())

        loader = DataLoader(
            unlabeled_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
        )

        return loader

    def train_dataloader(self) -> Union[Dict, DataLoader]:
        shuffle = self.cfg["data"]["train_shuffle"]
        batch_size_total = self.cfg["data"]["batch_size"]["total"]
        batch_size_human = self.cfg["data"]["batch_size"]["human"] if self._train_pseudo else batch_size_total
        batch_size_pseudo = self.cfg["data"]["batch_size"]["pseudo"] if self._train_pseudo else 0
        n_workers = self.cfg["data"]["num_workers"]

        if self.active_learning:
            return DataLoader(
                Subset(self._train_human, self.data_indices),
                batch_size=batch_size_total,
                shuffle=shuffle,
                num_workers=n_workers,
            )

        loaders = {}
        if self._train_human:
            loaders["human"] = DataLoader(
                self._train_human,
                batch_size=batch_size_human,
                shuffle=shuffle,
                num_workers=n_workers,
            )
        if self._train_pseudo:
            loaders["pseudo"] = DataLoader(
                self._train_pseudo,
                batch_size=batch_size_pseudo,
                shuffle=shuffle,
                num_workers=n_workers,
            )

        return loaders

    def val_dataloader(self) -> DataLoader:
        batch_size = self.cfg["data"]["batch_size"]["total"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(
            self._val,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=False,
        )

        return loader

    def test_dataloader(self) -> DataLoader:
        batch_size = self.cfg["data"]["batch_size"]["total"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(
            self._test,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=False,
        )

        return loader


def is_image(filename):
    return any(filename.endswith(ext) for ext in [".jpg", ".png"])


def is_numpy_array(filename):
    return any(filename.endswith(ext) for ext in [".np", ".npy"])


class FlightmareDataset(Dataset):
    def __init__(self, path_to_dataset: str, cfg: Dict, transformations: List[Transformation]):
        super().__init__()

        self.task = cfg["model"]["task"]
        self.cfg = cfg

        if not os.path.exists(path_to_dataset):
            raise FileNotFoundError(f"Cannot find dataset at {path_to_dataset}")

        # get path to all RGB images
        self.path_to_images = os.path.join(path_to_dataset, "image")
        if not os.path.exists(self.path_to_images):
            raise FileNotFoundError(f"Cannot find images at {self.path_to_images}")

        self.image_files = []
        for fname in os.listdir(self.path_to_images):
            if is_image(fname):
                self.image_files.append(fname)
        self.image_files.sort()

        # get path to all ground-truth annotations
        self.path_to_annos = os.path.join(path_to_dataset, "anno")
        if not os.path.exists(self.path_to_annos):
            raise FileNotFoundError(f"Cannot find labels at {self.path_to_annos}")

        self.anno_files = []
        for fname in os.listdir(self.path_to_annos):
            if is_image(fname):
                self.anno_files.append(fname)
        self.anno_files.sort()

        # get path to all annotation masks
        self.load_anno_masks = False
        self.path_to_anno_masks = os.path.join(path_to_dataset, "anno_mask")
        if os.path.exists(self.path_to_anno_masks):
            self.load_anno_masks = True
            self.anno_mask_files = []
            for fname in os.listdir(self.path_to_anno_masks):
                if is_numpy_array(fname):
                    self.anno_mask_files.append(fname)
            self.anno_mask_files.sort()

        # get path to all prediction uncertainties
        self.load_uncertainties = False
        self.path_to_uncertainties = os.path.join(path_to_dataset, "uncertainty")
        if os.path.exists(self.path_to_uncertainties):
            self.load_uncertainties = True
            self.uncertainty_files = []
            for fname in os.listdir(self.path_to_uncertainties):
                if is_image(fname):
                    self.uncertainty_files.append(fname)
            self.uncertainty_files.sort()

        # specify image transformations
        self.img_to_tensor = transforms.ToTensor()
        self.transformations = transformations

    def __getitem__(self, idx: int) -> Dict:
        path_to_current_img = os.path.join(self.path_to_images, self.image_files[idx])
        img_pil = Image.open(path_to_current_img)
        img = self.img_to_tensor(img_pil)

        path_to_current_anno = os.path.join(self.path_to_annos, self.anno_files[idx])
        anno = self.get_anno(path_to_current_anno)

        anno_mask = torch.ones((1, anno.size(1), anno.size(2))).bool()
        if self.load_anno_masks:
            path_to_current_anno_mask = os.path.join(self.path_to_anno_masks, self.anno_mask_files[idx])
            anno_mask = self.get_anno_mask(path_to_current_anno_mask)

        anno_mask_sum = torch.sum(anno_mask)

        uncertainty = torch.zeros((1, anno.size(1), anno.size(2)))
        if self.load_uncertainties:
            if len(self.uncertainty_files) > idx:
                path_to_current_uncertainty = os.path.join(self.path_to_uncertainties, self.uncertainty_files[idx])
                if os.path.exists(path_to_current_uncertainty):
                    uncertainty = self.get_uncertainty(path_to_current_uncertainty)

        # apply a set of transformations to the raw_image, image and anno
        for transformer in self.transformations:
            img_pil, img, anno, anno_mask, uncertainty = transformer(img_pil, img, anno, anno_mask, uncertainty)

        if self.task == "classification":
            anno = self.remap_annotation(anno.numpy())

        anno_mask = anno_mask.squeeze(dim=0)
        anno = self.apply_anno_mask(anno, anno_mask)

        return {
            "data": img,
            "image": img,
            "anno": anno,
            "anno_mask_sum": anno_mask_sum,
            "uncertainty": uncertainty,
            "index": idx,
        }

    @staticmethod
    def get_anno_mask(path_to_current_anno_mask: str) -> torch.Tensor:
        anno_mask = np.load(path_to_current_anno_mask)
        return torch.from_numpy(anno_mask).unsqueeze(dim=0).bool()

    def __len__(self) -> int:
        return len(self.image_files)

    def get_anno(self, path_to_current_anno: str) -> torch.Tensor:
        if self.task == "classification":
            anno = cv2.imread(path_to_current_anno)
            anno = anno.astype(np.int64)  # torch does not support conversion of uint16
            anno = np.moveaxis(anno, -1, 0)  # now in CHW mode
            return torch.from_numpy(anno).long()
        elif self.task == "regression":
            anno = cv2.imread(path_to_current_anno, cv2.IMREAD_GRAYSCALE)  # 0 to 255 value range
            anno = anno.astype(np.float32) / 255  # torch does not support conversion of uint16, now in HW mode
            anno = (
                self.cfg["model"]["value_range"]["min_value"]
                + (self.cfg["model"]["value_range"]["max_value"] - self.cfg["model"]["value_range"]["min_value"]) * anno
            )  # normalize targets to regression value range
            return torch.from_numpy(anno).float().unsqueeze(dim=0)
        else:
            raise NotImplementedError(f"{self.task} task is not implemented for Flightmare dataset!")

    @staticmethod
    def get_uncertainty(path_to_current_uncertainty: str) -> torch.Tensor:
        uncertainty = cv2.imread(path_to_current_uncertainty, cv2.IMREAD_GRAYSCALE)  # 0 to 255 value range
        uncertainty = uncertainty.astype(np.float32) / 255  # normalize uncertainties between 0 and 1
        return torch.from_numpy(uncertainty).float().unsqueeze(dim=0)

    @staticmethod
    def remap_annotation(anno: np.array) -> torch.Tensor:
        """
        After remapping the annotations have the labels:

        Flightmare:
            - 0 := Background: [0, 0, 0] and any non-defined color
            - 1 := Floor: [2, 73, 9]
            - 2 := Hangar: [32, 73, 65]
            - 3 := Fence: [6, 73, 72]
            - 4 := Road: [36, 73, 8]
            - 5 := Tank: [2, 73, 128]
            - 6 := Pipe: [32, 9, 201]
            - 7 := Container: [6, 9, 193]
            - 8 := Misc: [36, 9, 129]
            - 9 := Boundary: Any other color
        """

        dims = anno.shape
        assert len(dims) == 3, "wrong matrix dimension!!!"
        assert dims[0] == 3, "annotation must have 3 channels!!!"

        flightmare_labels = LABELS["flightmare"]
        remapped_anno = np.ones((dims[1], dims[2])) * flightmare_labels["boundary"]["id"]

        for label_key, label_info in flightmare_labels.items():
            if label_key == "boundary":
                continue

            label_color = np.flip(np.array(label_info["color"])).reshape((3, 1, 1))
            remapped_anno[(anno == label_color).all(axis=0)] = label_info["id"]

        return torch.from_numpy(remapped_anno).long()

    @staticmethod
    def apply_anno_mask(anno: torch.Tensor, anno_mask: torch.Tensor) -> torch.Tensor:
        anno[~anno_mask] = LABELS["flightmare"]["boundary"]["id"]
        return anno.long()
