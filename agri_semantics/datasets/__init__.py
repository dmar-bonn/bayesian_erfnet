from typing import Dict

from agri_semantics.datasets.CityscapesDataset import CityscapesDataModule
from agri_semantics.datasets.RIT18Dataset import RIT18DataModule
from agri_semantics.datasets.UAVDataset import UAVDataModule
from agri_semantics.datasets.WeedmapDataset import WeedmapDataModule
from agri_semantics.datasets.flightmare import FlightmareDataModule
from agri_semantics.datasets.potsdam import PotsdamDataModule
from pytorch_lightning import LightningDataModule


def get_data_module(cfg: Dict, human_data: bool = True, pseudo_data: bool = False) -> LightningDataModule:
    name = cfg["data"]["name"]
    if name == "uav":
        return UAVDataModule(cfg)
    elif name == "weedmap":
        return WeedmapDataModule(cfg)
    elif name == "cityscapes":
        return CityscapesDataModule(cfg)
    elif name == "rit18":
        return RIT18DataModule(cfg)
    elif name == "potsdam":
        return PotsdamDataModule(cfg, human_data, pseudo_data)
    elif name == "flightmare":
        return FlightmareDataModule(cfg, human_data, pseudo_data)
    else:
        raise ValueError(f"Dataset '{name}' not found!")
