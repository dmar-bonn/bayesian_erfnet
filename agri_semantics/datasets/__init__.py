from agri_semantics.datasets.CityscapesDataset import CityscapesDataModule
from agri_semantics.datasets.RIT18Dataset import RIT18DataModule
from agri_semantics.datasets.UAVDataset import UAVDataModule
from agri_semantics.datasets.WeedmapDataset import WeedmapDataModule
from agri_semantics.datasets.flightmare import FlightmareDataModule
from agri_semantics.datasets.potsdam import PotsdamDataModule
from pytorch_lightning import LightningDataModule


def get_data_module(cfg) -> LightningDataModule:
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
        return PotsdamDataModule(cfg)
    elif name == "flightmare":
        return FlightmareDataModule(cfg)
    else:
        raise ValueError(f"Dataset '{name}' not found!")
