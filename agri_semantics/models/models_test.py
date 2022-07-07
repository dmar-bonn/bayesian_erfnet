import os
import torch
import torch.nn as nn
import unittest

from . import get_model, get_net


class TestERFNet(unittest.TestCase):
    @classmethod
    def setUpConfig(cls) -> dict:
        cfg = {}
        cfg["cfg_1"] = {"model": {"name": "erfnet", "num_classes": 2, "in_channels": 3}}

        cfg["cfg_2"] = {"model": {"name": "erfnet", "num_classes": 2, "in_channels": 1}}

        cfg["fake_cfg_1"] = {"model": {"name": "erfnet", "num_classes": 0, "in_channels": 3}}

        cfg["fake_cfg_2"] = {"model": {"name": "erfnet", "num_classes": 0.5, "in_channels": 3}}

        return cfg

    def test_erfnet_init(self):
        cfg = self.setUpConfig()

        model = get_net(cfg["cfg_1"])
        self.assertTrue(isinstance(model, nn.Module))
        with self.assertRaises(RuntimeError):
            get_net(cfg["fake_cfg_1"])
        with self.assertRaises(RuntimeError):
            get_net(cfg["fake_cfg_2"])

        model = get_model(cfg["cfg_1"])
        self.assertTrue(isinstance(model, nn.Module))
        with self.assertRaises(IndexError):
            get_model(cfg["fake_cfg_1"])
        with self.assertRaises(ValueError):
            get_model(cfg["fake_cfg_2"])

    def test_erfnet_forward(self):
        cfg = self.setUpConfig()

        # cfg 1
        model = get_net(cfg["cfg_1"])

        tensor = torch.rand(1, 3, 32, 96)
        out = model(tensor)
        self.assertTrue(out.shape[2:] == tensor.shape[2:])
        self.assertTrue(out.shape[1] == cfg["cfg_1"]["model"]["num_classes"])

        tensor = torch.rand(1, 3, 64, 64)
        out = model(tensor)
        self.assertTrue(out.shape[2:] == tensor.shape[2:])
        self.assertTrue(out.shape[1] == cfg["cfg_1"]["model"]["num_classes"])

        tensor = torch.rand(1, 1, 64, 64)
        with self.assertRaises(RuntimeError):
            model(tensor)

        tensor = torch.rand(1, 1, 32, 97)
        with self.assertRaises(RuntimeError):
            model(tensor)

        # cfg 2
        model = get_net(cfg["cfg_2"])

        tensor = torch.rand(1, 1, 32, 96)
        out = model(tensor)
        self.assertTrue(out.shape[2:] == tensor.shape[2:])
        self.assertTrue(out.shape[1] == cfg["cfg_2"]["model"]["num_classes"])

        tensor = torch.rand(1, 1, 64, 64)
        out = model(tensor)
        self.assertTrue(out.shape[2:] == tensor.shape[2:])
        self.assertTrue(out.shape[1] == cfg["cfg_2"]["model"]["num_classes"])

        tensor = torch.rand(1, 3, 64, 64)
        with self.assertRaises(RuntimeError):
            model(tensor)

        tensor = torch.rand(1, 1, 32, 97)
        with self.assertRaises(RuntimeError):
            model(tensor)


if __name__ == "__main__":
    unittest.main()
