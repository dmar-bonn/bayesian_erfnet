""" Perform tests for the UAVDataModule.
"""

import unittest

import torch

from . import get_data_module


class TestUAVDataModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = {
            "data": {
                "name": "UAVDataModule",
                "path_to_dataset": "./samples/datasets/UAVbonn2017-sample",
                "img_prop": {"width": None, "height": None, "keep_aspect_ratio": False},
                "batch_size": 1,
                "train_shuffle": True,
                "num_workers": 1,
            }
        }

        cls.fake_cfg = {
            "data": {
                "name": "UAVDataModule",
                "path_to_dataset": "./path/to/invalid/dir",
                "img_prop": {"width": None, "height": None, "keep_aspect_ratio": False},
                "batch_size": 1,
                "train_shuffle": True,
                "num_workers": 1,
            }
        }

    def test_loaders(self):
        uav_dm = get_data_module(TestUAVDataModule.cfg)
        uav_dm.setup()

        self.assertTrue(len(uav_dm._uav_train) == 1)
        self.assertTrue(len(uav_dm._uav_val) == 1)
        self.assertTrue(len(uav_dm._uav_test) == 2)

        self.assertTrue(uav_dm._uav_train.image_files[0] == "sugar_f1_170912_01_subImages_2_frame20_crop1.png")
        self.assertTrue(uav_dm._uav_train.anno_files[0] == "sugar_f1_170912_01_subImages_2_frame20_crop1.png")

        self.assertTrue(uav_dm._uav_val.image_files[0] == "sugar_f1_170912_01_subImages_2_frame20_crop2.png")
        self.assertTrue(uav_dm._uav_val.anno_files[0] == "sugar_f1_170912_01_subImages_2_frame20_crop2.png")

        self.assertTrue(uav_dm._uav_test.image_files[0] == "sugar_f1_170912_01_subImages_2_frame20_crop1.png")
        self.assertTrue(uav_dm._uav_test.anno_files[0] == "sugar_f1_170912_01_subImages_2_frame20_crop1.png")
        self.assertTrue(uav_dm._uav_test.image_files[1] == "sugar_f1_170912_01_subImages_2_frame20_crop2.png")
        self.assertTrue(uav_dm._uav_test.anno_files[1] == "sugar_f1_170912_01_subImages_2_frame20_crop2.png")

        uav_fake_dm = get_data_module(TestUAVDataModule.fake_cfg)
        with self.assertRaises(FileNotFoundError):
            uav_fake_dm.setup()

    def test_getitem(self):
        uav_dm = get_data_module(TestUAVDataModule.cfg)
        uav_dm.setup()

        train_loader = uav_dm.train_dataloader()
        for batch in train_loader:
            batch_dict = batch
            img, anno = batch_dict["data"], batch_dict["anno"]

            self.assertTrue(img.shape[0] == TestUAVDataModule.cfg["data"]["batch_size"])
            self.assertTrue(anno.shape[0] == TestUAVDataModule.cfg["data"]["batch_size"])

            self.assertTrue(torch.max(img[0]) <= 1.0)
            self.assertTrue(torch.min(img[0]) >= 0.0)

            self.assertTrue(torch.max(anno[0]) == 2)
            self.assertTrue(torch.min(anno[0]) == 0)

            for index, label in enumerate(torch.unique(anno[0])):
                self.assertTrue(index == label)


if __name__ == "__main__":
    unittest.main()
