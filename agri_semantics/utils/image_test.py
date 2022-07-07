""" Perform tests for auxiliary image functions.
"""
import pdb
import unittest

import torch
import torchvision.transforms.functional as TF

from .image import interpolation_modes_from_int, resize


class TestImageFuncs(unittest.TestCase):
    def test_interpolation_modes_from_int(self):
        # nearest neighbor
        mode = interpolation_modes_from_int(0)
        self.assertTrue(mode.value == "nearest")

        # bilinear
        mode = interpolation_modes_from_int(1)
        self.assertTrue(mode.value == "bilinear")

        # bicubic
        mode = interpolation_modes_from_int(2)
        self.assertTrue(mode.value == "bicubic")

    def test_resize(self):
        image = torch.randn((1, 64, 128))

        resized = resize(image, interpolation=0)
        self.assertTrue(resized.shape[1] == 64)  # check height
        self.assertTrue(resized.shape[2] == 128)  # check width
        self.assertTrue(image.dtype == resized.dtype)

        resized = resize(image, width=32, interpolation=0)
        self.assertTrue(resized.shape[1] == 64)
        self.assertTrue(resized.shape[2] == 32)
        self.assertTrue(image.dtype == resized.dtype)

        resized = resize(image, width=32, keep_aspect_ratio=True)
        self.assertTrue(resized.shape[1] == 16)
        self.assertTrue(resized.shape[2] == 32)
        self.assertTrue(image.dtype == resized.dtype)

        resized = resize(image, width=32, keep_aspect_ratio=True, interpolation=1)
        self.assertTrue(resized.shape[1] == 16)
        self.assertTrue(resized.shape[2] == 32)
        self.assertTrue(image.dtype == resized.dtype)

        resized = resize(image, height=32)
        self.assertTrue(resized.shape[1] == 32)
        self.assertTrue(resized.shape[2] == 128)
        self.assertTrue(image.dtype == resized.dtype)

        resized = resize(image, height=32, keep_aspect_ratio=True)
        self.assertTrue(resized.shape[1] == 32)
        self.assertTrue(resized.shape[2] == 64)
        self.assertTrue(image.dtype == resized.dtype)

        resized = resize(image, height=32, keep_aspect_ratio=True, interpolation=1)
        self.assertTrue(resized.shape[1] == 32)
        self.assertTrue(resized.shape[2] == 64)
        self.assertTrue(image.dtype == resized.dtype)

        resized = resize(image, width=96, height=16)
        self.assertTrue(resized.shape[1] == 16)
        self.assertTrue(resized.shape[2] == 96)
        self.assertTrue(image.dtype == resized.dtype)

        with self.assertRaises(ValueError):
            resized = resize(image, width=96, height=16, keep_aspect_ratio=True)
            resized = resize(image, width=96.1, height=16, keep_aspect_ratio=False)

        image = torch.randint(0, 128, (1, 64, 128), dtype=torch.int64)
        resized = resize(image, interpolation=0)
        self.assertTrue(resized.shape[1] == 64)  # check height
        self.assertTrue(resized.shape[2] == 128)  # check width
        self.assertTrue(image.dtype == resized.dtype)


if __name__ == "__main__":
    unittest.main()
