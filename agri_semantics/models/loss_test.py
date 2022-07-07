""" Test different loss functions.
"""
import unittest
import torch
from . import get_criterion


class TestCrossEntropy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = {"model": {"loss": "xentropy"}}

    def test_xentropy(self):
        criterion = get_criterion(TestCrossEntropy.cfg)

        inputs = torch.rand((16, 9, 64, 64))
        targets = torch.randint(0, 9, (16, 64, 64))

        loss = criterion(inputs, targets)
        self.assertTrue(loss > 0)


if __name__ == "__main__":
    unittest.main()
