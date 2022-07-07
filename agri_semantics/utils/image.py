""" Auxiliary functions to modify images.
"""
from typing import Optional

import torch
import torchvision.transforms.functional as TF


def interpolation_modes_from_int(i: int) -> TF.InterpolationMode:
    """Specify the desired interpolation method to resize images.

    Args:
        i (int): specify a mode by the corresponding integer

    Returns:
        TF.InterpolationMode: interpolation mode
    """
    assert isinstance(i, int)
    assert i >= 0
    assert i <= 3

    inverse_modes_mapping = {
        0: TF.InterpolationMode.NEAREST,
        1: TF.InterpolationMode.BILINEAR,
        2: TF.InterpolationMode.BICUBIC,
    }
    return inverse_modes_mapping[i]


def resize(
    image: torch.Tensor,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_aspect_ratio: bool = False,
    interpolation: int = 0,
) -> torch.Tensor:
    """Resize image dimension.

    Args:
        image (torch.Tensor): input image
        width (Optional[int], optional): new width dimension. Defaults to None.
        height (Optional[int], optional): new height dimension. Defaults to None.
        keep_aspect_ratio (bool, optional): specify if aspect ratio should stay the same. Defaults to False.
        interpolation (int, optional): interpolation mode. Defaults to 0 (:= nearest neighboor).

    Returns:
        torch.Tensor: resized image
    """
    if width is not None:
        if not isinstance(width, int):
            raise ValueError("width must be of type int")

    if height is not None:
        if not isinstance(height, int):
            raise ValueError("height must be of type int")

    if (width is None) and (height is None):
        return image

    # original image dimension
    if len(image.shape) == 2:
        image = torch.unsqueeze(image, 0)

    h, w = image.shape[1], image.shape[2]
    aspect_ratio = w / h
    interpolation_mode = interpolation_modes_from_int(interpolation)

    # new heigth dimension is provided
    if (isinstance(height, int)) and (width is None):
        assert height > 0

        if keep_aspect_ratio:
            width_new = int(round(aspect_ratio * height))
        else:
            width_new = w

        image = TF.resize(image, [height, width_new], interpolation=interpolation_mode)
        image = torch.squeeze(image)

        return image

    # new width dimension is provided
    if (isinstance(width, int)) and (height is None):
        assert width > 0

        if keep_aspect_ratio:
            height_new = int(round(width * (1 / aspect_ratio)))
        else:
            height_new = h

        image = TF.resize(image, [height_new, width], interpolation=interpolation_mode)
        image = torch.squeeze(image)

        return image

    # new width and height dimensions are provided
    if (isinstance(width, int)) and (isinstance(height, int)):
        assert (height > 0) or (width > 0)
        if keep_aspect_ratio:
            raise ValueError(
                "In case width and height are changed the aspect ratio might change. Set 'keep_aspect_ratio' to False to resolve this issue."
            )

        image = TF.resize(image, [height, width], interpolation=interpolation_mode)
        image = torch.squeeze(image)

        return image
