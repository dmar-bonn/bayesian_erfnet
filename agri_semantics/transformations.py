""" Define a set of transformations which can be applied simultaneously to the raw image, input image and its corresponding anntations.

This is relevant for the task of semantic segmentation since the input image and its annotation need to be treated in the same way.
"""
import math
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from PIL.Image import Image as PILImage
import torch
import torchvision
import torchvision.transforms.functional as TF


class Transformation(ABC):
    """General transformation which can be applied simultaneously to the raw image, input image and its corresponding anntations."""

    @abstractmethod
    def __call__(
        self, raw_image: PILImage, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[PILImage, torch.Tensor, torch.Tensor]:
        """Apply a transformation to a given image and its corresponding annotation.

        Args:
          image (torch.Tensor): input image to be transformed.
          anno (torch.Tensor): annotation to be transformed.

        Returns:
          Tuple[torch.Tensor, torch.Tensor]: transformed image and its corresponding annotation
        """
        raise NotImplementedError


class MyCenterCropTransform(Transformation):
    """Extract a patch from the image center."""

    def __init__(self, crop_height: Optional[int] = None, crop_width: Optional[int] = None):
        """Set height and width of cropping region.

        Args:
            crop_height (Optional[int], optional): Height of cropping region. Defaults to None.
            crop_width (Optional[int], optional): Width of cropping region. Defaults to None.
        """
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(
        self, raw_image: PILImage, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[PILImage, torch.Tensor, torch.Tensor]:
        # dimension of each input should be identical
        assert raw_image.height == image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
        assert raw_image.width == image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

        if (self.crop_height is None) or (self.crop_width is None):
            return raw_image, image, anno

        img_chans, img_height, img_width = image.shape[:3]
        anno_chans = anno.shape[0]

        if self.crop_width > img_width:
            raise ValueError("Width of cropping region must not be greather than img width")
        if self.crop_height > img_height:
            raise ValueError("Height of cropping region must not be greather than img height.")

        raw_image_cropped: PILImage = TF.center_crop(raw_image, [self.crop_height, self.crop_width])  # type: ignore
        image_cropped = TF.center_crop(image, [self.crop_height, self.crop_width])
        anno_cropped = TF.center_crop(anno, [self.crop_height, self.crop_width])

        assert raw_image_cropped.height == self.crop_height, "Cropped raw image has not the desired size."
        assert raw_image_cropped.width == self.crop_width, "Cropped raw image has not the desired width."

        assert image_cropped.shape[0] == img_chans, "Cropped image has an unexpected number of channels."
        assert image_cropped.shape[1] == self.crop_height, "Cropped image has not the desired size."
        assert image_cropped.shape[2] == self.crop_width, "Cropped image has not the desired width."

        assert anno_cropped.shape[0] == anno_chans, "Cropped anno has an unexpected number of channels."
        assert anno_cropped.shape[1] == self.crop_height, "Cropped anno has not the desired size."
        assert anno_cropped.shape[2] == self.crop_width, "Cropped anno has not the desired width."

        return raw_image_cropped, image_cropped, anno_cropped


class MyRandomCropTransform(Transformation):
    """Extract a random patch from a given image and its corresponding annnotation."""

    def __init__(self, crop_height: Optional[int] = None, crop_width: Optional[int] = None):
        """Set height and width of cropping region.

        Args:
            crop_height (Optional[int], optional): Height of cropping region. Defaults to None.
            crop_width (Optional[int], optional): Width of cropping region. Defaults to None.
        """
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(
        self, raw_image: PILImage, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[PILImage, torch.Tensor, torch.Tensor]:
        """Apply cropping to an image and its corresponding annotation.

        Args:
            image (torch.Tensor): image to be cropped of shape [C x H x W]
            anno (torch.Tensor): annotation to be cropped of shape [1 x H x W]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cropped image, cropped anno
        """
        # dimension of each input should be identical
        assert raw_image.height == image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
        assert raw_image.width == image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

        if (self.crop_height is None) or (self.crop_width is None):
            return raw_image, image, anno

        img_chans, img_height, img_width = image.shape[:3]
        anno_chans = anno.shape[0]

        if self.crop_width > img_width:
            raise ValueError("Width of cropping region must not be greather than img width")
        if self.crop_height > img_height:
            raise ValueError("Height of cropping region must not be greather than img height.")

        max_x = img_width - self.crop_width
        x_start = random.randint(0, max_x)

        max_y = img_height - self.crop_height
        y_start = random.randint(0, max_y)

        assert (x_start + self.crop_width) <= img_width, "Cropping region (width) exceeds image dims."
        assert (y_start + self.crop_height) <= img_height, "Cropping region (height) exceeds image dims."

        raw_image_cropped: PILImage = TF.crop(raw_image, y_start, x_start, self.crop_height, self.crop_width)  # type: ignore
        image_cropped = TF.crop(image, y_start, x_start, self.crop_height, self.crop_width)
        anno_cropped = TF.crop(anno, y_start, x_start, self.crop_height, self.crop_width)

        assert raw_image_cropped.height == self.crop_height, "Cropped raw image has not the desired size."
        assert raw_image_cropped.width == self.crop_width, "Cropped raw image has not the desired width."

        assert image_cropped.shape[0] == img_chans, "Cropped image has an unexpected number of channels."
        assert image_cropped.shape[1] == self.crop_height, "Cropped image has not the desired size."
        assert image_cropped.shape[2] == self.crop_width, "Cropped image has not the desired width."

        assert anno_cropped.shape[0] == anno_chans, "Cropped anno has an unexpected number of channels."
        assert anno_cropped.shape[1] == self.crop_height, "Cropped anno has not the desired size."
        assert anno_cropped.shape[2] == self.crop_width, "Cropped anno has not the desired width."

        return raw_image_cropped, image_cropped, anno_cropped


class MyResizeTransform(Transformation):
    """Resize a given image and its corresponding annotation."""

    def __init__(self, height: Optional[int] = None, width: Optional[int] = None, keep_aspect_ratio: bool = False):
        """Set params for resize operation.

        Args:
            width (Optional[int], optional): New width dimension. Defaults to None.
            height (Optional[int], optional): New height dimension. Defaults to None.
            keep_aspect_ratio (bool, optional): Specify if aspect ratio should stay the same. Defaults to False.
        """
        if width is not None:
            if not isinstance(width, int):
                raise ValueError("width must be of type int")
        self.resized_width = width

        if height is not None:
            if not isinstance(height, int):
                raise ValueError("height must be of type int")
        self.resized_height = height

        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(
        self, raw_image: PILImage, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[PILImage, torch.Tensor, torch.Tensor]:
        """Apply resizing to an image and its corresponding annotation.

        Args:
            image (torch.Tensor): image to be resized of shape [C x H x W]
            anno (torch.Tensor): anno to be cropped of shape [C x H x W]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [description]
        """
        # dimension of each input should be identical
        assert raw_image.height == image.shape[1] == anno.shape[1], "Dimensions of all input should be identical."
        assert raw_image.width == image.shape[2] == anno.shape[2], "Dimensions of all input should be identical."

        if (self.resized_width is None) and (self.resized_height is None):
            return raw_image, image, anno

        # original image dimension
        h, w = image.shape[1], image.shape[2]
        aspect_ratio = w / h

        # 1st case - user provides height but not width
        if (self.resized_height is not None) and (self.resized_width is None):
            assert self.resized_height > 0

            if self.keep_aspect_ratio:
                resized_width = int(round(aspect_ratio * self.resized_height))
            else:
                resized_width = w

            raw_image_resized: PILImage = TF.resize(raw_image, [self.resized_height, resized_width], interpolation=TF.InterpolationMode.BILINEAR)  # type: ignore
            image_resized = TF.resize(
                image, [self.resized_height, resized_width], interpolation=TF.InterpolationMode.BILINEAR
            )
            anno_resized = TF.resize(
                anno, [self.resized_height, resized_width], interpolation=TF.InterpolationMode.NEAREST
            )
            del resized_width

            return raw_image_resized, image_resized, anno_resized

        # 2nd case - user provides width but not height
        if (self.resized_width is not None) and (self.resized_height is None):
            assert self.resized_width > 0

            if self.keep_aspect_ratio:
                resized_height = int(round(self.resized_width * (1 / aspect_ratio)))
            else:
                resized_height = h

            raw_image_resized: PILImage = TF.resize(raw_image, [resized_height, self.resized_width], interpolation=TF.InterpolationMode.BILINEAR)  # type: ignore
            image_resized = TF.resize(
                image, [resized_height, self.resized_width], interpolation=TF.InterpolationMode.BILINEAR
            )
            anno_resized = TF.resize(
                anno, [resized_height, self.resized_width], interpolation=TF.InterpolationMode.NEAREST
            )
            del resized_height

            return raw_image_resized, image_resized, anno_resized
        # 3rd case - user provides width and height
        if (self.resized_width is not None) and (self.resized_height is not None):
            assert (self.resized_width > 0) or (self.resized_height > 0)

        if self.keep_aspect_ratio:
            raise ValueError(
                "In case width and height are changed the aspect ratio might change. Set 'keep_aspect_ratio' to False to resolve this issue."
            )

        raw_image_resized: PILImage = TF.resize(raw_image, [self.resized_height, self.resized_width], interpolation=TF.InterpolationMode.BILINEAR)  # type: ignore
        image_resized = TF.resize(image, [self.resized_height, self.resized_width], interpolation=TF.InterpolationMode.BILINEAR)  # type: ignore
        anno_resized = TF.resize(anno, [self.resized_height, self.resized_width], interpolation=TF.InterpolationMode.NEAREST)  # type: ignore

        return raw_image_resized, image_resized, anno_resized


class MyRandomRotationTransform(Transformation):
    """Rotate a given image and its corresponding annotation."""

    def __init__(
        self, min_angle: Optional[float] = None, max_angle: Optional[float] = None, step_size: Optional[float] = None
    ):
        self.min_angle = min_angle  # degree
        self.max_angle = max_angle  # degree
        self.step_size = step_size  # degree

    def __call__(
        self, raw_image: PILImage, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[PILImage, torch.Tensor, torch.Tensor]:
        if (self.min_angle is None) or (self.max_angle is None) or (self.step_size is None):
            return raw_image, image, anno

        assert self.min_angle is not None
        assert self.max_angle is not None
        assert self.step_size is not None

        assert self.min_angle < self.max_angle
        assert self.step_size > 0

        angles = torch.arange(self.min_angle, self.max_angle, step=self.step_size)
        random_angle = float(random.choice(list(angles)))  # degree

        raw_image_rotated: PILImage = TF.rotate(raw_image, random_angle, interpolation=TF.InterpolationMode.BILINEAR)  # type: ignore
        image_rotated = TF.rotate(image, random_angle, interpolation=TF.InterpolationMode.BILINEAR)
        anno_rotated = TF.rotate(anno, random_angle, interpolation=TF.InterpolationMode.NEAREST)

        return raw_image_rotated, image_rotated, anno_rotated


class MyRandomColorJitterTransform(Transformation):
    """Apply colour jitter to a given image."""

    def __init__(
        self,
        brightness: Optional[float] = 0.0,
        contrast: Optional[int] = 0.0,
        saturation: Optional[int] = 0.0,
        hue: Optional[int] = 0.0,
    ):
        """Set colour jitter parameters.
        See: https://pytorch.org/vision/stable/transforms.html
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(
        self, raw_image: PILImage, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[PILImage, torch.Tensor, torch.Tensor]:

        jitter = torchvision.transforms.ColorJitter(
            brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue
        )
        raw_image_jitted = jitter(raw_image)
        image_jitted = jitter(image)

        return raw_image_jitted, image_jitted, anno


def get_transformations(cfg, stage: str) -> List[Transformation]:
    assert stage in ["train", "val", "test"]
    transformations = []

    try:
        if cfg[stage]["transformations"] is None:
            return transformations

        for tf_name in cfg[stage]["transformations"].keys():
            if tf_name == "random_rotation":
                min_angle = cfg[stage]["transformations"][tf_name]["min_angle"]
                max_angle = cfg[stage]["transformations"][tf_name]["max_angle"]
                step_size = cfg[stage]["transformations"][tf_name]["step_size"]
                transformer = MyRandomRotationTransform(min_angle, max_angle, step_size)

                transformations.append(transformer)

            if tf_name == "center_crop":
                crop_height = cfg[stage]["transformations"][tf_name]["height"]
                crop_width = cfg[stage]["transformations"][tf_name]["width"]
                transformer = MyCenterCropTransform(crop_height, crop_width)

                transformations.append(transformer)

            if tf_name == "random_crop":
                crop_height = cfg[stage]["transformations"][tf_name]["height"]
                crop_width = cfg[stage]["transformations"][tf_name]["width"]
                transformer = MyRandomCropTransform(crop_height, crop_width)

                transformations.append(transformer)

            if tf_name == "resize":
                resize_height = cfg[stage]["transformations"][tf_name]["height"]
                resize_width = cfg[stage]["transformations"][tf_name]["width"]
                keep_ap = cfg[stage]["transformations"][tf_name]["keep_aspect_ratio"]
                transformer = MyResizeTransform(resize_height, resize_width, keep_ap)

                transformations.append(transformer)

            if tf_name == "color_jitter":
                brightness = cfg[stage]["transformations"][tf_name]["brightness"]
                contrast = cfg[stage]["transformations"][tf_name]["contrast"]
                saturation = cfg[stage]["transformations"][tf_name]["saturation"]
                hue = cfg[stage]["transformations"][tf_name]["hue"]
                transformer = MyRandomColorJitterTransform(brightness, contrast, saturation, hue)

                transformations.append(transformer)

    except KeyError:
        return transformations

    return transformations
