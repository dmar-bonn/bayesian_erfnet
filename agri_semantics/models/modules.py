from typing import Tuple

import torch
import torch.nn as nn
from agri_semantics.models import blocks


##############################################################################################
#                                                                                            #
#  Pytorch Lightning ERFNet implementation from Jan Weyler. Our Bayesian-ERFNet              #
#  implementation builds upon Jan's ERFNet implementation.                                   #
#                                                                                            #
##############################################################################################


class AleatoricERFNetModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        dropout_prob: float,
        use_shared_decoder: bool = False,
        deep_encoder: bool = False,
        epistemic_version: str = "standard",
        output_fn: callable = None,
    ):
        super(AleatoricERFNetModel, self).__init__()

        self.output_fn = output_fn
        self.num_classes = num_classes
        self.use_shared_decoder = use_shared_decoder

        self.encoder = DropoutERFNetEncoder(in_channels, dropout_prob, epistemic_version)
        if deep_encoder:
            self.encoder = DropoutERFNetDeepEncoder(in_channels, dropout_prob)

        self.segmentation_decoder = DropoutERFNetDecoder(self.num_classes)
        self.aleatoric_uncertainty_decoder = ERFNetAleatoricUncertaintyDecoder(self.num_classes)
        self.shared_decoder = ERFNetAleatoricSharedDecoder(self.num_classes, dropout_prob, epistemic_version)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_enc = self.encoder(x)

        if self.use_shared_decoder:
            output_seg, output_std = self.shared_decoder(output_enc)
        else:
            output_seg = self.segmentation_decoder(output_enc)
            output_std = self.aleatoric_uncertainty_decoder(output_enc)

        if self.output_fn is not None:
            output_seg = self.output_fn(output_seg)

        return output_seg, output_std, output_enc


class DropoutERFNetEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout_prob: float = 0.3, epistemic_version: str = "standard"):
        super().__init__()
        self.initial_block = blocks.DownsamplerBlock(in_channels, 16)

        self.layers = nn.ModuleList()

        self.layers.append(blocks.DownsamplerBlock(16, 64))

        dropout_prob_1, dropout_prob_2, dropout_prob_3 = self.get_dropout_probs(dropout_prob, epistemic_version)

        for x in range(0, 5):  # 5 times
            self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))

        self.layers.append(blocks.DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            dropout_prob_tmp = dropout_prob_2 if x == 0 else dropout_prob_3
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 2))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 4))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 8))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 16))

    @staticmethod
    def get_dropout_probs(dropout_prob: float, epistemic_version: str) -> Tuple[float, float, float]:
        if epistemic_version == "all":
            return dropout_prob, dropout_prob, dropout_prob
        elif epistemic_version == "center":
            return 0, 0, dropout_prob
        elif epistemic_version == "classifier":
            return 0, 0, 0
        elif epistemic_version == "standard":
            return dropout_prob / 10, dropout_prob, dropout_prob
        else:
            raise ValueError(f"Epistemic version '{epistemic_version}' unknown!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.initial_block(x)

        for layer in self.layers:
            output = layer(output)

        return output


class DropoutERFNetDeepEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout_prob: float = 0.3):
        super().__init__()
        self.initial_block = blocks.DownsamplerBlock(in_channels, 16)

        self.layers = nn.ModuleList()

        self.layers.append(blocks.DownsamplerBlock(16, 64))

        for x in range(0, 10):  # 10 times
            self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob / 10, 1))

        self.layers.append(blocks.DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 3 times
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob, 2))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob, 4))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob, 8))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob, 16))

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class DropoutERFNetDecoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(blocks.UpsamplerBlock(128, 64))
        self.layers.append(blocks.non_bottleneck_1d(64, 0, 1))
        self.layers.append(blocks.non_bottleneck_1d(64, 0, 1))

        self.layers.append(blocks.UpsamplerBlock(64, 16))
        self.layers.append(blocks.non_bottleneck_1d(16, 0, 1))
        self.layers.append(blocks.non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class ERFNetAleatoricSharedDecoder(nn.Module):
    def __init__(self, num_classes: int, dropout_prob: float = 0.0, epistemic_version: str = "standard"):
        super().__init__()

        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        dropout_prob_1, dropout_prob_2, dropout_prob_3 = self.get_dropout_probs(dropout_prob, epistemic_version)

        self.layers.append(blocks.UpsamplerBlock(128, 64))
        self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))
        self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))

        self.layers.append(blocks.UpsamplerBlock(64, 16))
        self.layers.append(blocks.non_bottleneck_1d(16, dropout_prob_2, 1))
        self.layers.append(blocks.non_bottleneck_1d(16, dropout_prob_3, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes + 1, 2, stride=2, padding=0, output_padding=0, bias=True)
        self.output_std_fn = nn.Softplus(beta=1)

    @staticmethod
    def get_dropout_probs(dropout_prob: float, epistemic_version: str) -> Tuple[float, float, float]:
        if epistemic_version == "all":
            return dropout_prob, dropout_prob, dropout_prob
        elif epistemic_version == "center":
            return dropout_prob, 0, 0
        elif epistemic_version == "classifier":
            return 0, 0, dropout_prob
        elif epistemic_version == "standard":
            return 0, 0, 0
        else:
            raise ValueError(f"Epistemic version '{epistemic_version}' unknown!")

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = input

        for layer in self.layers:
            output = layer(output)

        output_seg, output_std = self.output_conv(output).split(self.num_classes, 1)
        output_std = self.output_std_fn(output_std) + 10 ** (-8)

        return output_seg, output_std


class ERFNetAleatoricUncertaintyDecoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        self.layers.append(blocks.UpsamplerBlock(128, 64))
        self.layers.append(blocks.non_bottleneck_1d(64, 0, 1))
        self.layers.append(blocks.non_bottleneck_1d(64, 0, 1))

        self.layers.append(blocks.UpsamplerBlock(64, 16))
        self.layers.append(blocks.non_bottleneck_1d(16, 0, 1))
        self.layers.append(blocks.non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, 1, 2, stride=2, padding=0, output_padding=0, bias=True)
        self.output_std_fn = nn.Softplus(beta=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)
        output = self.output_std_fn(output) + 10 ** (-8)
        return output


class ERFNetModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        dropout_prop: float = 0.0,
        deep_encoder: bool = False,
        epistemic_version: str = "standard",
        output_fn: callable = None,
    ):
        super(ERFNetModel, self).__init__()

        self.output_fn = output_fn
        self.encoder = ERFNetEncoder(in_channels, dropout_prop, epistemic_version)
        if deep_encoder:
            self.encoder = ERFNetDeepEncoder(in_channels, dropout_prop)

        self.decoder = ERFNetDecoder(num_classes, dropout_prop, epistemic_version)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output_enc = self.encoder(x)
        output_seg = self.decoder(output_enc)

        if self.output_fn is not None:
            output_seg = self.output_fn(output_seg)

        return output_seg, output_enc


class ERFNetEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout_prob: float = 0.3, epistemic_version: str = "standard"):
        super().__init__()
        self.initial_block = blocks.DownsamplerBlock(in_channels, 16)

        self.layers = nn.ModuleList()

        self.layers.append(blocks.DownsamplerBlock(16, 64))

        dropout_prob_1, dropout_prob_2, dropout_prob_3 = self.get_dropout_probs(dropout_prob, epistemic_version)

        for x in range(0, 5):  # 5 times
            self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))

        self.layers.append(blocks.DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            dropout_prob_tmp = dropout_prob_2 if x == 0 else dropout_prob_3
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 2))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 4))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 8))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prob_tmp, 16))

    @staticmethod
    def get_dropout_probs(dropout_prob: float, epistemic_version: str) -> Tuple[float, float, float]:
        if epistemic_version == "all":
            return dropout_prob, dropout_prob, dropout_prob
        elif epistemic_version == "center":
            return 0, 0, dropout_prob
        elif epistemic_version == "classifier":
            return 0, 0, 0
        elif epistemic_version == "standard":
            return dropout_prob / 10, dropout_prob, dropout_prob
        else:
            raise ValueError(f"Epistemic version '{epistemic_version}' unknown!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.initial_block(x)

        for layer in self.layers:
            output = layer(output)

        return output


class ERFNetDeepEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout_prop: float = 0.3):
        super().__init__()
        self.initial_block = blocks.DownsamplerBlock(in_channels, 16)

        self.layers = nn.ModuleList()

        self.layers.append(blocks.DownsamplerBlock(16, 64))

        for x in range(0, 10):  # 10 times
            self.layers.append(blocks.non_bottleneck_1d(64, dropout_prop / 10, 1))

        self.layers.append(blocks.DownsamplerBlock(64, 128))

        for x in range(0, 3):  # 3 times
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prop, 2))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prop, 4))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prop, 8))
            self.layers.append(blocks.non_bottleneck_1d(128, dropout_prop, 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.initial_block(x)

        for layer in self.layers:
            output = layer(output)

        return output


class ERFNetDecoder(nn.Module):
    def __init__(self, num_classes: int, dropout_prob: float = 0.0, epistemic_version: str = "standard"):
        super().__init__()

        self.layers = nn.ModuleList()

        dropout_prob_1, dropout_prob_2, dropout_prob_3 = self.get_dropout_probs(dropout_prob, epistemic_version)

        self.layers.append(blocks.UpsamplerBlock(128, 64))
        self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))
        self.layers.append(blocks.non_bottleneck_1d(64, dropout_prob_1, 1))

        self.layers.append(blocks.UpsamplerBlock(64, 16))
        self.layers.append(blocks.non_bottleneck_1d(16, dropout_prob_2, 1))
        self.layers.append(blocks.non_bottleneck_1d(16, dropout_prob_3, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    @staticmethod
    def get_dropout_probs(dropout_prob: float, epistemic_version: str) -> Tuple[float, float, float]:
        if epistemic_version == "all":
            return dropout_prob, dropout_prob, dropout_prob
        elif epistemic_version == "center":
            return dropout_prob, 0, 0
        elif epistemic_version == "classifier":
            return 0, 0, dropout_prob
        elif epistemic_version == "standard":
            return 0, 0, 0
        else:
            raise ValueError(f"Epistemic version '{epistemic_version}' unknown!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output
