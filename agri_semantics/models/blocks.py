import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, dropout_prob: float = 0.0):
        super(ConvBlock, self).__init__()

        mid_channels = out_channels if mid_channels is None else mid_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.dropout.p != 0:
            x = self.dropout(x)

        x = self.conv2(x)
        if self.dropout.p != 0:
            x = self.dropout(x)

        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.0):
        super(DownsampleBlock, self).__init__()

        self.down_sample_block = nn.Sequential(
            nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, dropout_prob=dropout_prob)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_sample_block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False, dropout_prob: float = 0.0):
        super(UpsampleBlock, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2, dropout_prob=dropout_prob)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels, dropout_prob=dropout_prob)

    def forward(self, x: torch.Tensor, x_enc: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = x_enc.size()[2] - x.size()[2]
        diff_x = x_enc.size()[3] - x.size()[3]

        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x_enc, x], dim=1)

        return self.conv(x)


class OutputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutputConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: int) -> torch.Tensor:
        return self.conv(x)


##############################################################################################
#                                                                                            #
#  ERFNET blocks from https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py  #
#                                                                                            #
##############################################################################################


class DropoutDownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, dropout_prob):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        output = F.gelu(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return output


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.gelu(output)


class DropoutUpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, dropout_prob):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = F.gelu(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.gelu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1)
        )
        self.conv1x3_2 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated)
        )
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.gelu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.gelu(output)

        output = self.conv3x1_2(output)
        output = F.gelu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.gelu(output + input)  # +input = identity (residual connection)
