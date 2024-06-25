"""
Implementation of the Adaptive U-Net using Adaptive Convolution to include acquisition parameters
(Voxel-Size, FoV-Orientation) in the deep learning model. The Filter Manifold Network (FMN) computes the convolution
filter kernel values based on the presented additional information.


Created by Simon Graf

The usage of the code may be conducted in line with the GPL-3.0 license of the DL-QSM repository.

"""

import torch
import torch.nn as nn


class DoubleConvBlockELU(nn.Module):
    """
    General class for performing double convolutions. Suitable for encoding and bottleneck. Usage of ELU.
    """

    def __init__(self, in_channel, out_channel):
        super(DoubleConvBlockELU, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False)
        self.BN1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False)
        self.BN2 = nn.BatchNorm3d(out_channel)
        self.elu = nn.ELU()

    def forward(self, input):
        out = self.conv1(input)
        out = self.BN1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.BN2(out)
        out = self.elu(out)
        return out


class DecodingConvBlockELU(nn.Module):
    """
    This block consists of the up-sampling through transposed convolution and the respective concatenation
    and double convolution.
    """

    def __init__(self, in_channel, out_channel):
        super(DecodingConvBlockELU, self).__init__()

        self.conv_up = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.BN_up = nn.BatchNorm3d(out_channel)
        self.elu_up = nn.ELU()
        # here in_channel since we have the concatenation
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False)
        self.BN1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False)
        self.BN2 = nn.BatchNorm3d(out_channel)
        self.elu = nn.ELU()

    def forward(self, input, input_enc):
        # up-sampling of feature maps from the middle block
        out = self.conv_up(input)
        out = self.BN_up(out)
        out = self.elu_up(out)
        # concatenate the information from encoder and decoder
        cat = torch.cat([out, input_enc], 1)
        # perform double convolution
        out = self.conv1(cat)
        out = self.BN1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.BN2(out)
        out = self.elu(out)
        return out


class LastConvBlockELU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LastConvBlockELU, self).__init__()

        self.conv_up = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.BN_up = nn.BatchNorm3d(out_channel)
        self.elu_up = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False)
        self.BN1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False)
        self.BN2 = nn.BatchNorm3d(out_channel)
        self.elu = nn.ELU(inplace=True)
        self.last = nn.Conv3d(out_channel, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0,
                              bias=True)

    def forward(self, input, input_first):
        # up-sampling of feature maps from the middle block
        out = self.conv_up(input)
        out = self.BN_up(out)
        out = self.elu_up(out)
        # concatenate the information from the first layer and decoder
        cat = torch.cat([out, input_first], 1)
        out = self.conv1(cat)
        out = self.BN1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.BN2(out)
        out = self.elu(out)
        out = self.last(out)
        return out


class FMNetworkELU(nn.Module):
    """
    THis class defines the FMN network, a fully-connected feed-forward network that receives a certain side information
    tensor as input. The output of the network are the weights for the 3d convolution operation. ELU activation is used.
    """

    def __init__(self, initial_size, size_weights):
        super(FMNetworkELU, self).__init__()
        self.initial_size = initial_size
        self.size_weights = size_weights

        def layer(in_features, out_features):
            linear_layer = [nn.Linear(in_features=in_features, out_features=out_features, bias=True),
                            nn.ELU()]
            return linear_layer

        self.layers = nn.Sequential(
            *layer(in_features=6, out_features=self.initial_size),
            *layer(in_features=self.initial_size, out_features=self.initial_size * 4),
            *layer(in_features=self.initial_size * 4, out_features=self.initial_size * 16),
            *layer(in_features=initial_size * 16, out_features=self.size_weights)
        )

    def forward(self, input):
        weights = self.layers(input)

        return weights


class AdaptiveConv(nn.Module):
    """
    This class defines the adaptive convolution operation using the FMN network.
    """

    def __init__(self, in_channel, out_channel):
        super(AdaptiveConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = 3
        self.size_FMN = int(in_channel * out_channel * self.kernel_size ** 3)
        self.FMN = FMNetworkELU(12, self.size_FMN)
        self.BN = nn.BatchNorm3d(out_channel)
        self.elu = nn.ELU(inplace=True)

    def forward(self, input, side_information):
        # compute the weights with the FMN
        conv_weight = self.FMN(side_information).reshape(self.out_channel, self.in_channel,
                                                         self.kernel_size, self.kernel_size,
                                                         self.kernel_size)
        # use the functional API for the convolution
        out = torch.nn.functional.conv3d(input, conv_weight, bias=None, stride=(1, 1, 1), padding=(1, 1, 1))
        out = self.BN(out)
        out = self.elu(out)

        return out


class AdaptiveConvBlock(nn.Module):
    """
    Class for performing double convolution with ELU activation.
    One Adaptive convolution and one conventional convolution.
    """

    def __init__(self, in_channel, out_channel):
        super(AdaptiveConvBlock, self).__init__()
        self.EncodingConv1 = AdaptiveConv(in_channel, out_channel)
        self.EncodingConv2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3),
                                       stride=(1, 1, 1), padding=1, bias=False)
        self.BN = nn.BatchNorm3d(out_channel)
        self.elu = nn.ELU(inplace=True)

    def forward(self, input, side_information):
        out = self.EncodingConv1(input, side_information)
        out = self.EncodingConv2(out)
        out = self.BN(out)
        out = self.elu(out)
        return out


class AdaptiveUNet3dELU(nn.Module):
    """
    Adaptive U-Net with ELU activation
    Res parameter enables residual learning (res = True)
    """

    def __init__(self, initial_channels, res):
        super(AdaptiveUNet3dELU, self).__init__()
        self.initial_channels = initial_channels

        # define the network layers
        self.first = DoubleConvBlockELU(1, initial_channels)
        self.adaptive = AdaptiveConvBlock(initial_channels, initial_channels * 2)
        self.enc2 = DoubleConvBlockELU(initial_channels * 2, initial_channels * 4)
        self.enc3 = DoubleConvBlockELU(initial_channels * 4, initial_channels * 8)
        self.middle = DoubleConvBlockELU(initial_channels * 8, initial_channels * 16)
        self.dec3 = DecodingConvBlockELU(initial_channels * 16, initial_channels * 8)
        self.dec2 = DecodingConvBlockELU(initial_channels * 8, initial_channels * 4)
        self.dec1 = DecodingConvBlockELU(initial_channels * 4, initial_channels * 2)
        self.last = LastConvBlockELU(initial_channels * 2, initial_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.res = res

    def forward(self, input, side_information):
        # Encoder
        first = self.first(input)
        max_first = self.maxpool(first)
        adaptive1 = self.adaptive(max_first, side_information)
        max1 = self.maxpool(adaptive1)
        enc2 = self.enc2(max1)
        max2 = self.maxpool(enc2)
        enc3 = self.enc3(max2)
        max3 = self.maxpool(enc3)
        # Bottleneck
        middle = self.middle(max3)
        # Decoder, the decoding layers have up-sampling included
        dec3 = self.dec3(middle, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, adaptive1)
        last = self.last(dec1, first)
        if self.res is True:
            out = input + last
            # print('residual')
        else:
            out = last.float()
            # print('No residual')

        return out


def number_parameter(model):
    """
    Function that gives the amount of trainable parameters as well as the total parameter amount
    :param model:
    :return:
    """
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    image = torch.rand((1, 1, 320, 320, 320))
    side_information = torch.Tensor((1, 1, 1, 0, 0, 1))
    model = AdaptiveUNet3dELU(16, 'True')
    print(number_parameter(model))
    print(model.eval())
    out = model(image, side_information)
