# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as T


class DownsamplerBlock(nn.Module):
    '''Downsampling by concatenating parallel output of
    3x3 conv(stride=2) & max-pooling'''

    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    '''Factorized residual layer
    dilation can gather more context (a.k.a. atrous conv)'''

    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        # factorize 3x3 conv to (3x1) x (1x3) x (3x1) x (1x3)
        # non-linearity can be added to each decomposed 1D filters
        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1 * dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1 * dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)  # non-linearity

        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)

        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)  # (N, 16, 256, 256)

        self.layers1 = nn.ModuleList()
        self.layers1.append(DownsamplerBlock(16, 64))  # (N, 64, 128, 128)

        for x in range(0, 5):  # 5 times (no dilation)
            self.layers1.append(non_bottleneck_1d(64, 0.03, 1))  # (N, 64, 128, 128)

        self.layers2 = nn.ModuleList()
        self.layers2.append(DownsamplerBlock(64, 128))  # (N, 128, 64, 64)
        for x in range(0, 2):  # 2 times (with dilation)
            self.layers2.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers2.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers2.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers2.append(non_bottleneck_1d(128, 0.3, 16))  # (N, 128, 64, 64)


    def forward(self, input):
        '''input: (N, 3, 512, 512)'''
        output = self.initial_block(input)  # (N, 16, 256, 256)
        output0 = output

        for layer in self.layers1:
            output = layer(output)  # (N, 64, 128, 128)
        output1 = output

        for layer in self.layers2:
            output = layer(output)  # (N, 128, 64, 64)
        output2 = output

        return output0, output1, output2

class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        # do not use max-unpooling operation
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class double_conv(nn.Module):
    '''(Conv + B.N + ReLU) x 2'''

    def __init__(self, in_ch, out_ch):
        super().__init__()

        # Conv block
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch, eps=1e-03),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch, eps=1e-03),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        output = self.conv(input)
        return output


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.up1 = UpsamplerBlock(128, 64)  # (N, 64, 128, 128)
        self.conv1 = nn.ModuleList()
        self.conv1.append(non_bottleneck_1d(128, 0, 1))
        self.conv1.append(non_bottleneck_1d(128, 0, 1))  # (N, 128, 128, 128)

        self.up2 = UpsamplerBlock(128, 64)  # (N, 64, 256, 256)
        self.conv2 = double_conv(64, 16)

        self.conv2_1 = nn.ModuleList()
        self.conv2_1.append(non_bottleneck_1d(32, 0, 1))
        self.conv2_1.append(non_bottleneck_1d(32, 0, 1))  # (N, 32, 256, 256)
        self.conv2_2 = double_conv(32, 32)

        self.emb_conv = nn.ConvTranspose2d(
            32, 32, 2, stride=2, padding=0, output_padding=0, bias=True)

        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feat_con2, feat_con1, feat_enc):
        '''feat_con1: (N, 64, 128, 128)
           feat_con2: (N, 16, 256, 256)
           feat_enc: (N, 128, 64, 64)'''
        feat_up1 = self.up1(feat_enc)  # (N, 64, 128, 128)
        feature = torch.cat([feat_up1, feat_con1], dim=1)  # (N, 128, 128, 128)
        for layer in self.conv1:
            feature = layer(feature)  # (N, 128, 128, 128)

        feat_up2 = self.up2(feature)  # (N, 64, 256, 256)
        feat_up2 = self.conv2(feat_up2)  # (N, 16, 256, 256)
        feature = torch.cat([feat_up2, feat_con2], dim=1)  # (N, 32, 256, 256)
        for layer in self.conv2_1:
            feature = layer(feature)  # (N, 32, 256, 256)
        feature = self.conv2_2(feature)  # (N, 32, 256, 256)

        embedding = self.emb_conv(feature)  # (N, 32, 512, 512)
        seg_out = self.output_conv(embedding)  # (N, # classes, 512, 512)

        return embedding, seg_out

