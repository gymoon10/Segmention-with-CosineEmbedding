"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet_ori as erfnet


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc

        return out


class ERFNet_Semantic_Ori(nn.Module):
    '''shared encoder + 2 branched decoders'''

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        print('Creating Original ERFNet model with {} classes'.format(num_classes))
        # num_classes=[3, 1], 3 for instance-branch (1 sigma) & 1 for seed branch
        # num_classes=[4, 1], 4 for instance-branch (2 sigma) & 1 for seed branch

        # shared encoder
        self.encoder = erfnet.Encoder()  # Encoder(3+1)
        self.decoder = erfnet.Decoder(3)

    def forward(self, input_):

        feat_enc = self.encoder(input_)  # (N, 128, 64, 64)
        image = self.decoder.forward(feat_enc)  # (N, 2, 512, 512)  * 2 for bg & disease

        return image
