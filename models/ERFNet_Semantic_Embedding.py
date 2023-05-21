"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet_embedding as network


class ERFNet_Semantic_Embedding(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        print('Creating Branched ERFNet_Semantic_Embedding with {} classes'.format(num_classes))

        self.encoder = network.Encoder()
        self.decoder = network.Decoder()


    def forward(self, input_):

        feat0, feat1, feat2 = self.encoder(input_)  # (N, 128, 64, 64)
        embedding, seg_out = self.decoder.forward(feat0, feat1, feat2)  # (N, 32, 512, 512)

        return embedding, seg_out
