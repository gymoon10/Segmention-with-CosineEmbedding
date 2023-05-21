"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.halo_utils import HaloCosineEmbeddingLoss


criterion_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

criterion_halo1 = HaloCosineEmbeddingLoss(num_classes=3,
                                          halo_margin=15)  # hyper-parameters
criterion_halo2 = HaloCosineEmbeddingLoss(num_classes=3,
                                          halo_margin=15)  # hyper-parameters


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        print('Halo-Loss 2head')

    def forward(self, prediction, embedding, class_label,
                iou=False, meter_plant=None, meter_disease=None):
        '''prediction : model output (N, 4, 512, 512)
           embedding1: decoder feature1 (N, 32, 512, 512)
           embedding2: decoder feature2 (N, 32, 512, 512)
           instances : GT plant-mask (N, 512, 512)
           instances_2 : GT disease-mask (N, 512, 512)'''

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        loss = 0

        for b in range(0, batch_size):

            # cross entropy loss
            pred = prediction[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_label[b].unsqueeze(0)  # (1, h, w)
            gt_label = gt_label.type(torch.long).cuda()

            ce_loss = criterion_ce(pred, gt_label)

            if iou:
                pred = pred.detach().max(dim=1)[1]

                pred_plant = (pred == 1)
                pred_disease = (pred == 2)

                gt_plant = (class_label[b].unsqueeze(0) == 1)
                gt_disease = (class_label[b].unsqueeze(0) == 2)

                meter_plant.update(calculate_iou(pred_plant, gt_plant.cuda()))
                meter_disease.update(calculate_iou(pred_disease, gt_disease.cuda()))

            loss += ce_loss

        # halo loss
        loss_halo1 = criterion_halo1(embedding, class_label, class_idx=1)
        loss_halo2 = criterion_halo2(embedding, class_label, class_idx=2)

        loss = loss + (1 * loss_halo1) + (5 * loss_halo2)
        loss = loss + (5 * loss_halo2)

        return loss + prediction.sum() * 0


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou
