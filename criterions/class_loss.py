import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000,
                 use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction="mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            #self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean")
            self.criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_index)

    def forward(self, emb, target):
        '''emb: (N, num_classes, H, W)
           target: (N, H, W) - [0, 1, ..., num_classes]'''
        b, c, h, w = emb.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(emb, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(emb, target)


#criterion_ce = OhemCrossEntropy2dTensor()
criterion_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, class_label, iou=False, meter_plant=None, meter_disease=None):
        '''embedding : embedding network output (N, 32, 512, 512)
           prediction : seg model output (N, 3, 512, 512) *3 for bg/plant/disease
           instances_all : GT plant-mask (N, 512, 512)'''

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        loss = 0

        for b in range(0, batch_size):

            # 3.cross entropy loss
            pred = prediction[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_label[b].unsqueeze(0)  # (1, h, w)
            gt_label = gt_label.type(torch.long).cuda()

            ce_loss = criterion_ce(pred, gt_label)

            # total loss
            loss = loss + ce_loss

            if iou:
                pred = pred.detach().max(dim=1)[1]

                pred_plant = (pred == 1)
                pred_disease = (pred == 2)

                gt_plant = (class_label[b].unsqueeze(0) == 1)
                gt_disease = (class_label[b].unsqueeze(0) == 2)

                meter_plant.update(calculate_iou(pred_plant, gt_plant.cuda()))
                meter_disease.update(calculate_iou(pred_disease, gt_disease.cuda()))

        return loss

def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou

