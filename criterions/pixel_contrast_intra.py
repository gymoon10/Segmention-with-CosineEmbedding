import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


criterion_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


def get_cosine_intra(labelmap, embedding):
    '''labelmap: (1, h, w)
       embedding: (E_channels, h, w)'''
    max_label = int(labelmap.unique().max().item())
    norm_means = [];
    means = []
    labels_intra_sum = 0.0

    E_channels = embedding.shape[0]
    cos_sim_dim0 = torch.nn.CosineSimilarity(dim=0)
    embedding_flat = embedding.view(E_channels, -1)  # (E_channels, h*w)

    for label in range(1, 2 + 1):  # exclude bg
        mask = (labelmap == label).flatten()  # roi pixels (pixels belonging to current object)
        count = mask.sum()

        # roi embedding (embeddings belonging to pixels of current object)
        mask_embedding = embedding_flat[:, mask]  # (E_channels, count)

        # get mean of roi embeddings per E_channels
        # mask_embedding[i].mean() = mean[i]
        mean = torch.sum(mask_embedding, dim=1) / count  # mean embedding  (E_channels, )
        means.append(mean)

        # l2_norm
        norm_mean = F.normalize(mean, p=2, dim=0)
        norm_means.append(norm_mean)

        # calculate within-instance loss term (per E_channels)
        # 1 - sum(cosine_sim(mean[i], mask_embedding[i])), i=1...E_channels
        norm_mean_repeat = norm_mean.unsqueeze(dim=1).expand(E_channels, count)  # (E_channels, count)
        intra = 1 - cos_sim_dim0(norm_mean_repeat, mask_embedding)
        labels_intra_sum += intra.sum() / count

    intra_mean = labels_intra_sum / (max_label)
    norm_means = torch.stack(norm_means, dim=0)  # (# of objects+1, E_channels), +1 for bg

    return means, norm_means, intra_mean


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding, prediction, class_label, iou=False, meter_plant=None, meter_disease=None):
        '''embedding : embedding network output (N, 32, 512, 512)
           prediction : seg model output (N, 3, 512, 512) *3 for bg/plant/disease
           instances_all : GT plant-mask (N, 512, 512)'''

        batch_size, height, width = embedding.size(
            0), embedding.size(2), embedding.size(3)

        loss = 0

        for b in range(0, batch_size):

            # 1.cosine intra loss
            label = class_label.type(torch.float64)
            label = label.type(torch.int64)

            _, _, intra_loss = \
                get_cosine_intra(label[b].unsqueeze(0), embedding[b])  # (3, 32) *3 for bg/plant/disease

            # 2.cross entropy loss
            pred = prediction[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_label[b].unsqueeze(0)  # (1, h, w)
            gt_label = gt_label.type(torch.long).cuda()

            ce_loss = criterion_ce(pred, gt_label)

            # total loss
            loss = loss + (intra_loss + ce_loss)

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
