import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


#criterion_ce_train = OhemCrossEntropy2dTensor()
criterion_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


def region_contrast_current(embedding, mask):
    """
    calculate region contrast value
    :param embedding: feature embedding of current class (C, H, W)
    :param mask: GT-mask fo current class (H, W) - fg/bg
    :return: value
    """
    mask = mask.numpy()
    E_channels = embedding.shape[0]
    embedding_flat = embedding.view(E_channels, -1)  # (E_channels, h*w)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_inner = cv2.erode(mask, se)  # dilated mask
    mask_boundary = mask - mask_inner

    area_inner = (mask_inner == 1).flatten()
    embedding_inner = embedding_flat[:, area_inner]  # (E_channels, count_inner)
    region_inner = torch.sum(embedding_inner, dim=1)  # (E_channels)

    area_boundary = (mask_boundary == 1).flatten()
    embedding_boundary = embedding_flat[:, area_boundary]
    region_boundary = torch.sum(embedding_boundary, dim=1)

    return F.cosine_similarity(region_inner, region_boundary, dim=0)


class CosineEmbeddingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, prediction, class_label, iou=False, meter_plant=None, meter_disease=None):
        '''embedding : embedding network output (N, 32, 512, 512)
           prediction : seg model output (N, 3, 512, 512) *3 for bg/plant/disease
           instances_all : GT plant-mask (N, 512, 512)'''

        batch_size, height, width = embeddings.size(
            0), embeddings.size(2), embeddings.size(3)

        loss = 0
        loss_intra = 0
        loss_inter = 0
        loss_ce = 0

        for b in range(0, batch_size):

            # region contrast
            embedding = embeddings[b]  # (C, H, W)
            emb_dim = embedding.shape[0]
            embedding_flat = embedding.view(emb_dim, -1)  # (C, H*W)

            plant_map = (class_label[b] == 1)
            disease_map = (class_label[b] == 2)

            mask_plant = (plant_map != 0).unsqueeze(0).flatten()
            mask_disease = (disease_map != 0).unsqueeze(0).flatten()

            emb_plant = embedding_flat[:, mask_plant]  # (C, M)
            emb_disease = embedding_flat[:, mask_disease]  # (C, N)

            region_plant = torch.sum(emb_plant, dim=1)
            region_disease = torch.sum(emb_disease, dim=1)

            contrast_loss = F.cosine_similarity(region_plant, region_disease, dim=0)

            # 3.cross entropy loss
            pred = prediction[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_label[b].unsqueeze(0)  # (1, h, w)
            gt_label = gt_label.type(torch.long).cuda()

            ce_loss = criterion_ce(pred, gt_label)

            # total loss
            loss = loss + (contrast_loss + ce_loss)
            #loss = loss + ce_loss
            loss_ce += ce_loss

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

