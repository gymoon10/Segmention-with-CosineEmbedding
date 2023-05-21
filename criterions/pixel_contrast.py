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

    for label in range(0, max_label+1):  # include bg
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

        if label != 0:
            intra = 1 - cos_sim_dim0(norm_mean_repeat, mask_embedding)
            labels_intra_sum += intra.sum() / count

    intra_mean = labels_intra_sum / (max_label)
    norm_means = torch.stack(norm_means, dim=0)  # (# of objects+1, E_channels), +1 for bg

    return means, norm_means, intra_mean


def get_neighbor_by_distance(label_map, distance=10, max_neighbor=3):
    '''neighbor_indice[i]: ids of neighboring classes for current i th class
       output: (max_neighbor, max_neighbor)'''
    label_map = label_map.copy()

    def _adjust_size(x):
        if len(x) >= max_neighbor:
            return x[0:max_neighbor]
        else:
            return np.pad(x, (0, max_neighbor - len(x)), 'constant', constant_values=(0, 0))

    # idx of instance labels
    unique = np.unique(label_map)
    assert unique[0] == 0
    if len(unique) <= 2:  # only one instance
        return None

    neighbor_indice = np.zeros((max_neighbor, max_neighbor))
    label_flat = label_map.reshape((-1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distance * 2 + 1, distance * 2 + 1))
    # kernel = np.ones((distance * 2 + 1, distance * 2 + 1))

    for i, label in enumerate(unique[1:]):
        assert i + 1 == label
        mask = (label_map == label)  # mask of specific class
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).reshape((-1))

        # dilated_mask n GT class mask
        neighbor_pixel_ind = np.logical_and(dilated_mask > 0, label_flat != 0)
        # (dilated_mask n GT insatnce mask) n (GT instance mask - current instance)
        neighbor_pixel_ind = np.logical_and(neighbor_pixel_ind, label_flat != label)

        # for current instance, save unique idxs of neighboring instances
        neighbors = np.unique(label_flat[neighbor_pixel_ind])
        neighbor_indice[i + 1, :] = _adjust_size(neighbors)  # padding

    return neighbor_indice.astype(np.int32)


def get_cosine_inter(means, neighbor, n_emb):
    '''means: normalized mean embeddings (# of objects+1, E_channels)
       neighbor: (3, 3)  * 3 for bg/plant/disease
       n_emb: E_channels'''

    cos_sim = torch.nn.CosineSimilarity(dim=1)
    bg_include_n_labels = len(means)  # bg + all instances

    # supoose bg_include_n_labels=3
    # main_means: mean(bg)x3, mean(plant)x3, mean(disease)x3
    # neighbor_means: [mean(bg), mean(plant), mean(disease)] x 3
    main_means = means.unsqueeze(1).expand(bg_include_n_labels, bg_include_n_labels, n_emb)
    neighbor_means = main_means.clone().permute(1, 0, 2)
    main_means = main_means.reshape(-1, n_emb)  # (3^2, 32)
    neighbor_means = neighbor_means.reshape(-1, n_emb)  # (3^2, 32)

    # calculate cosine similarity btw
    # mean(bg) & mean(bg)/mean(plant)/mean(disease)
    # mean(plant) & mean(bg)/mean(plant)/mean(disease)
    # mean(disease) & mean(bg)/mean(plant)/mean(disease)
    inter = cos_sim(neighbor_means, main_means).view(bg_include_n_labels, bg_include_n_labels)

    # local neighbor
    inter_mask = torch.zeros(bg_include_n_labels, bg_include_n_labels, dtype=torch.float)

    for main_label in range(1, bg_include_n_labels):
        for nei_label in neighbor[main_label]:  # neighbor[i]: ids of neighboring instances of i th instance
            if nei_label == 0:
                break
            inter_mask[main_label][nei_label] = 1.0

    inter_mask = inter_mask.cuda()
    inter_mean = torch.sum(inter * inter_mask) / torch.sum(inter_mask)

    return inter_mean


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

            mean_embeddings, norm_mean_embeddings, intra_loss = \
                get_cosine_intra(class_label[b], embedding[b])  # (3, 32) *3 for bg/plant/disease
            #print('intra_loss :', intra_loss)

            # 2. cosine inter loss
            # neighbor_indice[i]: ids of neighboring classes for current i th class
            label_map = class_label[b].type(torch.uint8).cpu().detach().numpy()  # (512, 512)
            neighbor = get_neighbor_by_distance(label_map, distance=10, max_neighbor=3)  # (3, 3)

            inter_loss = get_cosine_inter(norm_mean_embeddings, neighbor, n_emb=32)
            print('inter_loss :', inter_loss)
            #print()

            # 3.cross entropy loss
            pred = prediction[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_label[b].unsqueeze(0)  # (1, h, w)
            gt_label = gt_label.type(torch.long).cuda()

            ce_loss = criterion_ce(pred, gt_label)

            # total loss
            loss = loss + (intra_loss + inter_loss + ce_loss)

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
