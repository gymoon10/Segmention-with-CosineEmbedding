import warnings
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# http://devdoc.net/python/scikit-image-doc-0.13.1/api/skimage.draw.html#skimage.draw.circle
# from skimage.draw import circle  0.16.2
# from skimage.draw import circle_perimeter  # 0.19.2


# Same as from skimage.draw import circle  0.16.2
# https://github.com/scikit-image/scikit-image/blob/v0.13.1/skimage/draw/draw.py#L144
def _ellipse_in_shape(shape, center, radii, rotation=0.):
    r_lim, c_lim = np.ogrid[0:float(shape[0]), 0:float(shape[1])]
    r_org, c_org = center
    r_rad, c_rad = radii
    rotation %= np.pi
    sin_alpha, cos_alpha = np.sin(rotation), np.cos(rotation)
    r, c = (r_lim - r_org), (c_lim - c_org)
    distances = ((r * cos_alpha + c * sin_alpha) / r_rad) ** 2 \
                + ((r * sin_alpha - c * cos_alpha) / c_rad) ** 2
    return np.nonzero(distances < 1)


def ellipse(r, c, r_radius, c_radius, shape=None, rotation=0.):

    center = np.array([r, c])
    radii = np.array([r_radius, c_radius])
    # allow just rotation with in range +/- 180 degree
    rotation %= np.pi

    # compute rotated radii by given rotation
    r_radius_rot = abs(r_radius * np.cos(rotation)) \
                   + c_radius * np.sin(rotation)
    c_radius_rot = r_radius * np.sin(rotation) \
                   + abs(c_radius * np.cos(rotation))
    # The upper_left and lower_right corners of the smallest rectangle
    # containing the ellipse.
    radii_rot = np.array([r_radius_rot, c_radius_rot])
    upper_left = np.ceil(center - radii_rot).astype(int)
    lower_right = np.floor(center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:2]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    rr, cc = _ellipse_in_shape(bounding_shape, shifted_center, radii, rotation)
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc


def circle(r, c, radius, shape=None):
    return ellipse(r, c, radius, radius, shape)


def flatten(x):
    return x.view(x.size(0), -1)


def build_halo_mask(fixed_depth=30, margin=21, min_fragment=10):
    """
    Function builds a configuration for halo region building
    :param fixed_depth: Maximum object on an image
    :param margin: The size of halo region
    :param min_fragment: Minimal size of an object on the image
    :return: a function for generation labels, masks and object_lists used by halo loss
    """
    assert margin % 2 != 0, "Margin should be odd"

    rr, cc = circle(margin / 2, margin / 2, margin / 2 + 1, shape=(margin, margin))
    structure_element = np.zeros((margin, margin))
    structure_element[rr, cc] = 1
    structure_element = np.repeat(np.expand_dims(np.expand_dims(structure_element, 0), 0), fixed_depth, 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sel = torch.from_numpy(structure_element).float().cuda()

    def f(label):
        """

        :param label: batch of instance levels each instance must have unique id
        :return:  labels, masks and object_lists used by halo loss
        """
        back = np.zeros((label.shape[0], fixed_depth, label.shape[1], label.shape[2]))
        object_list = []
        for i in range(label.shape[0]):
            bincount = np.bincount(label[i].flatten())
            pixels = np.where(bincount > min_fragment)[0]
            if len(pixels) > fixed_depth:
                pixels = pixels[:fixed_depth]
                warnings.warn("Not all objects fits in fixed depth", RuntimeWarning)

            for l, v in enumerate(pixels):
                back[i, l, label[i] == v] = 1.
            object_list.append(np.array(range(l + 1)))

        labels = torch.from_numpy(back).float().cuda()
        masks = F.conv2d(labels, sel, groups=fixed_depth, padding='same')

        masks[masks > 0] = 1.
        masks[labels > 0] = 2.
        masks[:, 0, :, :] = 1.

        weights = masks.sum(-1, keepdim=True).sum(-2, keepdim=True)
        weights[weights == 0.] = 1.

        masks = masks / weights

        return labels, masks, object_list

    return f


class HaloCosineEmbeddingLoss(nn.Module):
    def __init__(self, num_classes=3, halo_margin=15):
        super().__init__()
        self.num_classes = num_classes
        self.halo_margin = halo_margin

    def forward(self, embeddings, class_labels, class_idx):
        '''embeddings: (N, 32, 512, 512)
           class_labels: GT mask (N, 512, 512)  * (0, 1,...num_classes-1)'''

        batch_size, height, width = embeddings.size(
            0), embeddings.size(2), embeddings.size(3)

        # calculate halo cosine similarity loss
        # fixed_depth: pre-define # of colors(instances/classes)  * 3 for bg/plant/disease
        # margin: define the area of halo region
        mask_builder = build_halo_mask(fixed_depth=self.num_classes, margin=self.halo_margin, min_fragment=10)

        # labels: (N, 3, H, W), # classes=3 (0=bg,1=plant,2=disease)
        # halo: (N, 3, H, W)
        # objects: [array(0, 1, 2)] x N
        labels, halo, objects = mask_builder(class_labels.cpu())
        classes = objects[0]  # include bg

        c = class_idx
        loss_halo = 0

        for b in range(0, batch_size):

            # bg = halo[b, 0] * labels[b, 0]  # (H, W)
            indexes = torch.LongTensor(objects[b][0:]).cuda()  # exclude bg

            valid_labels = labels[b].index_select(0, indexes)  # (3, H, W)  * 3 for bg/plant/disease
            valid_weights = halo[b].index_select(0, indexes)  # (3, H, W)

            target_pos = valid_labels * valid_weights  # positive area (3, H, W)
            target_neg = (1. - valid_labels) * valid_weights  # negative(halo) area (3, H, W)

            emb = embeddings[b]  # (C, H, W)
            emb_dim = emb.shape[0]
            emb_flat = emb.view(emb_dim, -1)  # (C, H*W)

            mask_pos = (target_pos[c] != 0).unsqueeze(0).flatten()  # M pixels
            mask_neg = (target_neg[c] != 0).unsqueeze(0).flatten()  # N pixels

            pos_embedding = emb_flat[:, mask_pos]  # (C, M)
            neg_embedding = emb_flat[:, mask_neg]  # (C, N)

            # region contrast
            region_pos = torch.mean(pos_embedding, dim=1)  # torch.Size([C])
            region_neg = torch.mean(neg_embedding, dim=1)  # torch.Size([C])

            loss_halo = loss_halo + F.cosine_similarity(region_pos, region_neg, dim=0)

        return loss_halo
