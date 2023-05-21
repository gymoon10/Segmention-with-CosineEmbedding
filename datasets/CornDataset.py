import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from numpy.core.fromnumeric import transpose
from scipy import ndimage, misc
from PIL import Image
from torch.utils.data import Dataset

h, w = 512, 512

class CornDataset(Dataset):

    def __init__(self, root_dir='./', type_="train", size=None, transform=None):
        self.root_dir = root_dir
        self.type = type_

        # get image, foreground and instance list
        image_list = glob.glob(os.path.join(self.root_dir, 'images/{}/'.format(self.type), '*_rgb.png'))
        image_list.sort()
        self.image_list = image_list
        print("# image files: ", len(image_list))

        if self.type != 'test':
            instance_list_all = glob.glob(os.path.join(self.root_dir, 'images/{}/'.format(self.type), '*_overall.png'))
            instance_list_all.sort()
            self.instance_list_all = instance_list_all
            print("# label_all files: ", len(instance_list_all))

        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform

        self.jitter = transforms.ColorJitter(brightness=0.1,
                                             contrast=0.1,
                                             saturation=0.1,
                                             hue=0.1)

        print('Corn dataset created [{}]'.format(self.type))

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # load image and foreground
        image = Image.open(self.image_list[index]).convert('RGB')
        image = image.resize((h, w), resample=Image.BILINEAR)
        width, height = image.size

        sample['image'] = image
        sample['im_name'] = self.image_list[index]

        if self.type != 'test':

            instance_map_all = skimage.io.imread(self.instance_list_all[index])  # := instance map
            instance_map_all = cv2.resize(instance_map_all, (h, w), interpolation=cv2.INTER_NEAREST)
            instance_map_all = cv2.cvtColor(instance_map_all, cv2.COLOR_RGBA2GRAY)
            class_ids = np.unique(instance_map_all)[1:]  # no background

            instance_all = np.zeros((height, width), dtype=np.uint8)
            label_all = np.zeros((height, width), dtype=np.uint8)

            for class_id in class_ids:
                if class_id == 10:
                    mask_plant = (instance_map_all == class_id)
                    instance_all[mask_plant] = 1
                    label_all[mask_plant] = 1

                elif class_id == 20:
                    mask_disease = (instance_map_all == class_id)
                    instance_all[mask_disease] = 2
                    label_all[mask_disease] = 2

            instance_plant = (instance_all == 1) * 1
            label_plant = instance_plant

            instance_disease = (instance_all == 2) * 1
            label_disease = instance_disease

        # --- data augmentation ---
        if self.type == 'train':
            # random hflip
            if random.random() > 0.5:
                # FLIP_TOP_BOTTOM
                sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
                instance_plant = np.fliplr(instance_plant)
                label_plant = np.fliplr(label_plant)
                instance_disease = np.fliplr(instance_disease)
                label_disease = np.fliplr(label_disease)
                instance_all = np.fliplr(instance_all)
                label_all = np.fliplr(label_all)

            # random vflip
            if random.random() > 0.5:
                # FLIP_LEFT_RIGHT
                sample['image'] = sample['image'].transpose(Image.FLIP_TOP_BOTTOM)
                instance_plant = np.flipud(instance_plant)
                label_plant = np.flipud(label_plant)
                instance_disease = np.flipud(instance_disease)
                label_disease = np.flipud(label_disease)
                instance_all = np.flipud(instance_all)
                label_all = np.flipud(label_all)

            # rotate 90 - clockwise
            if random.random() > 0.5:
                img_np = np.array(sample['image'])
                img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
                instance_plant = cv2.rotate(instance_plant, cv2.ROTATE_90_CLOCKWISE)
                label_plant = cv2.rotate(label_plant, cv2.ROTATE_90_CLOCKWISE)
                instance_disease = cv2.rotate(instance_disease, cv2.ROTATE_90_CLOCKWISE)
                label_disease = cv2.rotate(label_disease, cv2.ROTATE_90_CLOCKWISE)
                instance_all = cv2.rotate(instance_all, cv2.ROTATE_90_CLOCKWISE)
                label_all = cv2.rotate(label_all, cv2.ROTATE_90_CLOCKWISE)

                sample['image'] = Image.fromarray(img_np)

            # rotate 90 - counterclockwise
            if random.random() > 0.5:
                img_np = np.array(sample['image'])
                img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
                instance_plant = cv2.rotate(instance_plant, cv2.ROTATE_90_COUNTERCLOCKWISE)
                label_plant = cv2.rotate(label_plant, cv2.ROTATE_90_COUNTERCLOCKWISE)
                instance_disease = cv2.rotate(instance_disease, cv2.ROTATE_90_COUNTERCLOCKWISE)
                label_disease = cv2.rotate(label_disease, cv2.ROTATE_90_COUNTERCLOCKWISE)
                instance_all = cv2.rotate(instance_all, cv2.ROTATE_90_COUNTERCLOCKWISE)
                label_all = cv2.rotate(label_all, cv2.ROTATE_90_COUNTERCLOCKWISE)

                sample['image'] = Image.fromarray(img_np)

            # random background
            if random.random() > 0.5:
                img_pil = sample['image']
                img_pil = img_pil.convert('RGBA')

                key = np.random.choice([0, 1, 2, 3])

                if key == 0:
                    bg = Image.new('RGBA', img_pil.size, (255,) * 4)  # White image
                elif key == 1:
                    bg = Image.new('RGBA', img_pil.size, (0, 0, 0, 255))  # Black image
                elif key == 2:
                    img_np = np.array(img_pil)
                    mean_color = img_np.mean((0, 1))
                    bg = Image.new('RGBA', img_pil.size,
                                   (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]), 255))  # mean color
                elif key == 3:
                    bg = Image.new('RGBA', img_pil.size,
                                   (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255))  # random color

                img_pil = Image.blend(img_pil, bg, 0.25)  # blend
                sample['image'] = img_pil.convert('RGB')

            # random gamma
            if random.random() > 0.5:
                img_pil = sample['image']

                gain = 1
                gamma_range = [0.7, 1.8]
                min_gamma = gamma_range[0]
                max_gamma = gamma_range[1]
                gamma = np.random.rand() * (max_gamma - min_gamma) + min_gamma

                gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
                img_pil = img_pil.point(gamma_map)

                sample['image'] = img_pil

            # random grayscaling
            grayscaler = transforms.RandomGrayscale(p=0.3)
            sample['image'] = grayscaler(sample['image'])

            # random jittering
            if random.random() > 0.5:
                # need to applied on PIL Image
                sample['image'] = self.jitter(sample['image'])

        if self.type != 'test':
            label_plant = Image.fromarray(np.uint8(label_plant))
            instance_plant = Image.fromarray(np.uint8(instance_plant))

            sample['instance_plant'] = instance_plant
            sample['label_plant'] = label_plant
 #####################################################################

            label_disease = Image.fromarray(np.uint8(label_disease))
            instance_disease = Image.fromarray(np.uint8(instance_disease))

            sample['instance_disease'] = instance_disease
            sample['label_disease'] = label_disease

 #####################################################################

            label_all = Image.fromarray(np.uint8(label_all))
            instance_all = Image.fromarray(np.uint8(instance_all))

            sample['instance_all'] = instance_all
            sample['label_all'] = label_all

        # print(sample)


        # transform

        if (self.transform is not None):
            sample = self.transform(sample)


        return sample