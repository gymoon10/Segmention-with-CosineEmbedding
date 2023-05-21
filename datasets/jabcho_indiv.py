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


class jabcho_indiv(Dataset):

    def __init__(self, root_dir='./', type_="train", size=None, transform=None):
        self.root_dir = root_dir
        self.type = type_

        # get image, foreground and instance list
        image_list = glob.glob(os.path.join(self.root_dir, '{}/0'.format(self.type), '*.JPG'))
        image_list.sort()
        self.image_list = image_list
        print("# image files: ", len(image_list))

        if self.type != 'test':
            instance_list = glob.glob(os.path.join(self.root_dir, '{}/1'.format(self.type), '*.png'))
            instance_list.sort()
            self.instance_list = instance_list
            print("# label files: ", len(instance_list))

            instance_list_2 = glob.glob(os.path.join(self.root_dir, '{}/1'.format(self.type), '*.png'))
            instance_list_2.sort()
            self.instance_list_2 = instance_list_2
            print("# label_2 files: ", len(instance_list_2))

            instance_list_3 = glob.glob(os.path.join(self.root_dir, '{}/1'.format(self.type), '*.png'))
            instance_list_3.sort()
            self.instance_list_3 = instance_list_3
            print("# label_3 files: ", len(instance_list_3))

        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform

        self.jitter = transforms.ColorJitter(brightness=0.1,
                                             contrast=0.1,
                                             saturation=0.1,
                                             hue=0.1)

        print('CVPPP Dataset created [{}]'.format(self.type))

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # load image and foreground
        image = Image.open(self.image_list[index]).convert('RGB')

        image = image.resize((512, 512), resample=Image.BILINEAR)

        # black_canvas = Image.new("RGB", image.size, 0)
        # fg = fg.resize((512,512), resample=Image.NEAREST).convert('L')
        # image = Image.composite(image, black_canvas, fg) # remove background

        width, height = image.size


        sample['image'] = image
        sample['im_name'] = self.image_list[index]

        if self.type != 'test':
            # # convert labels to instance map
            # instance_map = skimage.io.imread(self.instance_list[index])  # := instance map
            # instance_map = cv2.resize(instance_map, (512, 512), interpolation=cv2.INTER_NEAREST)
            # # instance_map = cv2.cvtColor(instance_map, cv2.COLOR_RGBA2GRAY)
            #
            # instance_ids = np.unique(instance_map)[1:]  # no background
            #
            #
            #
            # instance = np.zeros((height,width), dtype=np.uint8)
            #
            # label = np.zeros((height,width), dtype=np.uint8)
            # instance_counter = 0
            # for instance_id in instance_ids:
            #     instance_counter = instance_counter + 1
            #     mask = (instance_map == instance_id)
            #
            #     instance[mask] = instance_counter
            #     label[mask] = 1
            # ########################################class2##########################################
            #
            # instance_map_2 = skimage.io.imread(self.instance_list_2[index])  # := instance map
            # instance_map_2 = cv2.resize(instance_map_2, (512, 512), interpolation=cv2.INTER_NEAREST)
            # instance_map_2 = cv2.cvtColor(instance_map_2, cv2.COLOR_RGBA2GRAY)
            # instance_ids_2 = np.unique(instance_map_2)[1:]  # no background
            #
            # instance_2 = np.zeros((height,width), dtype=np.uint8)
            # label_2 = np.zeros((height,width), dtype=np.uint8)
            # instance_counter_2 = 0
            # for instance_id_2 in instance_ids_2:
            #     instance_counter_2 = instance_counter_2 + 1
            #     mask_2 = (instance_map_2 == instance_id_2)
            #
            #     instance_2[mask_2] = instance_counter_2
            #     label_2[mask_2] = 1

            ########################################class2##########################################

            instance_map_3 = skimage.io.imread(self.instance_list_3[index])  # := instance map
            instance_map_3 = cv2.resize(instance_map_3, (512, 512), interpolation=cv2.INTER_NEAREST)
            # instance_map_3 = cv2.cvtColor(instance_map_3, cv2.COLOR_RGBA2GRAY)
            instance_ids_3 = np.unique(instance_map_3)[1:]  # no background

            instance_3 = np.zeros((height, width), dtype=np.uint8)
            label_3 = np.zeros((height, width), dtype=np.uint8)

            instance_1 = np.zeros((height, width), dtype=np.uint8)
            label_1 = np.zeros((height, width), dtype=np.uint8)

            instance_2 = np.zeros((height, width), dtype=np.uint8)
            label_2 = np.zeros((height, width), dtype=np.uint8)

            instance_counter_3 = 0
            for instance_id_3 in instance_ids_3:
                instance_counter_3 = instance_counter_3 + 1
                mask_3 = (instance_map_3 == instance_id_3)

                instance_3[mask_3] = instance_counter_3
                label_3[mask_3] = 1
                if instance_counter_3 == 1:
                    instance_1[mask_3] = 1
                    label_1[mask_3] = 1
                else:
                    instance_2[mask_3] = 1
                    label_2[mask_3] = 1

        # --- data augmentation ---
        if self.type == 'train':
            # random hflip
            if random.random() > 0.5:
                # FLIP_TOP_BOTTOM
                sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
                instance_1 = np.fliplr(instance_1)
                label_1 = np.fliplr(label_1)
                instance_2 = np.fliplr(instance_2)
                label_2 = np.fliplr(label_2)
                instance_3 = np.fliplr(instance_3)
                label_3 = np.fliplr(label_3)


            # random vflip
            if random.random() > 0.5:
                # FLIP_LEFT_RIGHT
                sample['image'] = sample['image'].transpose(Image.FLIP_TOP_BOTTOM)
                instance_1 = np.flipud(instance_1)
                label_1 = np.flipud(label_1)
                instance_2 = np.flipud(instance_2)
                label_2 = np.flipud(label_2)
                instance_3 = np.flipud(instance_3)
                label_3 = np.flipud(label_3)

            # rotate 90 - clockwise
            if random.random() > 0.5:
                img_np = np.array(sample['image'])
                img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
                instance_1 = cv2.rotate(instance_1, cv2.ROTATE_90_CLOCKWISE)
                label_1 = cv2.rotate(label_1, cv2.ROTATE_90_CLOCKWISE)
                instance_2 = cv2.rotate(instance_2, cv2.ROTATE_90_CLOCKWISE)
                label_2 = cv2.rotate(label_2, cv2.ROTATE_90_CLOCKWISE)
                instance_3 = cv2.rotate(instance_3, cv2.ROTATE_90_CLOCKWISE)
                label_3 = cv2.rotate(label_3, cv2.ROTATE_90_CLOCKWISE)

                sample['image'] = Image.fromarray(img_np)

            # rotate 90 - counterclockwise
            if random.random() > 0.5:
                img_np = np.array(sample['image'])
                img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
                instance_1 = cv2.rotate(instance_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                label_1 = cv2.rotate(label_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                instance_2 = cv2.rotate(instance_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                label_2 = cv2.rotate(label_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                instance_3 = cv2.rotate(instance_3, cv2.ROTATE_90_COUNTERCLOCKWISE)
                label_3 = cv2.rotate(label_3, cv2.ROTATE_90_COUNTERCLOCKWISE)


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
            label_1 = Image.fromarray(np.uint8(label_1))
            instance_1 = Image.fromarray(np.uint8(instance_1))

            sample['instance'] = instance_1
            sample['label'] = label_1
 #####################################################################

            label_2 = Image.fromarray(np.uint8(label_2))
            instance_2 = Image.fromarray(np.uint8(instance_2))

            sample['instance_2'] = instance_2
            sample['label_2'] = label_2

 #####################################################################

            label_3 = Image.fromarray(np.uint8(label_3))
            instance_3 = Image.fromarray(np.uint8(instance_3))

            sample['instance_3'] = instance_3
            sample['label_3'] = label_3
        # transform
        if (self.transform is not None):
            sample = self.transform(sample)

        return sample