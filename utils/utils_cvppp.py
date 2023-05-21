"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import os
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.utils import save_image
import torch


save_1 = 'D:/lsy/cluster_1'
save_2 = 'D:/lsy/cluster_2'

if not os.path.exists(save_1):
    os.makedirs(save_1)

if not os.path.exists(save_2):
    os.makedirs(save_2)


class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x / y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class) / len(self.avg_per_class)


class Visualizer:

    def __init__(self, keys):
        self.wins = {k: None for k in keys}

    def display(self, image, key):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1

        if self.wins[key] is None:
            self.wins[key] = plt.subplots(ncols=n_images)

        fig, ax = self.wins[key]
        n_axes = len(ax) if isinstance(ax, collections.Iterable) else 1

        assert n_images == n_axes

        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
            ax.imshow(self.prepare_img(image))
        else:
            for i in range(n_images):
                ax[i].cla()
                ax[i].set_axis_off()
                ax[i].imshow(self.prepare_img(image[i]))

        plt.draw()
        self.mypause(0.001)

    @staticmethod
    def prepare_img(image):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return


class Cluster:

    def __init__(self, ):


        xm = torch.linspace(0, 1, 512).view(1, 1, -1).expand(1, 512, 512)
        print(xm)
        ym = torch.linspace(0, 1, 512).view(1, -1, 1).expand(1, 512, 512)
        xym = torch.cat((xm, ym), 0)

        self.xym = xym.cuda()

    def cluster_with_gt(self, prediction, instance, instance_2, n_sigma=1, ):
        '''cluster the pixel embeddings into instance (training)'''

        height, width = prediction.size(1), prediction.size(2)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w

        # spatial_emb = offset_vectors (first 2 channels of model output) + coordinate maps
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w

        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).byte().cuda()
        instance_map_2 = torch.zeros(height, width).byte().cuda()

        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        unique_instances_2 = instance_2.unique()
        unique_instances_2 = unique_instances_2[unique_instances_2 != 0]

        # for each specific instance
        for id in unique_instances:
            # mask of specific instance (RoI area)
            # spatial_emb, sigma below consider only pixels of RoI area
            mask = instance.eq(id).view(1, height, width)  # (1, h, w)

            # center of instance (mean embedding)
            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

            # define sigma_k - e.q (7)
            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)
            s = torch.exp(s * 10)  # n_sigma x 1 x 1

            # calculate gaussian score (distance of each pixel embedding from the center)
            # high value -> pixel embedding is close to the center
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))  # (h, w)

            # e.q (11)
            proposal = (dist > 0.5)  # (h, w)
            instance_map[proposal] = id  # (h, w)

        for id in unique_instances_2:
            # mask of specific instance (RoI area)
            # spatial_emb, sigma below consider only pixels of RoI area
            mask = instance.eq(id).view(1, height, width)  # (1, h, w)

            # center of instance (mean embedding)
            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

            # define sigma_k - e.q (7)
            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)
            s = torch.exp(s * 10)  # n_sigma x 1 x 1

            # calculate gaussian score (distance of each pixel embedding from the center)
            # high value -> pixel embedding is close to the center
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))  # (h, w)

            # e.q (11)
            proposal_2 = (dist > 0.5)  # (h, w)
            instance_map_2[proposal_2] = id  # (h, w)


        return instance_map, instance_map_2

    def cluster(self, prediction, n_sigma=1, threshold=0.5, count_12=1):
        '''for inference
           prediction: (4, h, w)
           threshold: threshold for center'''

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width] # 좌표평면 0~1값으로 나타냄.
        # print('h')
        # print(torch.tanh(prediction[0]))

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w 0~1로 나타낸 값에 offset vector를 더해줌. 위치가 옮겨짐.
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w  sigma값을 추출함.

        # sigmoid is applied to seed_map
        # as the regression loss is used for training, b.g pixels become zero
        # it is similar to semantic mask
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w  seed map을 추출함.
        seed_map_2 = torch.sigmoid(prediction[2 + n_sigma + 1: 2 + n_sigma + 2])
        save_image(seed_map.squeeze(),
                   os.path.join(save_1, '{}_seed_mask_overall.png'.format(
                       count_12)))
                   # 'A:/220821_spatial_embedding/221007_NLB_semantic_both_segment_base_rgb/val_result/best_recon_iou/cluster_1/{}_seed_mask_overall.png'.format(
                   #     count_12))

        save_image(seed_map_2.squeeze(),
                   os.path.join(save_2, '{}_seed_mask_overall.png'.format(
                       count_12)))

        # save_image(seed_map_2.squeeze(),
        #            'A:/220821_spatial_embedding/221007_NLB_semantic_both_segment_base_rgb/val_result/best_recon_iou/cluster_2/{}_seed_mask_overall.png'.format(
        #                count_12))
        # save_image(seed_map.squeeze(),'A:/220821_spatial_embedding/220908_lsy_result_seedmap_cluster_1/seed_map_1.png'
        #            )
        # save_image(seed_map_2.squeeze(),'A:/220821_spatial_embedding/220908_lsy_result_seedmap_cluster_2/seed_map_2.png'
        #            )

        instance_map = torch.zeros(height, width).byte()
        instance_map_2 = torch.zeros(height, width).byte()
        instance_map_3 = torch.zeros(height, width).byte()
        instances = []
        instances_2 = []

        count = 1
        recount = 1
        recount_2 = 1

        # RoI pixels - set of pixels belonging to instances
        mask = seed_map > 0.5  # (1, h, w) seed_map에서 0.5 보다 큰 값을 가지는 픽셀만 true를 가짐.
        mask_2 = seed_map_2 > 0.5


        if mask.sum() > 128: #seed map에서 값의 합이 128을 넘는다면 실행됨.

            # only consider the pixels which belong to mask (RoI pixels)
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)  # true가 나온 영역의 값만을 가지는 vector가 나옴.
            # print(spatial_emb.shape)
            # save_image(spatial_emb[0],'A:/220821_plant_dataset/lsy_a4/images/save_test/spatial_emb/spatial_emb1.png')
            # save_image(spatial_emb[1], 'A:/220821_plant_dataset/lsy_a4/images/save_test/spatial_emb/spatial_emb2.png')




            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)  # (1, N) # true가 나온 영역(seed map이 0.5보다 큰 영역)의 값만을 가지는 sigma가 나옴.

            seed_map_masked = seed_map[mask].view(1, -1)  # (1, N) true가 나오는 영역의 값만을 가지는 seed_map이 나옴.



            unclustered = torch.ones(mask.sum()).byte().cuda() # seedmap 에서 0.5 보다 큰 영역의 갯수만큼 백터를 만듬.

            instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

            instance_mask_lsy = torch.zeros(height, width)
            seed_map_lsy = seed_map[mask].view(1, -1).squeeze()

            instance_mask_lsy[mask.squeeze().cpu()] = seed_map_lsy.cpu()
            # save_image(instance_mask_lsy,
            #            'A:/220821_spatial_embedding/221007_NLB_semantic_both_segment_base_rgb/val_result/best_recon_iou/cluster_1/{}_seed_mask1.png'.format(
            #                count_12))

            while (unclustered.sum() > 128): #mask의 true가 나오는 영역이 128 영역이 넘으면 원래 128

                # instance_mask = torch.zeros(height, width)
                # instance_mask[unclustered.squeeze().cpu()] = seed_map[unclustered.squeeze().cpu()]

                recount = recount+1
                # print(recount)




                # at inference time, we use a trained seed map output to define the center of instance
                # embedding with the highest seed score become instance's center
                seed = (seed_map_masked * unclustered.float()).argmax().item() #index 반환

                seed_score = (seed_map_masked * unclustered.float()).max().item() #값 반환


                # In order for the embedding of a particular pixel to be a center,
                # the seed score at that pixel must be greater than 0.9 (to prevent all pixels from becoming centers)
                # print('seedscore: {}'.format(seed_score))
                if seed_score < threshold:  # ths=0.9
                    break

                # define instance center (embedding with the highest seed score & > ths)
                center = spatial_emb_masked[:, seed:seed + 1]

                unclustered[seed] = 0  # mask out
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)  # accompanying sigma (instance specific margin)

                # calculate gaussian output - e.q (5)
                # spatial_emb_masked : (2, n) / center : (2, 1)
                # dist : (1, n)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                          center, 2) * s, 0, keepdim=True))  # (1, N)



                # instance_mask = torch.zeros(height, width)
                # print(lsy_test.shape)
                # print(dist.shape)
                # lsy_test = dist.squeeze()
                # instance_mask[mask.squeeze().cpu()] = dist.squeeze().cpu()




                # instance_mask[mask.squeeze().cpu()] = lsy_test.cpu()

                # save_image(instance_mask,
                #            'A:/220821_plant_dataset/lsy_a4/images/save_test/spatial_emb/instance_mask{}.png'.format(recount))


                # e.q (11)
                proposal = (dist > 0.5).squeeze()









                # mask out all clustered pixels in the seed map, until all seeds are masked
                if proposal.sum() > 64: #원래 64



                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:

                        # for i in range(seed_map_lsy.shape[0]):
                        #     if proposal[i] != 0:
                        #         seed_map_lsy[i] = 0

                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                        instances.append(
                            {'mask': instance_mask.squeeze() * 255, 'score': seed_score})
                        count += 1
                        # instance_mask_lsy[mask.squeeze().cpu()] = seed_map[mask.squeeze().cpu()]

                        instance_mask_lsy[mask.squeeze().cpu()] = seed_map_lsy.cpu()

                        # save_image(instance_mask_lsy,
                        #            'A:/220821_spatial_embedding/221007_NLB_semantic_both_segment_base_rgb/val_result/best_recon_iou/cluster_1/{}_seed_mask{}.png'.format(
                        #                count_12,recount))




                unclustered[proposal] = 0
                # seed_map_lsy[proposal] = 0
                # print(unclustered.sum())

        if mask_2.sum() > 128:  # seed map에서 값의 합이 128을 넘는다면 실행됨.

            # only consider the pixels which belong to mask (RoI pixels)
            spatial_emb_masked = spatial_emb[mask_2.expand_as(spatial_emb)].view(2,
                                                                               -1)  # true가 나온 영역의 값만을 가지는 vector가 나옴.
            # print(spatial_emb.shape)
            # save_image(spatial_emb[0],'A:/220821_plant_dataset/lsy_a4/images/save_test/spatial_emb/spatial_emb1.png')
            # save_image(spatial_emb[1], 'A:/220821_plant_dataset/lsy_a4/images/save_test/spatial_emb/spatial_emb2.png')

            sigma_masked = sigma[mask_2.expand_as(sigma)].view(n_sigma,
                                                             -1)  # (1, N) # true가 나온 영역(seed map이 0.5보다 큰 영역)의 값만을 가지는 sigma가 나옴.

            seed_map_masked = seed_map_2[mask_2].view(1, -1)  # (1, N) true가 나오는 영역의 값만을 가지는 seed_map이 나옴.

            unclustered = torch.ones(mask_2.sum()).byte().cuda()  # seedmap 에서 0.5 보다 큰 영역의 갯수만큼 백터를 만듬.


            instance_map_masked_2 = torch.zeros(mask_2.sum()).byte().cuda()
            instance_mask_lsy = torch.zeros(height, width)
            seed_map_lsy = seed_map_2[mask_2].view(1, -1).squeeze()

            instance_mask_lsy[mask_2.squeeze().cpu()] = seed_map_lsy.cpu()
            # save_image(instance_mask_lsy,
            #            'A:/220821_spatial_embedding/221007_NLB_semantic_both_segment_base_rgb/val_result/best_recon_iou/cluster_2/{}_seed_mask1.png'.format(
            #                count_12))

            while (unclustered.sum() > 128):  # mask의 true가 나오는 영역이 128 영역이 넘으면 원래 128

                # instance_mask = torch.zeros(height, width)
                # instance_mask[unclustered.squeeze().cpu()] = seed_map[unclustered.squeeze().cpu()]

                recount_2 = recount_2 + 1
                # print(recount)

                # at inference time, we use a trained seed map output to define the center of instance
                # embedding with the highest seed score become instance's center
                seed = (seed_map_masked * unclustered.float()).argmax().item()  # index 반환

                seed_score = (seed_map_masked * unclustered.float()).max().item()  # 값 반환

                # In order for the embedding of a particular pixel to be a center,
                # the seed score at that pixel must be greater than 0.9 (to prevent all pixels from becoming centers)
                # print('seedscore: {}'.format(seed_score))
                if seed_score < threshold:  # ths=0.9
                    break

                # define instance center (embedding with the highest seed score & > ths)
                center = spatial_emb_masked[:, seed:seed + 1]

                unclustered[seed] = 0  # mask out
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)  # accompanying sigma (instance specific margin)

                # calculate gaussian output - e.q (5)
                # spatial_emb_masked : (2, n) / center : (2, 1)
                # dist : (1, n)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                          center, 2) * s, 0, keepdim=True))  # (1, N)

                # instance_mask = torch.zeros(height, width)
                # print(lsy_test.shape)
                # print(dist.shape)
                # lsy_test = dist.squeeze()
                # instance_mask[mask_2.squeeze().cpu()] = dist.squeeze().cpu()

                # instance_mask[mask.squeeze().cpu()] = lsy_test.cpu()

                # save_image(instance_mask,
                #            'A:/220821_plant_dataset/lsy_a4/images/save_test/spatial_emb/instance_mask{}.png'.format(recount))

                # e.q (11)
                proposal = (dist > 0.5).squeeze()

                # mask out all clustered pixels in the seed map, until all seeds are masked
                if proposal.sum() > 64:  # 원래 64

                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:

                        # for i in range(seed_map_lsy.shape[0]):
                        #     if proposal[i] != 0:
                        #         seed_map_lsy[i] = 0

                        instance_map_masked_2[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask_2.squeeze().cpu()] = proposal.cpu().byte()
                        instances_2.append(
                            {'mask': instance_mask.squeeze() * 255, 'score': seed_score})
                        count += 1
                        # instance_mask_lsy[mask.squeeze().cpu()] = seed_map[mask.squeeze().cpu()]

                        instance_mask_lsy[mask_2.squeeze().cpu()] = seed_map_lsy.cpu()

                        # save_image(instance_mask_lsy,
                        #            'A:/220821_spatial_embedding/221007_NLB_semantic_both_segment_base_rgb/val_result/best_recon_iou/cluster_2/{}_seed_mask{}.png'.format(
                        #                count_12, recount_2))

                unclustered[proposal] = 0
                # seed_map_lsy[proposal] = 0
                # print(unclustered.sum())





            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
            instance_map_2[mask_2.squeeze().cpu()] = instance_map_masked_2.cpu()

            instance_map_3[mask.squeeze().cpu()] = instance_map_masked.cpu()
            instance_map_3[mask_2.squeeze().cpu()] = instance_map_masked_2.cpu()



        return instance_map, instances, mask, instance_map_2, instances_2, mask_2, instance_map_3


class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)
