"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import shutil
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image

import train_config
# from criterions.region_contrastV2 import Criterion
from criterions.pixel_contrast_intra import Criterion

from datasets import get_dataset
from models import get_model
from utils.utils_cvppp import AverageMeter, Cluster, Logger, Visualizer

torch.backends.cudnn.benchmark = True

args = train_config.get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    if not os.path.exists(args['save_dir1']):
        os.makedirs(args['save_dir1'])
    if not os.path.exists(args['save_dir2']):
        os.makedirs(args['save_dir2'])
    if not os.path.exists(args['save_dir3']):
        os.makedirs(args['save_dir3'])
if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# train dataloader
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# val dataloader
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=False, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
if args['model']['name'] == 'branched-erfnet' or args['model']['name'] == 'branched-erfnet2':
    model.init_output(args['loss_opts']['n_sigma'])
model = torch.nn.DataParallel(model).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
print('# params :', count_parameters(model))


# set criterion
criterion = Criterion()
criterion = torch.nn.DataParallel(criterion).to(device)

# set optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args['lr'], weight_decay=1e-4)


def lambda_(epoch):
    return pow((1 - ((epoch) / args['n_epochs'])), 0.9)


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda_, )


# Logger
logger = Logger(('train', 'val', 'iou_plant', 'iou_disease'), 'loss')

# resume
start_epoch = 0

best_iou_plant = 0
best_iou_disease = 0
best_iou_both = 0

if args['resume_path'] is not None and os.path.exists(args['resume_path']):
    print('Resuming model from {}'.format(args['resume_path']))
    state = torch.load(args['resume_path'])
    start_epoch = state['epoch'] + 1
    best_iou_both = state['best_iou_both']
    best_iou_disease  = state['best_iou_disease']
    model.load_state_dict(state['model_state_dict'], strict=True)
    optimizer.load_state_dict(state['optim_state_dict'])
    logger.data = state['logger_data']


def train(epoch):
    # define meters
    loss_meter = AverageMeter()
    intra_loss_meter = AverageMeter()
    inter_loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()

    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataset_it)):

        im = sample['image']  # (N, 3, 512, 512)
        instance_all = sample['instance_all'].squeeze(1)  # GT-class label mask (N, H, W)
        instance_all = instance_all.type(torch.int64)

        # embedding: (N, 32, 512, 512)
        # output: (N, 3, 512, 512)
        embedding, output = model(im)
        #print('embedding :', embedding.shape)
        #print('output :', output.shape)
        #print('instance_all :', instance_all.shape)

        loss = criterion(embedding, output, instance_all)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())

    return loss_meter.avg


def val(epoch):
    # define meters
    loss_meter = AverageMeter()
    recon_iou1, recon_iou2 = AverageMeter(), AverageMeter()

    # put model into eval mode
    model.eval()
    img_id = 0

    with torch.no_grad():

        for i, sample in enumerate(tqdm(val_dataset_it)):

            img_id += 1

            im = sample['image']  # (N, 3, 512, 512)
            instances_all = sample['instance_all'].squeeze(1)
            instances_all = instances_all.type(torch.int64)

            # embedding: (N, 32, 512, 512)
            # output: (N, 3, 512, 512)
            embedding, output = model(im)

            loss = criterion(embedding, output, instances_all,
                             iou=True, meter_plant=recon_iou1, meter_disease=recon_iou2)
            loss = loss.mean()
            loss_meter.update(loss.item())

            pred = output[0, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32)
            pred = pred.cpu().detach().max(dim=1)[1]

            pred_plant = (pred == 1).numpy()
            pred_disease = (pred == 2).numpy()

            Image.fromarray(pred_plant[0]).save(os.path.join(args['save_dir1'], '%d_pred.png' % img_id))
            Image.fromarray(pred_disease[0]).save(os.path.join(args['save_dir2'], '%d_pred.png' % img_id))

    return loss_meter.avg, recon_iou1.avg, recon_iou2.avg


def save_checkpoint(epoch, state, recon_best1,recon_best2, recon_best3, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if recon_best1:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_recon_iou_model_%d.pth'%(epoch)))

    if recon_best2:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_recon2_iou_model_%d.pth'%(epoch)))

    if recon_best3:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_recon_both_iou_model_%d.pth' % (epoch)))


for epoch in range(start_epoch, args['n_epochs']):
    print('Starting epoch {}'.format(epoch))

    train_loss = train(epoch)
    scheduler.step()
    val_loss, val_iou_plant, val_iou_disease = val(epoch)

    print('===> train loss: {:.5f}'.format(train_loss))
    print('===> val loss: {:.5f}, val iou plant: {:.5f}, val iou disease: {:.5f}'.format(val_loss, val_iou_plant, val_iou_disease))

    logger.add('iou_plant', val_iou_plant)
    logger.add('iou_disease', val_iou_disease)

    logger.add('train', train_loss)
    logger.add('val', val_loss)
    logger.plot(save=args['save'], save_dir=args['save_dir'])

    recon_best_plant = val_iou_plant > best_iou_plant
    best_iou_plant = max(val_iou_plant, best_iou_plant)

    recon_best_disease = val_iou_disease > best_iou_disease
    best_iou_disease = max(val_iou_disease, best_iou_disease)

    val_iou_both = (val_iou_plant + val_iou_disease) / 2

    recon_best_both = val_iou_both > best_iou_both
    best_iou_both = max(val_iou_both, best_iou_both)

    if args['save']:
        state = {
            'epoch': epoch,
            'best_iou_both': val_iou_both,
            'best_iou_disease': val_iou_disease,
            'best_iou_plant': val_iou_plant,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'logger_data': logger.data
        }
        save_checkpoint(epoch, state, recon_best_plant, recon_best_disease, recon_best_both)
