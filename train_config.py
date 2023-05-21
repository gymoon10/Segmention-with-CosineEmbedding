"""
Set training options.
"""
import copy
import os

import torch
from utils import transforms as my_transforms

#DATASET_DIR = 'C:/Users/iml/Desktop/ERFNet_ClassSegV3/Datas/citrus'
DATASET_DIR = 'C:/Users/iml/Desktop/ERFNet_ClassSegV3/Datas/corn'
# DATASET_DIR = 'C:/Users/iml/Desktop/ERFNet_ClassSeg/Datas/plant&disease'
# DATASET_DIR = 'A:/220821_plant_dataset/jabcho/custom2'

# -------- type_ 확인 -------------
name = 'Corn'

args = dict(
    cuda=True,
    display=False,
    display_it=5,

    save=True,
    save_dir='D:/EmbeddingBasedSeg/save/weight',
    save_dir1='D:/EmbeddingBasedSeg/save/result1',
    save_dir2='D:/EmbeddingBasedSeg/save/result2',
    save_dir3='D:/EmbeddingBasedSeg/save/sigma_offset',

    resume_path = None,

    train_dataset = {
        'name': name, 
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': 'train2',
            'size': None,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image', 'instance_plant', 'label_plant', 'instance_disease', 'label_disease', 'instance_all', 'label_all'],
                        'type': [torch.FloatTensor, torch.ByteTensor, torch.ByteTensor, torch.ByteTensor,
                                 torch.ByteTensor, torch.ByteTensor, torch.ByteTensor],
                    }
                },
            ]),
        },
        'batch_size': 2, 
        'workers': 0
    },

    val_dataset = {
        'name': name,
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': 'val2',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image', 'instance_plant', 'label_plant', 'instance_disease', 'label_disease', 'instance_all', 'label_all'],
                        'type': [torch.FloatTensor, torch.ByteTensor, torch.ByteTensor,torch.ByteTensor, torch.ByteTensor,torch.ByteTensor,torch.ByteTensor],
                    }
                },
            ]),
        },
        'batch_size': 1,
        'workers': 0
    },


    model = {
        'name': 'ERFNet_Semantic_Embedding',
        'kwargs': {
            'num_classes': 3,  # 3 for bg/plant/disease
        }
    },

    lr=0.00070,
    n_epochs=800,
)


def get_args():
    return copy.deepcopy(args)