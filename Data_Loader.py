from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from skimage import io
from torch.utils.data import Dataset
import random
import numpy as np


class Images_Dataset(Dataset):
    """Class for getting data as a Dict
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images

    Output:
        sample : Dict of images and labels"""

    def __init__(self, images_dir, labels_dir):

        self.labels_dir = labels_dir
        self.images_dir = images_dir

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):

        for i in range(len(self.images_dir)):
            image = io.imread(self.images_dir[i])
            label = io.imread(self.labels_dir[i])
            sample = {'images': image, 'labels': label}

        return sample


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir,):
        self.images = os.listdir(images_dir)
        self.labels = os.listdir(labels_dir)
        self.images.sort(key=lambda x:int(x[:-4]))
        self.labels.sort(key=lambda x:int(x[:-4]))
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.tx = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.RandomVerticalFlip(),
            #torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Normalize(mean=[0.5,], std=[0.5,])
            ])

        self.lx = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5,], std=[0.5,])
            ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i])
        l1 = Image.open(self.labels_dir + self.labels[i])

        img = self.tx(i1)
        label = self.lx(l1)

        return img, label

