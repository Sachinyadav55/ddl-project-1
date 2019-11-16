#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn            
import torch.optim as optim      
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import glob
import os.path as osp
import numpy as np
from PIL import Image

import os, sys
import skvideo
from skvideo import io as vp

class Cichlids(Dataset):
    """
    A customized data loader for Cichlids data
    """
    def __init__(self,
                 root,
                 transform=None,
                 spatial_transform=None,
                 preload=False):
        """ Intialize the Cichlids dataset
        
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        self.videos = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.spatial_transform = spatial_transform

        # read filenames
        for i, class_dir in enumerate(os.listdir(root)):
            filenames = glob.glob(osp.join(root, class_dir, '*.mp4'))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair
                
        # if preload dataset into memory
        if preload:
            self._preload()
            
        self.len = len(self.filenames)
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []
        for image_fn, label in self.filenames:            
            # load images
            video = vp.vread(image_fn)
            video = np.reshape(video, (video.shape[3], video.shape[0], video.shape[1], video.shape[2]))
            self.videos.append(video.copy())
            self.labels.append(label)

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.videos is not None:
            # If dataset is preloaded
            video = self.videos[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            video_fn, label = self.filenames[index]
            video = vp.vread(video_fn)
            video = np.reshape(video, (video.shape[3], video.shape[0], video.shape[1], video.shape[2]))
            
        # May use transform function to transform samples
        if self.transform is not None:
            clip = [self.transform(img) for img in video]
        video = torch.stack(clip, 0).permute(0, 2, 1, 3)
        
        # spatial transformations - under development 
        # if self.spatial_transform is not None:
        #     self.spatial_transform.randomize_parameters()
        #     clip = [self.spatial_transform(img) for img in video]
        #     video = torch.stack(clip, 0).permute(0, 2, 1, 3)
        
        # return image and label
        return video, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len