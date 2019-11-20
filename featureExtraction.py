import torch
from torch import nn
import torchvision

import torch.optim as optim

import torch.nn as nn            # containing various building blocks for your neural networks
import torch.optim as optim      # implementing various optimization algorithms
import torch.nn.functional as F  # a lower level (compared to torch.nn) interface

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import glob
import os.path as osp
import numpy as np
from PIL import Image

import os
import skvideo
from skvideo import io as vp

model = torchvision.models.video.mc3_18(pretrained=True, progress=True)

for name,param in model.named_parameters():
    param.requires_grad = False
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

class Cichlids(Dataset):
    """
    A customized data loader for Cichlids.
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
            # load videos
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
        
        if self.transform is not None:
            clip = [self.transform(img) for img in video]
        video = torch.stack(clip, 0).permute(0, 2, 1, 3)
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            video = video.reshape((video.shape[1], video.shape[2], video.shape[3], video.shape[0]))
            clip = [self.spatial_transform(img) for img in video]
            video = torch.stack(clip, 0).permute(0, 2, 1, 3)
        
        # return image and label
        return video, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)

scales = [1.0]
for i in range(1, 5):
    scales.append(scales[-1] * 0.84089641525)

spatial_transform = Compose([
            MultiScaleRandomCrop(scales, 112),
            RandomHorizontalFlip(),
            ToTensor(), Normalize([0, 0, 0], [1, 1, 1])
        ])

trainset = Cichlids(
    root='MLclips/training',
    preload=False, spatial_transform=None, transform=transforms.ToTensor()
)

trainset_loader = DataLoader(trainset, batch_size=3, shuffle=True, num_workers=3)

# Load the testset
testset = Cichlids(
    root='MLclips/testing',
    preload=False, spatial_transform= None, transform=transforms.ToTensor()
)

# Use the torch dataloader to iterate through the dataset
testset_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda:1" if use_cuda else "cpu")

import torch.nn.functional as F
from time import time

def train_model(epochs=5, log_interval=1000):
    torch.cuda.empty_cache()
    model.to(device)
    model.train()
    for t in range(epochs):
        start = time()
        iteration = 0
        for batch_idx, (data, target) in enumerate(trainset_loader):
            
            data, target = data.to(device), target.to(device)
            
            data = data.float()

            output = model(data)
            
            loss = F.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    t, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
            
        end = time()
        print('Time taken for this epoch: {:.2f}s'.format(end-start))
        check_accuracy() # evaluate at the end of epoch
    torch.cuda.empty_cache()

def check_accuracy():
    num_correct = 0
    num_samples = 0
    test_loss = 0
    correct = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            data = data.float()
            output = model(data)
            test_loss += F.cross_entropy(output, target) # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))
    
train_model()

# for state in optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()