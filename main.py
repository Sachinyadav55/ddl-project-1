import torch
from torch import nn
import torchvision

import torch.optim as optim

import torch.nn as nn            
import torch.optim as optim      
import torch.nn.functional as F  

import torchvision.transforms as transforms
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import glob
import os.path as osp
import numpy as np
from PIL import Image

import os
import skvideo
from skvideo import io as vp
from time import time

model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
    
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)

model = model.cuda()
model = nn.DataParallel(model, device_ids=None)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)

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
                
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            video = video.reshape((video.shape[1], video.shape[2], video.shape[3], video.shape[0]))
            clip = [self.spatial_transform(Image.fromarray(img)) for img in video]
            video = torch.stack(clip, 0).permute(1, 0, 2, 3)
        
        # return video and label
        return video, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop,
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
    preload=False, spatial_transform=spatial_transform, transform=None
)

trainset_loader = DataLoader(trainset, batch_size=3, shuffle=True, num_workers=6)

# Load the testset
testset = Cichlids(
    root='MLclips/testing',
    preload=False, spatial_transform= spatial_transform, transform=None
)

testset_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

results = 'r3d_18FTweights'
if not os.path.isdir(results):
    os.mkdir(results)

def train_model(epochs=50, log_interval=1000):    
    model.train()
    for t in range(epochs):
        start = time()
        iteration = 0
        avg_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(trainset_loader):
            
            target = target.cuda(async=True)
            
            data = Variable(data)
            target = Variable(target)
            
            output = model(data)
            
            lossFunction = nn.CrossEntropyLoss()
            lossFunction = lossFunction.cuda()
            
            loss = lossFunction(output, target)
            avg_loss += loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    t, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        end = time()
        print ('\nSummary: Epoch {}'.format(t))
        print('Time taken for this epoch: {:.2f}s'.format(end-start))
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        avg_loss/len(trainset_loader.dataset), correct, len(trainset_loader.dataset),
        100. * correct / len(trainset_loader.dataset)))
        
        save_file_path = os.path.join(results,
                                      'save_{}.pth'.format(t))
        states = {
            'epoch': t + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
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
            target = target.cuda(async=True)
    
            data = Variable(data)
            target = Variable(target)

            output = model(data)
            
            lossFunction = nn.CrossEntropyLoss()
            lossFunction = lossFunction.cuda()
            
            test_loss += lossFunction(output, target) # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(testset_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))
    
train_model()