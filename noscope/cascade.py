#!/usr/bin/env python


import torch
import torchvision
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Transformations needed for the input

input_size =224
data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# Loading the training and the testing datasets, with transformations applied

trainset =  datasets.ImageFolder(root='Accelerator/0004_vid_cropped/temp',
                                           transform=data_transform)
testset =  datasets.ImageFolder(root='Accelerator/val',
                                           transform=data_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=16)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=16)
classes = ('0','1') # Since binary classification


# Defining the neural networks, as paper suggested, started with a alexnet and alexnet-like models

class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Removing the last layer from Alexnet
class AlexNet_modified(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet_modified, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=7, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(192 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Selecting the GPU, default device is assumed to be "cuda:0"

device_id = "cuda:0"
device = torch.device( device_id if torch.cuda.is_available() else "cpu")

# selecting the model to train on 

selected_model = AlexNet_modified()
# Testing the Alexnet-like model first, loading the model onto the device 'device_id'

net = selected_model
net =net.to(device)


# Defining loss function and the optimizer, switched to Adam optimizer for faster convergence

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# Training for 10 epochs, num_epochs control the number of epochs

num_epochs = 10
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for data in tqdm(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print(str(epoch+1) + "/10")
    print("            Loss:" + str(loss.item()))
    

print('Finished Training')


PATH = './alnet_m.pth'
torch.save(net.state_dict(), PATH)


## Testing it now on the testing dataset

# Loading the images on to the device and checking the ground truth for first 8 images

num_to_show = 8
dataiter = iter(testloader)
images, labels = dataiter.next()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(num_to_show)))
images, labels = images.to(device), labels.to(device)


# Loading the saved checkpoint

net = selected_model
net =net.to(device)
net.load_state_dict(torch.load(PATH))


# Starting inference on the test dataset

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(num_to_show)))


# Calculating the accuracy

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# Calculating the accuracy of each class individually

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))



