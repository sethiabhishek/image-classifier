import argparse
import PIL
import numpy as np
import sys, getopt
import json
import os

# Torch libraries
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Arg Parsing
parser = argparse.ArgumentParser(description='Udacity Image Classifier Project',)
parser.add_argument('--data_dir', dest="data_dir", default="flowers", type=str)
parser.add_argument('--save_dir', dest="save_dir", default="checkpoints", type=str)
parser.add_argument('--learning_rate', dest="learning_rate", default=0.001, type=float)
parser.add_argument('--hidden_units', dest="hidden_units", default=1024, type=int)
parser.add_argument('--epochs', action="store", dest="epochs", default=20, type=int)
parser.add_argument('--gpu', action="store_true", dest="gpu", default=True)
parser.add_argument('--topk', action="store", dest="top_k", default=5, type=bool)
parser.add_argument('--arch', action="store", dest="arch",default="vgg16", type=str)
parser.add_argument('--checkpoint_name', action="store", dest="checkpoint_name", default="checkpoint_2.pth", type=str)
args = parser.parse_args()

# Create variables for parameters
data_dir = args.data_dir
save_dir = args.save_dir
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu
top_k = args.top_k
arch = args.arch
checkpoint_name = args.checkpoint_name

# Set device to GPU CPU
device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
# Set up data directories
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
checkpoint_dir = save_dir + '/' + checkpoint_name
# Set up transformations
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# Open json file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Choose pretrained model architecture
model = None
try :
    if arch == "vgg11":
        model = models.vgg11(pretrained=True)
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    if arch == "vgg19":
        model = models.vgg19(pretrained=True)
except:
    model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
# Set up classifier
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = classifier
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

steps = 0
train_losses, test_losses = [], []

print(f'Model Information:')
print(f'Architecture: {arch}')
print(f'Hidden Units: {hidden_units}')
print(f'Epochs: {epochs}')
print(f'Learning Rate: {learning_rate}')
print(f'TopK: {top_k}')
print(f'Checkpoint Name: {checkpoint_name}')
print(f'Beginning training. Training model in {device} mode....')

for e in range(epochs):
    running_loss = 0

    for images, labels in iter(trainloader):

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            # load in images from the validation set
            for ii, (images, labels) in enumerate(validloader):
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                # get the predicted classifications
                top_p, top_class = ps.topk(1, dim=1)
                # get a tensor of the probabilities seeing if the classification was correct or not
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        # get values for the training and test loss
        train_losses.append(running_loss/len(validloader))
        test_losses.append(test_loss/len(validloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))


# Save the checkpoint
model.class_to_idx = train_data.class_to_idx
checkpoint = {'input_size': 25088,
              'hidden_units':hidden_units,
              'output_size': 102,
              'arch':arch,
              'epochs': epochs,
              'classifier': model.classifier,
              'optimizer':optimizer,
              'optimizer_state': optimizer.state_dict,
              'class_to_idx': train_data.class_to_idx,
              'state_dict': model.state_dict()}

torch.save(checkpoint, checkpoint_dir)
print(f'Training Complete! Checkpoint saved as {checkpoint_name}')