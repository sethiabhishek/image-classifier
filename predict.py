import argparse
import PIL
import numpy as np
import json

# Torch libraries
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Arg Parse
parser = argparse.ArgumentParser(description='Udacity Image Classifier Project',)
parser.add_argument('--image_path', dest="image_path", default = "flowers/test/1/image_06743.jpg", type=str)
parser.add_argument('--checkpoint', dest="checkpoint", default = "checkpoint_2.pth", type=str)
parser.add_argument('--topk', dest="top_k", default = 5, type=int)
parser.add_argument('--category_names', dest="category_names", default = "cat_to_name.json", type=str)
parser.add_argument('--gpu', action="store_true", dest="gpu", default=False)
args = parser.parse_args()

# Create variables for parameters
image_path = args.image_path
checkpoint = args.checkpoint
top_k = args.top_k
gpu = args.gpu
category_names = args.category_names

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    # Set up pre trained network
    checkpoint = torch.load(filepath)
    # Choose pretrained model architecture
    model = None
    try :
        arch = checkpoint["arch"]
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
    # Load in checkpoint
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer_state']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Set up model
    model.eval()
    class_to_idx = model.class_to_idx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process Image
    img_array = process_image(image_path)
    tensor_type = torch.FloatTensor
    if torch.cuda.is_available() and gpu:
        images = tensor_type = torch.cuda.FloatTensor
    img_tensor = torch.from_numpy(img_array).type(tensor_type)
    img_tensor = img_tensor.unsqueeze_(0)

    # Predict Image
    with torch.no_grad():
        output = model.forward(img_tensor)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)

    # Convert tensors to numpy arrays
    top_class_array = top_class.cpu().data.numpy()[0]
    top_p_array = top_p.cpu().data.numpy()[0]

    # Remap Image Predictions
    idx_to_class = {x: y for y, x in class_to_idx.items()}
    mapped_top_class_array = []
    for i in top_class_array:
        mapped_top_class_array += [idx_to_class[i]]

    return top_p_array, mapped_top_class_array


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = PIL.Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    img_transform = transform(pil_image)

    img_array = np.array(img_transform)
    return img_array

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

# Console outputs
device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
print(f'Predicting Image: {image_path} in {device} mode')
print(f'Using Checkpoint: {checkpoint}')
print(f'Category file: {category_names}')

# Run prediction
model = load_checkpoint(checkpoint)
model.to(device)
probs, classes = predict(image_path, model, top_k)

# load in JSON categories
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Print results
print('\nProbabilities:')
flower_cat = []
for i in classes:
    flower_name = cat_to_name[f'{i}']
    flower_cat.append(flower_name)
for i in range(len(flower_cat)):
    print(f'{flower_cat[i]} {"%.4f" % probs[i]}')