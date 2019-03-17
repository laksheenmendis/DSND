import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from math import floor


def load_checkpoint(filepath):
    '''method which loads the checkpoint file and generates model'''
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    arch = checkpoint['arch']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    width, height = img.size[0], img.size[1]
    
    size = 256, 256
    
    if width > height:
        ratio = float(width) / float(height)
        newheight = ratio * size[0]
        img = img.resize((size[0], int(floor(newheight))), Image.ANTIALIAS)
    else:
        ratio = float(height) / float(width)
        newwidth = ratio * size[0]
        img = img.resize((int(floor(newwidth)), size[0]), Image.ANTIALIAS)
    
    img_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    img_tensor = img_transform(img)
    
    return img_tensor
    