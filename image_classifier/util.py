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
    
    # Find pixels to crop on to create 224x224 image
    center = width/4, height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    cropped_image = img.crop((left, top, right, bottom))

    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(cropped_image)/255 # Divided by 255 because imshow() expects integers (0:1)!!

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image
    