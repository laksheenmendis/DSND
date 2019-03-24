import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
from util import load_checkpoint, process_image
import json

parser = argparse.ArgumentParser(description='Train a new network on a dataset')
parser.add_argument('path', type=str, default='flowers/test/101/image_07952.jpg', help='Path to image file')
parser.add_argument('--top_k', type=int, default=0, help='Number of top probabilities required')
parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='Path to checkpoint file')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Name of the file which contains class to category name mapping')
parser.add_argument('--gpu', type=bool, default=True, help='Whether GPU should be used when inferencing')
args=parser.parse_args()


image_path = args.path

model = load_checkpoint(args.checkpoint)

device = 'cuda' if args.gpu else 'cpu'

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

def predict(image_path, model, topk=5, device='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file 
    image = process_image(image_path)
   
    model.to(device)
    model.eval()
    
    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(image, axis=0)).type(torch.FloatTensor).to("cpu")
    
    with torch.no_grad():
        output = model(torch_image.to(device).float())
        
        #gets the top 5 probabilities
        probs, labels = torch.topk(output, topk)        
        probs = probs.exp()
        
        indexes = {model.class_to_idx[k]: k for k in model.class_to_idx}

        classes = []
        
        for label in labels.to('cpu').numpy()[0]:
            classes.append(indexes[label])

        return probs.to('cpu').numpy()[0], classes

   

if args.top_k:
    probs, classes = predict(image_path, model, args.top_k, device)
    print('Probabilities of top {} flowers:'.format(args.top_k))
    
    class_names = []
    for index in classes:
        class_names.append(cat_to_name[index])
    
    for i in range(args.top_k):
        print('{} : {:.2f}'.format(class_names[i],probs[i]))
else:
    probs, classes = predict(image_path, model)
    print('Flower is predicted to be {} with a probability of {:.2f}'.format(cat_to_name[classes[0]], probs[0]))
    
    