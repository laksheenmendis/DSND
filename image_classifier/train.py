import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Train a new network on a dataset')
parser.add_argument('data_directory', type=str, default='flowers', help='Directory which contains data')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='Network architecture of the model (vgg16 or densenet121)')
parser.add_argument('--learning_rate', type=str, default=0.001, help='Learning rate to be used when training the network')
parser.add_argument('--hidden_units', type=list, default=[4096,2048], help='List of number of units in the hidden layers')
parser.add_argument('--epochs', type=str, default=5, help='Number of epochs for training the model')
parser.add_argument('--gpu', type=bool, default=True, help='Whether training of the model should be done in GPU or not')
args=parser.parse_args()

data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define the transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

# Define the dataloaders, using the image datasets and the trainforms
trainloader = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_datasets,batch_size=64,shuffle=True)
testloader = torch.utils.data.DataLoader(test_datasets,batch_size=64,shuffle=True)

# choose the device to train the model based on input arguments
device = "cuda" if args.gpu else "cpu"

# retreive the required trained model (vgg16 or densenet121)
model = getattr(models, args.arch)(pretrained=True)

# freeze the features of the model
for param in model.parameters():
    param.requires_grad = False
    
output_size = 102
input_size = 25088 if args.arch == 'vgg16' else 1024
hidden_sizes = args.hidden_units

#untrained feed-forward network as a classifier, using ReLU activations and dropout
classifier = nn.Sequential(nn.Linear(input_size,hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Dropout(p=0.3),
                          nn.Linear(hidden_sizes[0],hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Dropout(p=0.3),
                          nn.Linear(hidden_sizes[1],output_size),
                          nn.LogSoftmax(dim=1))

model.classifier = classifier

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(device)

#training the classifier
epochs = args.epochs
count = 0
running_loss = 0
print_every = 100

for e in range(epochs):
    for images, labels in trainloader:
        
        count += 1
        
        #changing to training mode
        model.train()
        
        #moving images and labels to relevant device
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if count % print_every == 0 :
            validation_loss = 0
            accuracy = 0
            
            #changing to evaluation mode
            model.eval()
            
            with torch.no_grad():
                for images, labels in validationloader:
                    
                    #moving images and labels to relevant device
                    images, labels = images.to(device), labels.to(device)
                    
                    logps = model.forward(images)
                    batch_loss = criterion(logps,labels)
                    validation_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_k, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            
# validation on the test set
test_loss = 0
accuracy = 0
total = 0
correct = 0
            
# changing to evaluation mode
model.eval()
            
# Turn off gradients for validation, saves memory and computations
with torch.no_grad():
    for images, labels in testloader:
                    
        #moving images and labels to relevant device
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        max_value, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    accuracy = (correct*100)/total
    print("Test accuracy: {:.3f}".format(accuracy))

# Save the checkpoint 
checkpoint = {'classifier': classifier,
              'arch': 'vgg16',
              'epochs': 5,
              'optimizer_state': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': train_datasets.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')
