#Imports
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms, models

from PIL import Image

import json


def parse():
    """ enable inputs """
    
    parser = argparse.ArgumentParser(description='Predict with nueronal network')
    parser.add_argument('--image_input', default='flowers/test/102/image_08004.jpg', help='Path of image to classify.')
    parser.add_argument('--check', default='.', help='Specify the path to the checkpoint file.')
    parser.add_argument('--label_mapping', default='cat_to_name.json', help='Specify label mapping')
    parser.add_argument('--topk', type=int, default=5, help='Number of classes and probabilities to predict')                      
    args = parser.parse_args()
    
    return args


def load_checkpoint(check):
    '''loads previous model'''
    
    checkpoint = torch.load(check)
    pretrained_model = checkpoint['pretrained_model']
    
    if args.arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = 25088
    elif args.arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    epochs = checkpoint['epochs']
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    model.class_to_idx = checkpoint['mapped_classes']
    
    for param in model.parameters():
        param.requiers_grad = False
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array'''

    im = Image.open(image_path)
    
    transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])])
    
    return transform(im) 


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

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    
    processed_image = process_image(image_path)
    model.to(device)
    
    with torch.no_grad():
        processed_image = processed_image.unsqueeze_(0).to(device)
               
        output = model.forward(processed_image)
            
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim = 1)
        
        probs, classes = top_p.data.cpu().numpy()[0], top_class.data.cpu().numpy()[0]
        
        idx_to_class = dict(map(reversed, model.class_to_idx.items()))

        classes = [idx_to_class[classes[i]] for i in range(classes.size)]

        return probs, classes
    
    
    
def main(model, check, label_mapping):
    
    model = load_checkpoint(check)
    print(model)
    
    predict() 
    
    
    with open(label_mapping, 'r') as f:
        cat_to_name = json.load(f)
        
    name_classes = [cat_to_name[i] for i in classes] 

    #Image
    imgg = Image.open(image_path)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(imgg)

    #Barplot
    y_pos = np.arange(len(classes)) 
    plt.barh(y_pos, probs)
    plt.yticks(y_pos, name_classes)
    plt.show()


if __name__ == '__main__':
    main()
