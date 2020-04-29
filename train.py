#Imports

import argparse

import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms, models

import json
import os


def parse():
    """ enable inputs """
    
    parser = argparse.ArgumentParser(description='Train a nueronal network.')
    parser.add_argument('--data_dir', default='flowers', help='Name a data drectory')
    parser.add_argument('--arch', default='vgg19', help='Choose a model: vgg19, densenet, alexnet')
    parser.add_argument('--hidden_units', type=int, default=512, help='Set the number of hidden nodes')
    parser.add_argument('--learning_rate', type= float, default=0.01, help='Set the learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Set the number of epochs')
    parser.add_argument('--gpu', action='store_true', help='enable cuda when used')                        
    parser.add_argument('--save_dir', default='.', help='Name a driectory to save the model')
    
    args = parser.parse_args()
    
    return args
    

def load_data(data_dir):
    """ transform datasets and define dataloaders """
    
    train_dir = args.data_dir + '/train'  
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    
    #transform
    train_transforms =  transforms.Compose([transforms.CenterCrop(224),
                                            transforms.RandomRotation(45),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    #load datasets
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    
    #define dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    
    return trainloader, testloader, validloader, train_data
    
    
def build_model(arch, hidden_units, gpu, learning_rate):
    """ create new model from pretrained model + new classifier """
    
    #load pretrained model
    if args.arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = 25088
    elif args.arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
       
    
    #enable switching to GPU 
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    
    #Freeze Features
    for param in model.parameters():
        param.requiers_grad = False
        
    #calculate number of output nodes based on number of labels from folders 
    folders=0
    for _, dirnames, filenames in os.walk(train_dir):
        folders += len(dirnames) 
    output_size = folders

    #Build new Classifier
    model.classifier = nn.Sequential(nn.Linear(input_size, args.hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(p = 0.2),
                                    nn.Linear(args.hidden_units, 1568),
                                    nn.ReLU(),
                                    nn.Dropout(p = 0.2),
                                    nn.Linear(1568, output_size),
                                    nn.LogSoftmax(dim = 1))

    #Define Loss and Optimizer
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
    
    #Switch to GPU
    model.to(device)
    
    return criterion, optimizer, model, device, input_size, output_size
    
    
    
def train_model(model, optimizer, criterion, trainloader, validloader, device, epochs):
    """ train new model to predict flower classes """
    
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 20

    #Loop Train
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            #Move to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            #Train
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #Update Training Loss
            running_loss += loss.item()

            #Validation
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = model.forward(inputs)
                        loss = criterion(outputs, labels)

                        valid_loss += loss.item()

                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Train Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(valid_loss/len(validloader)),                 
                      "Validation Accuracy: {}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()
    return model


def save_checkpoint(train_data, model, input_size, output_size, arch, optimizer, epochs, save_dir):
    """ save new model for later """
    
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'pretrained_model': args.arch,
                  'input_layer': input_size,
                  'output_layer': output_size,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'epochs': args.epochs,
                  'optimizer_state': optimizer.state_dict(),
                  'mapped_classes': model.class_to_idx}             

    torch.save(checkpoint, args.save_dir)
    return None

    
    
def main():
    global args
    args = parse()
    load_data()
    build_model()
    train_model()
    save_checkpoint()
    
    print('Done!')   
    
    
    
if __name__ == '__main__':
    main()
    
    
