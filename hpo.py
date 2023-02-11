#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import os
import sys
import logging

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs=model(inputs)
        test_loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += test_loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    logger.info(f"Test set: Average loss: {total_loss}, Testing Accuracy: {100*total_acc}")

    
def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item() 
                running_samples+=len(inputs)
                
                if running_samples % 512  == 0:
                    running_acc = running_corrects / running_samples

                    logger.info("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        running_samples,
                        len(image_dataset[phase].dataset),
                        100.0 * (running_samples / len(image_dataset[phase].dataset)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0*running_acc,
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                #if running_samples>(0.2*len(image_dataset[phase].dataset)):
                #    break
                    

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            logger.info("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        running_samples,
                        len(image_dataset[phase].dataset),
                        100.0 * (running_samples / len(image_dataset[phase].dataset)),
                        epoch_loss,
                        running_corrects,
                        running_samples,
                        100.0*epoch_acc,
                        )
                    )
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1      
            
            
        if loss_counter==1:
            break
            
    return model

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model
    


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Getting data loaders")
    
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    valid_path=os.path.join(data, 'valid')
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        ])
                                                            
    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        ])

    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               shuffle=True)

    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    test_loader  = torch.utils.data.DataLoader(testset,
                                               batch_size=batch_size,
                                               shuffle=False)

    validation_data = torchvision.datasets.ImageFolder(root=valid_path, transform=transform_test)
    validation_loader  = torch.utils.data.DataLoader(validation_data,
                                                     batch_size=batch_size,
                                                     shuffle=False)
    
    return train_loader, test_loader, validation_loader

    

def main(args):  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    train_loader, test_loader, validation_loader=create_data_loaders(args.data_dir, args.batch_size)        
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model=model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''   
    logger.info("Training...")
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing...")
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving model...")
    torch.save(model, os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser=argparse.ArgumentParser()
        
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    
    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    
    args=parser.parse_args()    
    
    main(args)
