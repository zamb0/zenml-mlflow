import logging
import torch
import torchvision
from torchvision import models
from zenml import step
from src.model import CNN

@step
def train(train_dataset: torchvision.datasets.ImageFolder, val_dataset:torchvision.datasets.ImageFolder) -> torch.nn.Module:
    logging.info('Training model')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    CNN_model = CNN(num_classes=2)
    
    optimizer = torch.optim.SGD(CNN_model.model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    CNN_model.train(train_loader, 
                    val_loader, 
                    criterion=torch.nn.CrossEntropyLoss(), 
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_epochs=25, 
                    dataset_sizes=[len(train_dataset), len(val_dataset)], device="cpu")
    