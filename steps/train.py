import logging
import torch
import torchvision
import mlflow
from zenml import step
from src.model import CNN
from typing import Annotated
from zen_client import experiment_tracker
from config import Config

@step(experiment_tracker=experiment_tracker)
def train(train_dataset: torchvision.datasets.ImageFolder, 
          val_dataset:torchvision.datasets.ImageFolder) \
        -> Annotated[CNN, 'model']:
            
    logging.info('Training model')
    mlflow.log_param('batch_size', Config.batch_size)
    mlflow.log_param('lr', Config.lr)
    mlflow.log_param('momentum', Config.momentum)
    mlflow.log_param('gamma', Config.gamma)
    mlflow.log_param('step_size', Config.step_size)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    
    CNN_model = CNN(num_classes=2)
    
    optimizer = torch.optim.SGD(CNN_model.model.parameters(), lr=Config.lr, momentum=Config.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.gamma, gamma=Config.gamma)

    accuracy = CNN_model.train_model(train_loader, 
                    val_loader, 
                    criterion=torch.nn.CrossEntropyLoss(), 
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_epochs=1, 
                    dataset_sizes=[len(train_dataset), len(val_dataset)], device="cpu")
    
    mlflow.log_metric('accuracy', accuracy)
    mlflow.pytorch.log_model(CNN_model.model, "model")
    
    return CNN_model
    