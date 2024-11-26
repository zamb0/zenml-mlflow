import logging
import torchvision
import mlflow
from typing import Tuple, Annotated
from torchvision import transforms
from zenml import step
from zen_client import experiment_tracker

@step(experiment_tracker=experiment_tracker)
def transform(train_dataset: torchvision.datasets.ImageFolder, 
              val_dataset:torchvision.datasets.ImageFolder) \
        -> Tuple[Annotated[torchvision.datasets.ImageFolder, 'train'], 
                Annotated[torchvision.datasets.ImageFolder, 'val']]:
        
    logging.info('Transforming data')
    
    train_transformation = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    val_transformation = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset.transform = train_transformation
    val_dataset.transform = val_transformation
    
    #print the number of samples in the train and val dataset
    print('Number of samples in train_dataset: ', len(train_dataset))
    print('Number of samples in val_dataset: ', len(val_dataset))
    
    #print image size
    print('Image size: ', train_dataset[0][0].size())
    print('Image size: ', val_dataset[0][0].size())
    
    return train_dataset, val_dataset