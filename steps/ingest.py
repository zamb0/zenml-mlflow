import logging
import mlflow.data
import mlflow.data
import mlflow.data.dataset_source
import torchvision
import os
import mlflow
from typing import Tuple, Annotated
from zenml import step
from zen_client import experiment_tracker
from torchvision.datasets import ImageFolder

@step(experiment_tracker=experiment_tracker)
def ingest(root:str="hymenoptera_data") \
        -> Tuple[Annotated[ImageFolder, 'train'],
                Annotated[ImageFolder, 'val']]:
        
    logging.info('Ingesting data')
    
    train_path = os.path.join(root, 'train')
    val_path = os.path.join(root, 'val')
    
    train_dataset = ImageFolder(train_path)
    val_dataset = ImageFolder(val_path)
    
    return train_dataset, val_dataset
