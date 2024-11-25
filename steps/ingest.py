import logging
import torchvision
import os
from typing import Tuple, Annotated
from dataset import ImageFolder
from zenml import step


@step
def ingest(root:str="hymenoptera_data") -> Tuple[Annotated[torchvision.datasets.ImageFolder, 'train'], Annotated[torchvision.datasets.ImageFolder, 'val']]:
    logging.info('Ingesting data')
    
    train_path = os.path.join(root, 'train')
    val_path = os.path.join(root, 'val')
    
    return ImageFolder(train_path), ImageFolder(val_path)
