import logging
import os
from typing import Tuple, Annotated
from zenml import step
from torchvision.datasets import ImageFolder

@step(enable_cache=True)
def ingest(root:str="hymenoptera_data") \
        -> Tuple[Annotated[ImageFolder, 'train'],
                 Annotated[ImageFolder, 'val']]:
            
    """
    Ingests data from a given path

    Args:
        root: str: Path to the data
            
    Returns:
        Tuple[ImageFolder, ImageFolder]: Dataset object
    """
        
    logging.info('Ingesting data')
    
    train_path = os.path.join(root, 'train')
    val_path = os.path.join(root, 'val')
    
    train_dataset = ImageFolder(train_path)
    val_dataset = ImageFolder(val_path)
    
    return train_dataset, val_dataset
