import torchvision
import os
import logging
from zenml import step
from typing import Annotated
from torchvision.datasets import ImageFolder

@step(enable_cache=False)
def dynamic_importer(root:str="hymenoptera_data") \
        -> Annotated[torchvision.datasets.ImageFolder, 'data']:
    """
    Ingests data from a given path
    
    Args:
        root: str: Path to the data
        
    Returns:
        ImageFolder: Dataset object
    """
        
    logging.info('Ingesting data')
    
    data_path = os.path.join(root, 'test')

    data_dataset = ImageFolder(data_path)
    
    return data_dataset