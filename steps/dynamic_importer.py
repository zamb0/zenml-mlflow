from zenml import step
from typing import Tuple, Annotated
import torchvision
import os
import logging
from torchvision.datasets import ImageFolder

@step(enable_cache=False)
def dynamic_importer(root:str="hymenoptera_data") \
        -> Annotated[torchvision.datasets.ImageFolder, 'data']:
        
    logging.info('Ingesting data')
    
    data_path = os.path.join(root, 'test')

    data_dataset = ImageFolder(data_path)
    
    return data_dataset