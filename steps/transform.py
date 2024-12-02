import logging
from typing import Tuple, Annotated
from torchvision import transforms
from zenml import step
from torchvision.datasets import ImageFolder

@step()
def transform(train_dataset: ImageFolder = None, val_dataset:ImageFolder = None) \
            -> Tuple[Annotated[ImageFolder, 'train_dataset'], Annotated[ImageFolder, 'val_dataset']]:
    """
    Step to transform the data
    
    Args:
        train_dataset: ImageFolder: Training dataset
        val_dataset: ImageFolder: Validation dataset
        
    Returns:
        Tuple[ImageFolder, ImageFolder]: Transformed datasets
    """
        
    logging.info('Transforming data')
    
    if train_dataset is not None:
       
        train_transformation = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset.transform = train_transformation
                                  
    if val_dataset is not None:
        val_transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
        val_dataset.transform = val_transformation
        
    

    return train_dataset, val_dataset

    