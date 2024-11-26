import logging
import torch
import torchvision
from zenml import step

@step
def inference(val_dataset: torchvision.datasets.ImageFolder):
    logging.info('Evaluating model') 
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print('Number of samples in val_loader: ', len(val_loader))
    