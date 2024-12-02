import logging
import torch
import mlflow
import numpy as np
from zenml import step
from src.model import CNN
from typing import Tuple, Annotated
from config import Config
from zen_client import experiment_tracker
from mlflow.models.signature import infer_signature
from torchvision.datasets import ImageFolder

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def train(train_dataset: ImageFolder,
          val_dataset: ImageFolder) \
        -> Tuple[Annotated[torch.nn.Module, Config.model_name], 
                 Annotated[float, 'accuracy']]:
    """
    Step to train the model
    
    Args:
        train_dataset: ImageFolder: Training dataset
        val_dataset: ImageFolder: Validation dataset
        
    Returns:
        Tuple[torch.nn.Module, float]: Trained model and accuracy
    """
          
    logging.info('Training model')
    mlflow.pytorch.autolog() 
    mlflow.log_param('batch_size', Config.batch_size)
    mlflow.log_param('lr', Config.lr)
    mlflow.log_param('momentum', Config.momentum)
    mlflow.log_param('gamma', Config.gamma)
    mlflow.log_param('step_size', Config.step_size)
    mlflow.log_param('num_epochs', Config.num_epochs)
    
    CNN_model = CNN(num_classes=2)
    
    optimizer = torch.optim.SGD(CNN_model.model.parameters(), lr=Config.lr, momentum=Config.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.gamma, gamma=Config.gamma)

    accuracy = CNN_model.train_model(train_dataset, 
                    val_dataset, 
                    criterion=torch.nn.CrossEntropyLoss(), 
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_epochs=Config.num_epochs, 
                    dataset_sizes=[len(train_dataset), len(val_dataset)], device="cpu")

    mlflow.log_metric("accuracy", accuracy)
    signature = infer_signature(model_input=np.array(train_dataset[0][0].reshape(1, 3, 224, 224)), model_output=np.array([1]))
    mlflow.pytorch.log_model(CNN_model.model, Config.model_name, signature=signature, registered_model_name=Config.model_name)
    #signature = infer_signature(train_dataset, prediction)    
 
    return CNN_model.model, float(accuracy)
    