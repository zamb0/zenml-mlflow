from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from torchvision.datasets import ImageFolder
from typing import Tuple
import numpy as np
import torch
import torchvision
import mlflow
from zen_client import experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def predictor(service: MLFlowDeploymentService, data: ImageFolder) -> Tuple[np.ndarray, np.ndarray]:
    service.start(timeout=60)
    
    data.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    data_loader = torch.utils.data.DataLoader(data, batch_size = len(data), shuffle = False)
    
    for image, label in data_loader:
        images = image
        labels = label
        break
    
    for i, image in enumerate(images):
        mlflow.log_image(image.permute(1, 2, 0).numpy(), key=f'image_{i}')
        
    for i, label in enumerate(labels):
        mlflow.log_metric(value=label, key=f'true_label_{i}')
    
    prediction = service.predict(images.numpy())
    
    return prediction, labels.numpy()