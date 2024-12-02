import os
import time
import torch
from typing import Annotated
from tempfile import TemporaryDirectory
from torch import nn
from torchvision import models
from abc import ABC, abstractmethod
from config import Config
from torchvision.datasets import ImageFolder

class Model(ABC):
    """
    Abstract class for the model
    """
    
    @abstractmethod
    def train_model(self, train_dataset: ImageFolder, val_dataset: ImageFolder, **kwargs):
        """
        Train the model
        """
        pass

class CNN(Model):
    def __init__(self, num_classes):
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.num_classes = num_classes
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
    
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    def train_model(self, train_dataset: ImageFolder, val_dataset: ImageFolder, criterion, optimizer, scheduler, num_epochs=25, dataset_sizes=[0], device="cpu") \
                    -> Annotated[float, 'accuracy']:
        """
        Train the model
        
        Args:
        
            train_dataset: ImageFolder: Training dataset
            val_dataset: ImageFolder: Validation dataset
            criterion: nn.Module: Loss function
            optimizer: torch.optim: Optimizer
            scheduler: torch.optim: Scheduler
            num_epochs: int: Number of epochs
            dataset_sizes: List[int]: Dataset sizes
            device: str: Device to train the model
            
        Returns:
            float: Accuracy
        """
        
        since = time.time()
        model = self.model.to(device)
        
        data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
        data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=True)
        
        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train() 
                        data_loader = data_loader_train
                        datasize = dataset_sizes[0]
                    else:
                        model.eval() 
                        data_loader = data_loader_val
                        datasize = dataset_sizes[1]

                    running_loss = 0.0
                    running_corrects = 0

                    for inputs, labels in data_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / datasize
                    epoch_acc = running_corrects.double() / datasize

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)

                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            # load best model weights
            model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        return best_acc