import mlflow
from zenml import pipeline
from steps.ingest import ingest
from steps.transform import transform
from steps.train import train

@pipeline()
def training_pipe(data_root:str="hymenoptera_data"):
    """
    Training pipeline
    
    Args:
        data_root: Root directory of the data
        
    Returns:
        None
    """
    train_dataset, val_dataset = ingest(data_root)
    train_dataset, val_dataset = transform(train_dataset, val_dataset)
    model, accuracy = train(train_dataset, val_dataset)