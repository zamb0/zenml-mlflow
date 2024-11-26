import mlflow
from zenml import pipeline
from steps.ingest import ingest
from steps.transform import transform
from steps.train import train

mlflow.set_tracking_uri('http://127.0.0.1:5000')

@pipeline()
def training_pipe(data_root:str="hymenoptera_data"):
      
    train_dataset, val_dataset = ingest(data_root)
    train_dataset, val_dataset = transform(train_dataset, val_dataset)
    model = train(train_dataset, val_dataset)
    
    #mlflow.pytorch.log_model(model, "model")