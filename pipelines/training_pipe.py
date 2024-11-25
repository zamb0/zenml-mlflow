from zenml import pipeline
from steps.ingest import ingest
from steps.transform import transform
from steps.train import train
from steps.evaluate import evaluate

@pipeline
def training_pipe(data_root:str="hymenoptera_data"):
    train_dataset, test_dataset = ingest(data_root)
    train_dataset, test_dataset = transform(train_dataset, test_dataset)
    train(train_dataset)
    evaluate(test_dataset)