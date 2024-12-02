from zenml.client import Client
import mlflow
from config import Config

mlflow.set_tracking_uri('http://127.0.0.1:5000')

experiment_tracker = Client().active_stack.experiment_tracker.name
print(experiment_tracker)
mlflow.set_experiment(experiment_name=experiment_tracker)