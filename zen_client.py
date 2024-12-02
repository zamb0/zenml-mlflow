from zenml.client import Client
import mlflow
from config import Config

mlflow.set_tracking_uri('http://127.0.0.1:5000')

experiment_tracker = Client().active_stack.experiment_tracker
print(Client().active_stack.experiment_tracker.get_tracking_uri())