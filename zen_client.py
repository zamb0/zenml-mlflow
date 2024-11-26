from zenml.client import Client

experiment_tracker = Client().activate_stack('mlflow_stack')