from pipelines.training_pipe import training_pipe
from zen_client import Client
import mlflow

if __name__ == "__main__":
    training_pipe(data_root="hymenoptera_data")
    