from zenml import step
from zen_client import experiment_tracker

@step(enable_cache=False)
def deployment_trigger(accuracy: float, min_accuracy: float=0.92):
    return accuracy >= min_accuracy