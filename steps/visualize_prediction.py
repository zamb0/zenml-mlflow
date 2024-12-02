import numpy as np
import mlflow
from zenml import step
from zen_client import experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def visualize_prediction(predictions: np.ndarray, true_label: np.ndarray) -> list[list[int]]:
    """
    Step to visualize the predictions of the model
    
    Args:
        predictions: np.ndarray: Predictions of the model
        true_label: np.ndarray: True labels of the data
        
    Returns:
        list[list[int]]: Predicted and true labels
    """
    
    output = []
        
    i=0
    for prediction, true in zip(predictions, true_label):
        output.append([int(np.argmax(prediction)), int(true)])
        mlflow.log_metric(key=f'predicted_label_{i}', value=int(np.argmax(prediction)))
        i+=1
    
    print(output)
    return output