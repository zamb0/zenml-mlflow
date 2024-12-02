from zenml import step
import numpy as np


@step()
def visualize_prediction(predictions: np.ndarray, true_label: np.ndarray) -> list[list[int]]:
    
    output = []
        
    for prediction, true in zip(predictions, true_label):
        output.append([int(np.argmax(prediction)), int(true)])
        
    print(output)
    return output