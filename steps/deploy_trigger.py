from zenml import step

@step(enable_cache=False)
def deployment_trigger(accuracy: float, min_accuracy: float=0.92):
    """
    Step to trigger deployment based on accuracy
    
    Args:
        accuracy: Accuracy of the model
        min_accuracy: Minimum accuracy required to deploy the model
        
    Returns:
        bool: True if accuracy is greater than min_accuracy, False otherwise
    """
    return accuracy >= min_accuracy