class Config:
    """
    Configuration class
    
    Attributes:
        lr: float: Learning rate
        momentum: float: Momentum
        step_size: int: Step size
        gamma: float: Gamma
        batch_size: int: Batch size
        num_epochs: int: Number of epochs
        min_accuracy: float: Minimum accuracy required to deploy the model
        model_name: str: Model name
        stack_name: str: Stack name
        experiment_name: str: Experiment name
    """
    
    lr=0.001
    momentum=0.9
    step_size=7
    gamma=0.1
    batch_size=4
    num_epochs=25
    min_accuracy=0.8
    model_name='resnet18'
    stack_name='mlflow_stack'
    experiment_name='hymenoptera_experiment'