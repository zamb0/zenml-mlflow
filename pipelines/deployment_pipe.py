
from zenml.config import DockerSettings
from zenml.pipelines import pipeline
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from steps.ingest import ingest
from steps.transform import transform
from steps.train import train
from steps.deploy_trigger import deployment_trigger
from steps.inference_service_loader import inference_service_loader
from steps.dynamic_importer import dynamic_importer
from steps.predictor import predictor
from steps.visualize_prediction import visualize_prediction

from config import Config

docker_settings = DockerSettings(required_integrations=["MLFLOW"])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def deployment_pipeline(min_accuracy: float=0.92, workers: int=1, timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    """
    Deployment pipeline
    
    Args:
        min_accuracy: Minimum accuracy required to deploy the model
        workers: Number of workers to use
        timeout: Timeout for the deployment
        
    Returns:
        None
    """
    train_dataset, val_dataset = ingest()
    train_dataset, val_dataset = transform(train_dataset, val_dataset)
    model, accuracy = train(train_dataset, val_dataset)
    decision = deployment_trigger(accuracy, min_accuracy)
    mlflow_model_deployer_step(model=model, 
                               deploy_decision=decision, 
                               workers=workers, 
                               timeout=timeout, 
                               model_name=Config.model_name, 
                               experiment_name=Config.experiment_name)
    

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    """
    Inference pipeline
    
    Args:
        pipeline_name: Name of the pipeline
        pipeline_step_name: Name of the pipeline step
        
    Returns:
        None
    """
    data = dynamic_importer()
    service = inference_service_loader(pipeline_name=pipeline_name, pipeline_step_name=pipeline_step_name, running=False, model_name=Config.model_name)
    prediction, true_label = predictor(service, data)
    visualize_prediction(prediction, true_label)
    return prediction