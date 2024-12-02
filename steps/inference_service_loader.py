import logging
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

@step(enable_cache=False)
def inference_service_loader(pipeline_name: str, pipeline_step_name: str, running: bool, model_name: str = 'model') -> MLFlowDeploymentService:
    """
    Step to load the inference service
    
    Args:
        pipeline_name: Name of the pipeline
        pipeline_step_name: Name of the pipeline step
        running: Whether the service is running
        model_name: Name of the model
        
    Returns:
        MLFlowDeploymentService: Inference service
    """
    
    logging.info('Evaluating model') 
    
    mlflow_model_deployment_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployment_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )
    
    if not existing_services:
        raise RuntimeError(
            f"Could not find any running services for pipeline {pipeline_name}, step {pipeline_step_name}, model {model_name}."
        )
        
    return existing_services[0]
    
    
    