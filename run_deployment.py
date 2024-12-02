import click
from typing import cast
from rich import print
from pipelines.deployment_pipe import deployment_pipeline, inference_pipeline
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from mlflow.tracking import get_tracking_uri
from config import Config

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model (`deploy`), or to "
    "only run a prediction against the deployed model "
    "(`predict`). By default both will be run "
    "(`deploy_and_predict`).",
)
@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required to deploy the model",
)

def run_deployment(config, min_accuracy: float):
    """
    Run deployment pipeline
    
    Args:
        config: Configuration for the deployment
        min_accuracy: Minimum accuracy required to deploy the model
        
    Returns:
        None
    """
    
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    
    print(mlflow_model_deployer_component)
    
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    
    if deploy:
        print("Running deployment pipeline")
        deployment_pipeline(min_accuracy=min_accuracy, workers=4, timeout=60)
    elif predict:
        print("Running prediction pipeline")
        inference_pipeline(
            pipeline_name="deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step"
        )

    print(
        f"You can run:\n"
        f"[italic green]mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]\n"
        f"...to inspect your experiment runs within the MLflow UI.\n"
        f"You can find your runs tracked within the "
        f"[bold]mlflow_example_pipeline[/bold] experiment. There you'll also be able to "
        f"compare two or more runs.\n"
    )


    # fetch existing services with same pipeline name, step name and model name
    # stuff from the tutorials on the ZenML website
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name=Config.model_name,
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        service.start(timeout=60)
        
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally as a daemon "
                f"process service and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml model-deployer models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )
        
if __name__ == "__main__":
    run_deployment()

        
