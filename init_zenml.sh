#!/bin/bash

echo "Initializing ZenML"
echo "Insert the mlflow experiment name: "
read experiment_name

# Initialize ZenML
zenml init

# Install mlflow integration
zenml integration install mlflow -y

# Create experiment tracker
zenml experiment-tracker register mlflow_tracker --flavor=mlflow

# Create model deploymer
zenml model-deployer register mlflow --flavor=mlflow

# Create stack
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set

mlflow experiments create --experiment-name=$experiment_name