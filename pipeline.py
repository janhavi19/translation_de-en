import os
import argparse
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential
import time
import uuid

def create_translation_pipeline(subscription_id, resource_group, workspace_name, language_pair, compute_target):
    """Create an Azure ML pipeline for the translation model."""
    # Connect to Azure ML workspace
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name
    )
    
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    
    # Define environment
    env = Environment(
        name=f"translation-env-{language_pair}-{timestamp}",
        description=f"Environment for {language_pair} translation model",
        build=BuildContext(path="./azure/environments")
    )
    ml_client.environments.create_or_update(env)
    
    # Define the training job
    train_job = command(
        name=f"translation-training-{language_pair}-{timestamp}-{unique_id}",
        display_name=f"{language_pair} Translation Model Training",
        description=f"Fine-tune a {language_pair} translation model",
        inputs={
            "data_dir": Input(
                type="uri_folder",
                path=f"azureml://datastores/workspaceblobstore/paths/data/translation/{language_pair}"
            ),
            "language_pair": language_pair
        },
        outputs={
            "model_dir": Output(
                type="uri_folder",
                path=f"azureml://datastores/workspaceblobstore/paths/models/translation/{language_pair}"
            )
        },
        code="./src",
        command="python training.py --language-pair ${{inputs.language_pair}} --data-dir ${{inputs.data_dir}} --output-dir ${{outputs.model_dir}}",
        environment=env.name + "@latest",
        compute=compute_target  # Add compute target here
    )
    
    # Create pipeline with the job
    timestamp_pipeline = int(time.time())
    unique_id_pipeline = str(uuid.uuid4())[:8]
    
    @pipeline(
        name=f"translation-pipeline-{language_pair}-{timestamp_pipeline}-{unique_id_pipeline}",
        description=f"Pipeline for {language_pair} translation model",
        display_name=f"translation-pipeline-{language_pair}-{timestamp_pipeline}-{unique_id_pipeline}"
    )
    def translation_pipeline():
        # Run the training job
        training_result = train_job()
        # Return the outputs properly
        return {
            "model_output": training_result.outputs.model_dir
        }
    
    return translation_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Azure ML pipeline for translation")
    parser.add_argument("--subscription-id", type=str, required=True, help="Azure subscription ID")
    parser.add_argument("--resource-group", type=str, required=True, help="Azure resource group")
    parser.add_argument("--workspace-name", type=str, required=True, help="Azure ML workspace name")
    parser.add_argument("--language-pair", type=str, default="en-de", help="Language pair (e.g., en-de)")
    parser.add_argument("--compute-target", type=str, required=True, help="Compute target name for training")
    args = parser.parse_args()
    
    # Create the pipeline
    pipeline_func = create_translation_pipeline(
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
        args.language_pair,
        args.compute_target
    )
    
    # Connect to Azure ML workspace
    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name
    )
    
    # Create the pipeline job
    pipeline_job = pipeline_func()
    
    # Submit the pipeline job
    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Submitted job: {returned_job.name}")
    
    # Stream the logs
    ml_client.jobs.stream(returned_job.name)