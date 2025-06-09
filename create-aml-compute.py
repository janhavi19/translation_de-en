from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

# Connect to workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="d9222192-f611-4b4d-bdd6-9f331c6a00b9",
    resource_group_name="rg-janhavipuranik1995-4580_ai",
    workspace_name="ml-workspace"
)

# Create compute config
compute_name = "translation-compute"
compute = AmlCompute(
    name=compute_name,
    size="Standard_DS3_v2",
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=120  # Scale down after 120 minutes of inactivity
)

print(f"Creating new compute cluster '{compute_name}'...")
operation = ml_client.begin_create_or_update(compute)
created_compute = operation.result()
print(f"Successfully created compute cluster: {created_compute.name}")

print("\nRun the pipeline with:")
print(f"python3 pipeline.py --subscription-id d9222192-f611-4b4d-bdd6-9f331c6a00b9 --resource-group rg-janhavipuranik1995-4580_ai --workspace-name ml-workspace --language-pair de-en --compute-target {compute_name}")