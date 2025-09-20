# promote model

import os
import mlflow

def promote_model_to_staging():
    mlflow.set_tracking_uri('http://44.203.159.181:5000/')
    client = mlflow.MlflowClient()
    model_name = "my_model"
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(int(v.version) for v in versions)
    # Set alias "staging" to latest version
    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=latest_version
    )
    print(f"Promoted {model_name} v{latest_version} to alias 'staging'")


def promote_model_to_production():
    mlflow.set_tracking_uri('http://44.203.159.181:5000/')
    client = mlflow.MlflowClient()
    model_name = "my_model"
    # Get version with alias "staging"
    staging_version_info = client.get_model_version_by_alias(model_name, "staging")
    staging_version = staging_version_info.version
    # Set alias "production" to this version
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=staging_version
    )
    print(f"Model version {staging_version} promoted to alias 'production'")

if __name__ == "__main__":
    promote_model_to_staging()
    promote_model_to_production()