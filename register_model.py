import os
import json
import subprocess

def register_model():
    """
    Registers a model in Vertex AI using a direct REST API call via `curl`.
    """

    # 🔹 Get environment variables (from Harness Input Sets or system)
    project = os.getenv("GCP_PROJECT")
    region = os.getenv("GCP_REGION")
    display_name = os.getenv("MODEL_DISPLAY_NAME")
    artifact_uri = os.getenv("MODEL_ARTIFACT_URI")
    model_id = os.getenv("MODEL_ID", None)  # Optional
    parent_model = os.getenv("PARENT_MODEL", None)  # Optional for versioning
    version_aliases = os.getenv("MODEL_VERSION_ALIASES", "v1").split(",")  # Optional
    description = os.getenv("MODEL_DESCRIPTION", "My Model")
    encryption_spec_key_name = os.getenv("KMS_KEY", None)  # Optional KMS Key

    if not project or not region or not artifact_uri:
        raise ValueError("❌ GCP_PROJECT, GCP_REGION, and MODEL_ARTIFACT_URI must be set")

    # 🔹 Get an Access Token (Assuming Harness is handling OIDC)
    access_token = os.getenv("ACCESS_TOKEN")
    if not access_token:
        raise ValueError("❌ ACCESS_TOKEN is required but not found.")

    # 🔹 Construct API Endpoint
    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models:upload"

    # 🔹 Construct the Model JSON Payload
    model_payload = {
        "model": {
            "displayName": display_name,
            "artifactUri": artifact_uri,
            "description": description,
            "versionAliases": version_aliases,
            "containerSpec": {
                "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
            }
        }
    }

    if model_id:
        model_payload["modelId"] = model_id
    if parent_model:
        model_payload["parentModel"] = parent_model
    if encryption_spec_key_name:
        model_payload["model"]["encryptionSpec"] = {"kmsKeyName": encryption_spec_key_name}

    # 🔹 Convert payload to JSON string
    model_json = json.dumps(model_payload)

    # 🔹 Construct `curl` Command
    curl_cmd = [
        "curl", "-X", "POST",
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json",
        "-d", model_json,
        api_endpoint
    ]

    # 🔹 Run the `curl` command
    try:
        print("🚀 Running `curl` command to register model in Vertex AI...")
        result = subprocess.run(curl_cmd, capture_output=True, text=True, check=True)
        print(f"✅ Model registered successfully: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Model registration failed: {e.stderr}")

