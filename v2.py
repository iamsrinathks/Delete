import os
import json
import subprocess
import requests

def register_model():
    """
    Registers a new model in Vertex AI Model Registry using REST API (curl).
    If the model exists, it registers a new version (v2, v3, etc.).
    """

    # 🔹 Fetch required environment variables
    project = os.getenv("GCP_PROJECT")
    region = os.getenv("GCP_REGION")
    display_name = os.getenv("MODEL_DISPLAY_NAME")
    artifact_uri = os.getenv("MODEL_ARTIFACT_URI")
    labels_json = os.getenv("MODEL_LABELS", "{}")
    aliases = os.getenv("MODEL_ALIASES", "latest,stable").split(",")
    access_token = os.getenv("ACCESS_TOKEN")  # OAuth token provided by Harness OIDC

    if not project or not region or not artifact_uri or not display_name:
        raise ValueError("❌ GCP_PROJECT, GCP_REGION, MODEL_DISPLAY_NAME, and MODEL_ARTIFACT_URI must be set.")

    labels = json.loads(labels_json)

    # 🔹 API endpoint for listing models
    list_models_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # 🔹 Step 1: Check if the model already exists
    response = requests.get(list_models_url, headers=headers)
    response.raise_for_status()
    models = response.json().get("models", [])

    existing_model_id = None

    for model in models:
        if model["displayName"] == display_name:
            existing_model_id = model["name"].split("/")[-1]  # Extract model ID
            break

    # 🔹 Step 2: Construct the model registration payload
    model_payload = {
        "displayName": display_name,
        "artifactUri": artifact_uri,
        "containerSpec": {
            "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
        },
        "labels": labels
    }

    if existing_model_id:
        print(f"🔄 Model '{display_name}' exists. Registering a new version (v2, v3, etc.)...")
        parent_model = f"projects/{project}/locations/{region}/models/{existing_model_id}"
        model_payload["parentModel"] = parent_model  # ✅ Register as a new version

    # 🔹 Step 3: Register the model using `curl`
    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models:upload"
    curl_cmd = [
        "curl", "-X", "POST",
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({"model": model_payload}),
        api_endpoint
    ]

    try:
        print("🚀 Registering model in Vertex AI...")
        result = subprocess.run(curl_cmd, capture_output=True, text=True, check=True)
        response_json = json.loads(result.stdout)

        model_id = response_json["model"].split("/")[-1]  # Extract registered model ID
        print(f"✅ Model registered successfully: {model_id}")

        # 🔹 Step 4: Assign aliases (latest, stable) to the newly registered version
        merge_aliases(model_id, project, region, access_token, aliases)

    except subprocess.CalledProcessError as e:
        print(f"❌ Model registration failed: {e.stderr}")

# -----------------------------
# Step 5: Merge Version Aliases
# -----------------------------
def merge_aliases(model_id, project, region, access_token, aliases):
    """
    Assigns version aliases (e.g., 'latest', 'stable') to the new model version.
    """
    print(f"🔄 Merging version aliases {aliases} for model {model_id}...")

    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}:mergeVersionAliases"

    payload = json.dumps({"versionAliases": aliases})

    curl_cmd = [
        "curl", "-X", "POST",
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json",
        "-d", payload,
        api_endpoint
    ]

    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, check=True)
        print(f"✅ Version aliases merged successfully: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to merge version aliases: {e.stderr}")
