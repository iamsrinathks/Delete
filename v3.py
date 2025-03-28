import os
import json
import joblib
import subprocess
import requests
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Helper Function: Run `curl`
# ---------------------------
def run_curl(method, url, payload=None, data_file=None):
    """
    Helper function to run a `curl` request with authentication.
    Supports JSON payloads or binary file uploads.
    """
    access_token = os.getenv("ACCESS_TOKEN")
    headers = [
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json"
    ]
    
    curl_cmd = ["curl", "-X", method] + headers + [url]

    if payload:
        curl_cmd += ["-d", json.dumps(payload)]
    
    if data_file:
        curl_cmd += ["--data-binary", f"@{data_file}"]

    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout) if result.stdout else None
    except subprocess.CalledProcessError as e:
        print(f"❌ Curl request failed: {e.stderr}")
        return None

# ---------------------------
# Step 1: Train Model
# ---------------------------
def train_model():
    """
    Loads the Iris dataset, trains a RandomForestClassifier, and saves the model.
    Reads parameters from environment variables.
    """
    random_state = int(os.getenv("MODEL_RANDOM_STATE", 42))
    output_file = os.getenv("MODEL_OUTPUT_FILE", "model.joblib")

    # Load dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split dataset (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Train the model
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(clf, output_file)
    print(f"✅ Model trained with random_state {random_state} and saved to {output_file}")

# ---------------------------
# Step 2: Upload Model to GCS
# ---------------------------
def upload_model():
    """
    Uploads a file to a GCS bucket using the XML API with the PUT method.
    Reads parameters from environment variables.
    """
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    source_file = os.getenv("MODEL_OUTPUT_FILE", "model.joblib")
    destination_blob = os.getenv("GCS_DESTINATION_BLOB", "models/model.joblib")

    if not bucket_name:
        raise ValueError("❌ GCS_BUCKET_NAME is not set")

    # Construct GCS URL for PUT upload
    url = f"https://storage.googleapis.com/{bucket_name}/{destination_blob}"

    print("🚀 Uploading model to GCS...")
    response = run_curl("PUT", url, data_file=source_file)

    if response:
        print(f"✅ File uploaded successfully to gs://{bucket_name}/{destination_blob}")
    else:
        print("❌ Failed to upload file.")

# ---------------------------
# Step 3: Register Model in Vertex AI
# ---------------------------
def register_model():
    """
    Registers a new model in Vertex AI Model Registry using REST API.
    If the model exists, registers a new version (v2, v3, etc.).
    """
    project = os.getenv("GCP_PROJECT")
    region = os.getenv("GCP_REGION")
    display_name = os.getenv("MODEL_DISPLAY_NAME")
    artifact_uri = os.getenv("MODEL_ARTIFACT_URI")
    labels_json = os.getenv("MODEL_LABELS", "{}")
    aliases = os.getenv("MODEL_ALIASES", "latest,stable").split(",")

    if not project or not region or not artifact_uri or not display_name:
        raise ValueError("❌ Missing required environment variables.")

    labels = json.loads(labels_json)

    # Check if the model already exists
    list_models_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models"
    models = run_curl("GET", list_models_url)

    existing_model_id = None
    for model in models.get("models", []):
        if model["displayName"] == display_name:
            existing_model_id = model["name"].split("/")[-1]
            break

    # Prepare model registration payload
    model_payload = {
        "displayName": display_name,
        "artifactUri": artifact_uri,
        "containerSpec": {
            "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
        },
        "labels": labels
    }

    if existing_model_id:
        print(f"🔄 Model '{display_name}' exists. Registering a new version...")
        model_payload["parentModel"] = f"projects/{project}/locations/{region}/models/{existing_model_id}"

    # Register the model
    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models:upload"
    response = run_curl("POST", api_endpoint, {"model": model_payload})

    if response:
        model_id = response["model"].split("/")[-1]
        print(f"✅ Model registered successfully: {model_id}")

        # Assign version aliases (latest, stable)
        merge_aliases(model_id, project, region, aliases)
    else:
        print("❌ Model registration failed.")

# ---------------------------
# Step 4: Merge Version Aliases
# ---------------------------
def merge_aliases(model_id, project, region, aliases):
    """
    Assigns version aliases (e.g., 'latest', 'stable') to the new model version.
    """
    print(f"🔄 Merging version aliases {aliases} for model {model_id}...")

    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}:mergeVersionAliases"
    response = run_curl("POST", api_endpoint, {"versionAliases": aliases})

    if response:
        print(f"✅ Version aliases merged successfully.")
    else:
        print("❌ Failed to merge version aliases.")

# ---------------------------
# Additional Model Management Functions
# ---------------------------
def patch_model():
    """Updates model metadata (labels, description)."""
    project, region, model_id = os.getenv("GCP_PROJECT"), os.getenv("GCP_REGION"), os.getenv("MODEL_ID")
    updated_labels = json.loads(os.getenv("UPDATED_LABELS", "{}"))

    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}"
    response = run_curl("PATCH", api_endpoint, {"labels": updated_labels, "description": "Updated model description"})

    print("✅ Model updated successfully." if response else "❌ Failed to update model.")

def export_model():
    """Exports a model from Vertex AI to a GCS bucket."""
    project, region, model_id, gcs_destination = os.getenv("GCP_PROJECT"), os.getenv("GCP_REGION"), os.getenv("MODEL_ID"), os.getenv("GCS_EXPORT_PATH")

    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}:export"
    response = run_curl("POST", api_endpoint, {"outputConfig": {"artifactDestination": {"outputUriPrefix": gcs_destination}}})

    print("✅ Model export initiated successfully." if response else "❌ Failed to export model.")


# --------------------------------
# Delete a Specific Model Version
# --------------------------------
def delete_model_version():
    """
    Deletes a specific model version from Vertex AI.
    """
    project, region, model_id = os.getenv("GCP_PROJECT"), os.getenv("GCP_REGION"), os.getenv("MODEL_ID")

    if not project or not region or not model_id:
        raise ValueError("❌ Missing required environment variables.")

    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}"

    print("🗑️ Deleting model version...")
    response = run_curl("DELETE", api_endpoint)
    
    if response:
        print(f"✅ Model version deleted successfully: {response}")
    else:
        print("❌ Failed to delete model version.")

def delete_model():
    """Deletes an entire model (all versions)."""
    project, region, model_id = os.getenv("GCP_PROJECT"), os.getenv("GCP_REGION"), os.getenv("MODEL_ID")
    if not project or not region or not model_id:
        raise ValueError("❌ Missing required environment variables.")
    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}:delete"

    response = run_curl("DELETE", api_endpoint)
    print("✅ Model deleted successfully." if response else "❌ Failed to delete model.")


def set_model_iam_policy():
    """Sets IAM policy for the registered model in Vertex AI."""
    project, region, model_id = os.getenv("GCP_PROJECT"), os.getenv("GCP_REGION"), os.getenv("MODEL_ID")
    iam_role = os.getenv("IAM_ROLE")
    iam_members = os.getenv("IAM_MEMBERS")

    # Ensure required variables are provided
    if not all([project, region, model_id, iam_role, iam_members]):
        raise ValueError("❌ Missing required environment variables. Ensure MODEL_ID, IAM_ROLE, and IAM_MEMBERS are set.")

    # Convert IAM members to a list
    members_list = [member.strip() for member in iam_members.split(",")]

    # API endpoint for setting IAM policy
    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}:setIamPolicy"

    # IAM policy payload
    policy_payload = {
        "policy": {
            "bindings": [
                {
                    "role": iam_role,
                    "members": members_list
                }
            ]
        }
    }

    print("🚀 Setting IAM policy for the model...")

    # Call the run_curl() helper function
    response = run_curl("POST", api_endpoint, policy_payload)

    if response:
        print("✅ IAM Policy set successfully!")
    else:
        print("❌ Failed to set IAM Policy.")


# ---------------------------
# Main: Run based on HARNESS_STAGE variable
# ---------------------------
if __name__ == "__main__":
    actions = {
        "train": train_model,
        "upload": upload_model,
        "register": register_model,
        "patch": patch_model,
        "export": export_model,
        "delete": delete_model,
        "set_iam_policy": set_model_iam_policy,  # ✅ Added IAM Policy Function
    }

    harness_stage = os.getenv("HARNESS_STAGE")

    if harness_stage in actions:
        actions[harness_stage]()
    else:
        raise ValueError("❌ Invalid HARNESS_STAGE.")

