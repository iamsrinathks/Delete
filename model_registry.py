import os
import json
import joblib
import requests
from google.cloud import aiplatform
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Step 1: Training the Model
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(clf, output_file)
    print(f"✅ Model trained with random_state {random_state} and saved to {output_file}")

# -----------------------------
# Step 2: Uploading to GCS using PUT Method
# -----------------------------
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

    # Read the model file
    with open(source_file, "rb") as file_data:
        file_contents = file_data.read()

    # Get OAuth token (Harness OIDC handles authentication)
    token_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
    headers = {"Metadata-Flavor": "Google"}
    response = requests.get(token_url, headers=headers)
    response.raise_for_status()
    access_token = response.json()["access_token"]

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/octet-stream"
    }

    # Use PUT request to upload the file
    response = requests.put(url, headers=headers, data=file_contents)

    if response.status_code in (200, 201):
        print(f"✅ File uploaded successfully to gs://{bucket_name}/{destination_blob}")
    else:
        print(f"❌ Failed to upload file: {response.status_code} {response.text}")

# -----------------------------
# Step 3: Registering the Model
# -----------------------------
def register_model():
    """
    Registers a model artifact with Vertex AI and attaches labels/aliases.
    Reads parameters from environment variables.
    """
    project = os.getenv("GCP_PROJECT")
    region = os.getenv("GCP_REGION")
    display_name = os.getenv("MODEL_DISPLAY_NAME")
    artifact_uri = os.getenv("MODEL_ARTIFACT_URI")
    labels_json = os.getenv("MODEL_LABELS", "{}")  # Read labels as JSON string

    if not project or not region or not artifact_uri:
        raise ValueError("❌ GCP_PROJECT, GCP_REGION, and MODEL_ARTIFACT_URI must be set")

    labels = json.loads(labels_json)

    # Initialize Vertex AI
    aiplatform.init(project=project, location=region)

    # Register the model
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
        labels=labels
    )
    print(f"✅ Model '{display_name}' registered with resource name: {model.resource_name}")

# -----------------------------
# Main: Run based on HARNESS_STAGE variable
# -----------------------------
if __name__ == "__main__":
    harness_stage = os.getenv("HARNESS_STAGE")

    if harness_stage == "train":
        train_model()
    elif harness_stage == "upload":
        upload_model()
    elif harness_stage == "register":
        register_model()
    else:
        raise ValueError("❌ HARNESS_STAGE must be one of 'train', 'upload', or 'register'")
