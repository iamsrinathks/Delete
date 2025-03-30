import os
import json
import subprocess

def patch_model():
    """
    Dynamically updates a Vertex AI model using user-defined fields in Harness.
    Fields to update and their values are passed as environment variables.
    """
    project = os.getenv("GCP_PROJECT")
    region = os.getenv("GCP_REGION")
    model_id = os.getenv("MODEL_ID")
    access_token = os.getenv("ACCESS_TOKEN")

    if not all([project, region, model_id, access_token]):
        raise ValueError("❌ Missing required environment variables: GCP_PROJECT, GCP_REGION, MODEL_ID, or ACCESS_TOKEN.")

    # User specifies the fields to update as a comma-separated list (e.g., "displayName,labels,description")
    update_fields = os.getenv("MODEL_UPDATE_FIELDS", "").split(",")

    if not update_fields or update_fields == [""]:
        raise ValueError("❌ No fields specified for update. Set MODEL_UPDATE_FIELDS.")

    # Prepare the update payload dynamically
    update_payload = {}

    for field in update_fields:
        env_var = f"MODEL_UPDATE_{field.upper()}"  # Convert to uppercase (e.g., MODEL_UPDATE_DISPLAYNAME)
        value = os.getenv(env_var)
        if value:
            try:
                update_payload[field] = json.loads(value)  # Handle JSON fields (like labels)
            except json.JSONDecodeError:
                update_payload[field] = value  # Handle string values

    if not update_payload:
        raise ValueError("❌ No valid updates found. Check environment variable values.")

    # Construct the update mask (comma-separated list of fields)
    update_mask = ",".join(update_payload.keys())

    # API endpoint
    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}"

    print(f"🚀 Updating model {model_id} with fields: {update_mask}...")

    try:
        result = subprocess.run(
            [
                "curl", "-X", "PATCH",
                "-H", f"Authorization: Bearer {access_token}",
                "-H", "Content-Type: application/json",
                f"{api_endpoint}?updateMask={update_mask}",
                "-d", json.dumps(update_payload)
            ],
            capture_output=True, text=True, check=True
        )

        print(f"✅ Model updated successfully! Response: {result.stdout}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to update model. Error: {e.stderr}")
        raise


export MODEL_UPDATE_FIELDS="displayName,labels"
export MODEL_UPDATE_DISPLAYNAME="New Model Name"
export MODEL_UPDATE_LABELS='{"env": "staging", "owner": "data-science"}'
