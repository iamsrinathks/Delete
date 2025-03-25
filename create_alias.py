import os
import json
import subprocess

def merge_version_aliases():
    """
    Merges new version aliases for an existing model in Vertex AI.
    """
    # 🔹 Get environment variables
    project = os.getenv("GCP_PROJECT")
    region = os.getenv("GCP_REGION")
    model_id = os.getenv("MODEL_ID")
    version_id = os.getenv("MODEL_VERSION_ID")  # The specific version to alias
    new_aliases = os.getenv("NEW_VERSION_ALIASES", "").split(",")  # Comma-separated list of new aliases

    if not project or not region or not model_id or not version_id or not new_aliases:
        raise ValueError("❌ GCP_PROJECT, GCP_REGION, MODEL_ID, MODEL_VERSION_ID, and NEW_VERSION_ALIASES must be set.")

    # 🔹 Get Access Token (Assuming Harness is handling OIDC)
    access_token = os.getenv("ACCESS_TOKEN")
    if not access_token:
        raise ValueError("❌ ACCESS_TOKEN is required but not found.")

    # 🔹 Construct API Endpoint
    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}:mergeVersionAliases"

    # 🔹 Construct JSON Payload
    payload = {
        "versionAliases": new_aliases
    }

    # 🔹 Convert payload to JSON string
    payload_json = json.dumps(payload)

    # 🔹 Construct `curl` Command
    curl_cmd = [
        "curl", "-X", "POST",
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json",
        "-d", payload_json,
        api_endpoint
    ]

    # 🔹 Run the `curl` command
    try:
        print("🚀 Running `curl` command to merge version aliases...")
        result = subprocess.run(curl_cmd, capture_output=True, text=True, check=True)
        print(f"✅ Version aliases merged successfully: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Merging version aliases failed: {e.stderr}")

