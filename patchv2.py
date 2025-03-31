import os
import json

def patch_model():
    """Dynamically updates a Vertex AI model based on user-provided fields."""
    project = os.getenv("GCP_PROJECT")
    region = os.getenv("GCP_REGION")
    model_id = os.getenv("MODEL_ID")
    access_token = os.getenv("ACCESS_TOKEN")
    update_payload_str = os.getenv("MODEL_UPDATE_PAYLOAD")

    if not all([project, region, model_id, access_token, update_payload_str]):
        raise ValueError("❌ Missing required environment variables.")

    try:
        update_payload = json.loads(update_payload_str)  # Parse the user-provided JSON
    except json.JSONDecodeError:
        raise ValueError("❌ MODEL_UPDATE_PAYLOAD is not valid JSON.")

    # Generate the updateMask from JSON keys
    update_mask = ",".join(update_payload.keys())

    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}?updateMask={update_mask}"

    print(f"🚀 Updating Model: {model_id} with fields: {update_mask}")

    response = run_curl("PATCH", url, payload=update_payload)

    if response:
        print("✅ Model updated successfully!")
    else:
        print("❌ Model update failed.")

