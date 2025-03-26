import os
import json
import subprocess

def set_model_iam_policy():
    """
    Sets IAM policy for a registered Vertex AI model using curl.
    """

    # 🔹 Get required environment variables
    project = os.getenv("GCP_PROJECT")
    region = os.getenv("GCP_REGION")
    model_id = os.getenv("MODEL_ID")  # Retrieved from the previous step
    access_token = os.getenv("ACCESS_TOKEN")  # Workload Identity or OAuth2 token
    iam_role = os.getenv("IAM_ROLE")  # User-provided IAM role
    iam_members = os.getenv("IAM_MEMBERS")  # User-provided IAM members (comma-separated)

    # 🔹 Validate inputs
    if not project or not region or not model_id or not access_token or not iam_role or not iam_members:
        raise ValueError("❌ GCP_PROJECT, GCP_REGION, MODEL_ID, ACCESS_TOKEN, IAM_ROLE, and IAM_MEMBERS must be set")

    # 🔹 Convert IAM members input (comma-separated) into a JSON array
    members_list = [member.strip() for member in iam_members.split(",")]  # Convert to list

    # 🔹 Construct API Endpoint
    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}:setIamPolicy"

    # 🔹 Construct IAM Policy JSON
    policy_payload = {
        "policy": {
            "bindings": [
                {
                    "role": iam_role,  # ✅ Use dynamic role from user input
                    "members": members_list  # ✅ Use dynamic members from user input
                }
            ]
        }
    }

    policy_json = json.dumps(policy_payload)

    # 🔹 Construct `curl` Command
    curl_cmd = [
        "curl", "-X", "POST",
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json",
        "-d", policy_json,
        api_endpoint
    ]

    # 🔹 Execute the request
    try:
        print("🚀 Setting IAM policy for the model...")
        result = subprocess.run(curl_cmd, capture_output=True, text=True, check=True)
        print(f"✅ IAM Policy set successfully: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to set IAM Policy: {e.stderr}")
