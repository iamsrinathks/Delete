def set_model_iam_policy():
    """Sets IAM policy for the registered model."""
    project = os.getenv("GCP_PROJECT")
    region = os.getenv("GCP_REGION")
    model_id = os.getenv("MODEL_ID")  # User can provide this manually
    access_token = os.getenv("ACCESS_TOKEN")
    iam_role = os.getenv("IAM_ROLE")
    iam_members = os.getenv("IAM_MEMBERS")

    # Check for all required environment variables
    if not all([project, region, model_id, access_token, iam_role, iam_members]):
        raise ValueError("❌ Required variables missing. Ensure all required variables, including MODEL_ID, are provided.")

    # Prepare the IAM members list
    members_list = [member.strip() for member in iam_members.split(",")]

    api_endpoint = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/models/{model_id}:setIamPolicy"

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

    try:
        # Run the curl command to set the IAM policy
        result = subprocess.run(
            ["curl", "-X", "POST", "-H", f"Authorization: Bearer {access_token}", "-H", "Content-Type: application/json",
             "-d", json.dumps(policy_payload), api_endpoint],
            capture_output=True, text=True, check=True
        )

        # Print the result or output for success confirmation (optional)
        print(f"Response: {result.stdout}")

        print("✅ IAM Policy set successfully!")

    except subprocess.CalledProcessError as e:
        # Handle potential errors during the subprocess call
        print(f"❌ Failed to set IAM Policy. Error: {e.stderr}")
        raise
