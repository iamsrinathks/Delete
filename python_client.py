import os
from google.auth.credentials import Credentials
from google.cloud import storage  # Example: Google Cloud Storage Client

# Read the access token from the environment variable
access_token = os.environ.get("CLOUDSDK_AUTH_ACCESS_TOKEN")

if not access_token:
    raise ValueError("CLOUDSDK_AUTH_ACCESS_TOKEN is not set!")

# Create Google credentials using the access token
credentials = Credentials(token=access_token)

# Example: Use the credentials to access Google Cloud Storage
client = storage.Client(credentials=credentials)

# List all buckets in the project
buckets = list(client.list_buckets())
print("Buckets in the project:", [bucket.name for bucket in buckets])
