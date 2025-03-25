import os
from google.cloud import storage
import config
from auths import setup_google_auth

def upload_to_gcs(local_path, gcs_path):
    """Upload a file to Google Cloud Storage"""
    print(f"Uploading {local_path} to GCS...")
    
    # Make sure the project ID matches your service account's project
    print(f"Using project: {config.PROJECT_ID}")
    client = storage.Client(project=config.PROJECT_ID)
    
    # Check if bucket exists, create if it doesn't
    try:
        bucket = client.get_bucket(config.BUCKET_NAME)
        print(f"Bucket {config.BUCKET_NAME} exists")
    except Exception as e:
        print(f"Bucket {config.BUCKET_NAME} doesn't exist, creating it...")
        print(f"Creating in region: {config.REGION}")
        try:
            bucket = client.create_bucket(config.BUCKET_NAME, location=config.REGION)
        except Exception as create_error:
            print(f"Error creating bucket: {create_error}")
            # Alternative: Try to use an existing bucket in your project
            print("Listing available buckets:")
            buckets = list(client.list_buckets())
            for b in buckets:
                print(f" - {b.name}")
            if buckets:
                print(f"Using existing bucket: {buckets[0].name}")
                bucket = buckets[0]
            else:
                raise Exception("No buckets available and couldn't create new one")
    
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    
    full_uri = f"gs://{bucket.name}/{gcs_path}"
    print(f"Uploaded to {full_uri}")
    return full_uri

# Make sure we're authenticated
credentials = setup_google_auth()
print(f"Project ID in config: {config.PROJECT_ID}")

if credentials:
    # If using service account, print the email to verify
    print(f"Using service account: {credentials.service_account_email}")
    # Extract project from service account email
    email_parts = credentials.service_account_email.split('@')
    if len(email_parts) > 1:
        domain = email_parts[1]
        if domain.endswith('.iam.gserviceaccount.com'):
            project = domain.split('.')[0]
            print(f"Service account project: {project}")
            # Update config project if needed
            if project != config.PROJECT_ID:
                print(f"Warning: Project mismatch. Updating config.PROJECT_ID from {config.PROJECT_ID} to {project}")
                config.PROJECT_ID = project

# Upload the dataset files
train_data_path = os.path.join(config.DATA_DIR, "train_data.jsonl")
eval_data_path = os.path.join(config.DATA_DIR, "eval_data.jsonl")

train_data_uri = upload_to_gcs(
    train_data_path, 
    f"data/{os.path.basename(train_data_path)}"
)

eval_data_uri = upload_to_gcs(
    eval_data_path, 
    f"data/{os.path.basename(eval_data_path)}"
)

print("\nDataset uploaded to GCS!")
print(f"Training data URI: {train_data_uri}")
print(f"Evaluation data URI: {eval_data_uri}")
print("\nYou can now run fine-tuning with: python vertex_finetune.py")