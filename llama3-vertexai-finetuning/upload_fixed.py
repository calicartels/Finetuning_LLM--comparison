import os
from google.cloud import storage

def upload_to_gcs(local_path, gcs_path, bucket_name="logic-puzzle-dataset"):
    """Upload a file to Google Cloud Storage"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    
    full_uri = f"gs://{bucket_name}/{gcs_path}"
    print(f"Uploaded to {full_uri}")
    return full_uri

# Paths
data_dir = "/Users/vishnumukundan/Documents/Duke Courses/Spring_Sem'25/LLMS/group_porject/LLMs_Finetuning/llama3-vertexai-finetuning/data"
train_fixed_path = os.path.join(data_dir, "train_data_fixed.jsonl")
eval_fixed_path = os.path.join(data_dir, "eval_data_fixed.jsonl")

# Upload to GCS
train_uri = upload_to_gcs(train_fixed_path, "data/train_data_fixed.jsonl")
eval_uri = upload_to_gcs(eval_fixed_path, "data/eval_data_fixed.jsonl")

print(f"Train data URI: {train_uri}")
print(f"Eval data URI: {eval_uri}")