"""
Configuration settings for Llama 3 fine-tuning on Vertex AI
"""
import os
from pathlib import Path

import os


# Google Cloud settings
PROJECT_ID = "aipi590-454522"
LOCATION = "us-central1"  
REGION = "us-central1"
BASE_MODEL_REGION = "us-central1"  # Region for the base model endpoint
BUCKET_NAME = "logic-puzzle-dataset"
BUCKET_URI = f"gs://{BUCKET_NAME}"


BASE_ENDPOINT_ID = "2082533297124016128"
FINETUNED_ENDPOINT_ID = "3447123984217276416"


# Path to credentials folder at root
CREDENTIALS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials")

# Explicitly set the path to your aipi590-454522 service account key
DEFAULT_CREDENTIALS_PATH = os.path.join(CREDENTIALS_FOLDER, "aipi590-454522-84ac69f97395.json")

# Will look for any .json file in the credentials folder
DEFAULT_CREDENTIALS_PATH = next((
    os.path.join(CREDENTIALS_FOLDER, f) for f in os.listdir(CREDENTIALS_FOLDER) 
    if f.endswith('.json')
), None) if os.path.exists(CREDENTIALS_FOLDER) else None

# Vertex AI Model settings
BASE_MODEL_ID = "projects/695116221974/locations/us-central1/models/7845865386670030848"
BASE_MODEL_DISPLAY_NAME = "Llama 3 8B"
TUNED_MODEL_DISPLAY_NAME = "llama3-logic-puzzle"

# Dataset settings
DATASET_ID = "hivaze/LOGIC-701"
TRAIN_TEST_SPLIT = 0.9

# Training hyperparameters
TRAIN_STEPS = 1000
LEARNING_RATE = 0.0002
BATCH_SIZE = 8

# Inference settings
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.2
TOP_K = 40
TOP_P = 0.8

# Local visualization settings
STREAMLIT_PORT = 8501
WEB_PORT = 8080

# Path settings
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Paths for local data
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.jsonl")
EVAL_DATA_PATH = os.path.join(DATA_DIR, "eval_data.jsonl")
SAMPLE_PUZZLES_PATH = os.path.join(DATA_DIR, "sample_puzzles.json")

# Example prompts for testing
SAMPLE_PROMPTS = [
    """Optimization of actions and planning
You have a list of tasks to complete: A, B, C, D, and E. Task A takes 1 hour, B takes 3 hours, C takes 2 hours, D takes 4 hours, and E takes 5 hours. You have two workers at your disposal. Each of the workers can only handle one task at a time, but tasks can be done in parallel. What is the minimum number of hours required to finish all tasks if the workers can work simultaneously and start at the same time?""",
]