"""
Configuration settings for Llama 3 fine-tuning on Vertex AI
"""
import os
from pathlib import Path

import os

# Google Cloud settings
PROJECT_ID = "aipi590-454522"
LOCATION = "us-east1"
REGION = "us-east1"
BUCKET_NAME = "logic-puzzle-dataset"
BUCKET_URI = f"gs://{BUCKET_NAME}"

# Path to credentials folder at root
CREDENTIALS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials")

# Explicitly set the path to your aipi590-454522 service account key
DEFAULT_CREDENTIALS_PATH = os.path.join(CREDENTIALS_FOLDER, "aipi590-454522-84ac69f97395.json")

# Path to credentials folder at root
CREDENTIALS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials")
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
DATASET_ID = "olegbask/LogicPuzzleBaron"
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
    """Five friends (Alex, Brett, Charlie, Dana, and Emerson) each ordered a different drink (coffee, tea, soda, water, and juice). From the following clues, can you determine who ordered which drink?
- The person who ordered coffee sits between the people who ordered tea and juice.
- Dana sits next to the person who ordered water.
- Brett ordered soda.
- Charlie sits at one end of the table.
- Emerson sits next to Alex.""",
    
    """In a small town, there are only three barbers (Albert, Bernard, and Charles). From the following clues, determine which barber cuts whose hair:
- No barber cuts his own hair.
- Albert doesn't cut Bernard's hair.
- The person who cuts Charles's hair has his hair cut by the person whose hair is cut by Charles.
- Albert's hair is not cut by the person whose hair Albert cuts.""",
    
    """Five books (Art, Biology, Chemistry, Drama, and Economics) need to be arranged on a shelf. From the following clues, determine their order from left to right:
- Art and Drama must be separated by exactly one book.
- Biology must be somewhere to the right of Chemistry.
- Economics must be at one of the ends.
- The Biology book must be adjacent to exactly one other book."""
]