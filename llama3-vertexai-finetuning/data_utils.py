"""
Dataset handling utilities for Llama 3 fine-tuning
"""
import os
import json
import logging
from typing import Dict, List, Tuple, Any

import pandas as pd
from datasets import load_dataset
from google.cloud import storage

import config
from auths import setup_google_auth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_utils")


def load_logic_puzzle_dataset() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Load the LogicPuzzleBaron dataset from Hugging Face and prepare it for Vertex AI fine-tuning.
    
    Returns:
        Tuple containing (train_data, eval_data) as lists of dictionaries
    """
    logger.info(f"Loading dataset: {config.DATASET_ID}")
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(config.DATASET_ID)
        logger.info(f"Dataset loaded with {len(dataset['train'])} examples")
        
        # Format for instruction fine-tuning
        formatted_data = []
        for item in dataset["train"]:
            answer = item.get('label_a', 'Answer not available')
            # Convert answer to string if it's not already
            if not isinstance(answer, str):
                answer = json.dumps(answer)
                
            formatted_data.append({
                "input_text": f"Solve this logic puzzle:\nStory: {item.get('story', '')}\nClues: {item.get('clues', '')}",
                "output_text": answer
            })            app.run(port=5001)
        
        # Split into train and evaluation sets
        train_size = int(len(formatted_data) * config.TRAIN_TEST_SPLIT)
        train_data = formatted_data[:train_size]
        eval_data = formatted_data[train_size:]
        
        logger.info(f"Split into {len(train_data)} training and {len(eval_data)} evaluation examples")
        
        # Save to local files
        save_to_jsonl(train_data, config.TRAIN_DATA_PATH)
        save_to_jsonl(eval_data, config.EVAL_DATA_PATH)
        
        # Also save some sample puzzles for testing
        save_sample_puzzles(dataset["train"][:10])
        
        return train_data, eval_data
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def save_to_jsonl(data: List[Dict[str, str]], output_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        output_path: Path to save the JSONL file
    """
    logger.info(f"Saving {len(data)} examples to {output_path}")
    
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def upload_to_gcs(local_path: str, gcs_path: str) -> str:
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        local_path: Local file path
        gcs_path: GCS path (without gs:// prefix)
        
    Returns:
        Full GCS URI of the uploaded file
    """
    logger.info(f"Uploading {local_path} to GCS: {gcs_path}")
    
    try:
        # Make sure we're authenticated
        setup_google_auth()
        
        client = storage.Client()
        bucket = client.bucket(config.BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        
        full_uri = f"gs://{config.BUCKET_NAME}/{gcs_path}"
        logger.info(f"Uploaded to {full_uri}")
        return full_uri
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}")
        raise


def prepare_data_for_vertex_ai() -> Tuple[str, str]:
    """
    Prepare dataset for Vertex AI fine-tuning by loading data,
    formatting it, and uploading to GCS.
    
    Returns:
        Tuple of (train_data_uri, eval_data_uri) as GCS URIs
    """
    # Load and process data
    train_data, eval_data = load_logic_puzzle_dataset()
    
    # Upload to GCS
    train_data_uri = upload_to_gcs(
        config.TRAIN_DATA_PATH, 
        f"data/{os.path.basename(config.TRAIN_DATA_PATH)}"
    )
    
    eval_data_uri = upload_to_gcs(
        config.EVAL_DATA_PATH, 
        f"data/{os.path.basename(config.EVAL_DATA_PATH)}"
    )
    
    return train_data_uri, eval_data_uri


def save_sample_puzzles(puzzle_examples: List[Dict[str, Any]]) -> None:
    """
    Save sample puzzles for testing and visualization.
    
    Args:
        puzzle_examples: List of puzzle examples from the dataset
    """
    samples = {}
    
    for i, example in enumerate(puzzle_examples):
        puzzle_name = f"Puzzle {i+1}"
        
        # Handle the case where example is a string
        if isinstance(example, str):
            samples[puzzle_name] = {
                "question": example,
                "answer": "No answer available"
            }
        # Handle the case where example is a dictionary
        else:
            samples[puzzle_name] = {
                "question": example.get("Question", example.get("story", "No question available")),
                "answer": example.get("Answer", example.get("label_a", "No answer available"))
            }
    
    with open(config.SAMPLE_PUZZLES_PATH, 'w') as f:
        json.dump(samples, f, indent=2)
    
    logger.info(f"Saved {len(samples)} sample puzzles to {config.SAMPLE_PUZZLES_PATH}")


def load_sample_puzzles() -> Dict[str, Dict[str, str]]:
    """
    Load sample puzzles for testing.
    
    Returns:
        Dictionary of puzzle name to puzzle content
    """
    try:
        with open(config.SAMPLE_PUZZLES_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # If file doesn't exist, create some default samples
        samples = {}
        for i, prompt in enumerate(config.SAMPLE_PROMPTS):
            samples[f"Puzzle {i+1}"] = {
                "question": prompt,
                "answer": "Sample answer not available"
            }
        return samples


if __name__ == "__main__":
    # Quick test of the functions
    print("Testing dataset utilities...")
    
    # Make sure we're authenticated
    setup_google_auth()
    
    train_data, eval_data = load_logic_puzzle_dataset()
    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(eval_data)} evaluation examples")
    
    # Print a sample
    print("\nSample training example:")
    print(json.dumps(train_data[0], indent=2))
    
    print("\nSample puzzles:")
    samples = load_sample_puzzles()
    for name, content in list(samples.items())[:2]:
        print(f"\n{name}:")
        print(content["question"][:200] + "...")