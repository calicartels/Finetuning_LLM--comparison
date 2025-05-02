import os
import json
import random
import logging
from datasets import load_dataset
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process_logic701")

# Create data directory if it doesn't exist
os.makedirs(config.DATA_DIR, exist_ok=True)

try:
    # Load the already-downloaded dataset
    logger.info("Loading LOGIC-701 dataset from Hugging Face...")
    dataset = load_dataset("hivaze/LOGIC-701", "en")
    
    logger.info(f"Dataset loaded with {len(dataset['train'])} examples")
    
    # Show example data structure
    if len(dataset['train']) > 0:
        example = dataset['train'][0]
        logger.info(f"Example fields: {list(example.keys())}")
        logger.info(f"Example problem: {example.get('problem_statement', '')[:100]}...")
    
    # Format for instruction fine-tuning
    logger.info("Formatting dataset for fine-tuning...")
    formatted_data = []
    
    for item in dataset['train']:
        # Extract problem and solution
        problem = item.get('problem_statement', '')
        solution = item.get('solution', '')
        
        formatted_data.append({
            "input_text": f"Solve this logic puzzle:\n{problem}",
            "output_text": solution
        })
    
    logger.info(f"Formatted {len(formatted_data)} examples")
    
    # Shuffle the data
    random.seed(42)  # For reproducibility
    random.shuffle(formatted_data)
    
    # Split into train and evaluation sets (90% train, 10% eval)
    train_size = int(len(formatted_data) * 0.9)
    train_data = formatted_data[:train_size]
    eval_data = formatted_data[train_size:]
    
    logger.info(f"Split into {len(train_data)} training and {len(eval_data)} evaluation examples")
    
    # Save to JSONL files
    train_data_path = os.path.join(config.DATA_DIR, "train_data.jsonl")
    eval_data_path = os.path.join(config.DATA_DIR, "eval_data.jsonl")
    
    # Save train data
    with open(train_data_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved training data to {train_data_path}")
    
    # Save eval data
    with open(eval_data_path, 'w') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved evaluation data to {eval_data_path}")
    
    # Save sample puzzles for testing
    sample_puzzles_path = os.path.join(config.DATA_DIR, "sample_puzzles.json")
    samples = {}
    
    # Get 10 random examples for samples
    sample_indices = random.sample(range(len(dataset['train'])), min(10, len(dataset['train'])))
    
    for i, idx in enumerate(sample_indices):
        example = dataset['train'][idx]
        puzzle_name = f"Puzzle {i+1}"
        samples[puzzle_name] = {
            "question": example.get("problem_statement", "No question available"),
            "answer": example.get("solution", "No answer available")
        }
    
    with open(sample_puzzles_path, 'w') as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Saved {len(samples)} sample puzzles to {sample_puzzles_path}")
    
    logger.info("Dataset processing complete!")
    
except Exception as e:
    logger.error(f"Error processing dataset: {e}", exc_info=True)