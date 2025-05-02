"""
Evaluation utilities for Llama 3 fine-tuning
"""
import logging
import os
import json
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import *
from src.model_service import generate_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evaluation")

# Suppress verbose logging from transformers
for module in ["transformers.tokenization_utils", 
               "transformers.configuration_utils", 
               "transformers.modeling_utils"]:
    logging.getLogger(module).setLevel(logging.ERROR)

try:
    from bert_score import score, plot_example
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer, util
    MODEL_IMPORTS_SUCCESSFUL = True
except ImportError:
    logger.warning("Some evaluation dependencies not installed. Run: pip install bert_score datasets sentence-transformers")
    MODEL_IMPORTS_SUCCESSFUL = False

def load_logic_dataset():
    """Load the LOGIC-701 dataset"""
    try:
        ds = load_dataset("hivaze/LOGIC-701", "en")
        train = pd.DataFrame(ds["train"])
        logger.info(f"Loaded LOGIC-701 dataset with {len(train)} examples")
        return train
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def calculate_bert_scores(model_outputs: List[str], 
                         reference_outputs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate BERT scores between model outputs and references"""
    if not MODEL_IMPORTS_SUCCESSFUL:
        logger.error("Cannot calculate BERT scores - required packages not installed")
        return None, None, None
        
    try:
        logger.info(f"Calculating BERT scores for {len(model_outputs)} examples")
        P, R, F1 = score(model_outputs, reference_outputs, lang="en", 
                         verbose=True, rescale_with_baseline=True)
        return P, R, F1
    except Exception as e:
        logger.error(f"Error calculating BERT scores: {str(e)}")
        return None, None, None

def calculate_embedding_similarity(model_outputs: List[str], 
                                  reference_outputs: List[str]) -> np.ndarray:
    """Calculate embedding similarity between model outputs and references"""
    if not MODEL_IMPORTS_SUCCESSFUL:
        logger.error("Cannot calculate embedding similarity - required packages not installed")
        return None
        
    try:
        # Load sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Compute embeddings
        logger.info(f"Calculating embeddings for {len(model_outputs)} examples")
        model_embeddings = model.encode(model_outputs, convert_to_tensor=True)
        ref_embeddings = model.encode(reference_outputs, convert_to_tensor=True)
        
        # Compute cosine similarity
        cos_sim_matrix = util.cos_sim(model_embeddings, ref_embeddings)
        
        # Extract diagonal (each output compared to its corresponding reference)
        similarities = [cos_sim_matrix[i][i].item() for i in range(len(model_outputs))]
        return similarities
    except Exception as e:
        logger.error(f"Error calculating embedding similarity: {str(e)}")
        return None

def evaluate_model(model_type: str = "finetuned", 
                  num_examples: int = 20,
                  save_results: bool = True) -> Dict[str, Any]:
    """
    Evaluate model performance on LOGIC-701 dataset
    
    Args:
        model_type: Type of model to evaluate ("base" or "finetuned")
        num_examples: Number of examples to evaluate
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary with evaluation results
    """
    # Load dataset
    dataset = load_logic_dataset()
    if dataset is None or len(dataset) == 0:
        return {"error": "Failed to load dataset"}
    
    # Limit to specified number of examples
    if num_examples > 0 and num_examples < len(dataset):
        dataset = dataset.sample(num_examples, random_state=42)
    
    # Generate responses for each example
    model_outputs = []
    reference_outputs = []
    
    logger.info(f"Generating responses using {model_type} model for {len(dataset)} examples")
    for i, row in dataset.iterrows():
        puzzle = row.get("puzzle", "") or row.get("problem_statement", "")
        reference = row.get("solution", "")
        
        if not puzzle or not reference:
            continue
            
        # Generate response
        response = generate_text(puzzle, model_type=model_type, temperature=0.1)
        
        # Skip if error in generation
        if "error" in response:
            logger.warning(f"Error generating response for example {i}: {response['error']}")
            continue
            
        model_output = response.get("response_only", "")
        
        model_outputs.append(model_output)
        reference_outputs.append(reference)
        
        # Log progress
        if (i + 1) % 5 == 0:
            logger.info(f"Generated {i+1}/{len(dataset)} responses")
    
    # Calculate BERT scores
    P, R, F1 = calculate_bert_scores(model_outputs, reference_outputs)
    
    # Calculate embedding similarity
    similarities = calculate_embedding_similarity(model_outputs, reference_outputs)
    
    # Prepare results
    results = {
        "model_type": model_type,
        "num_examples": len(model_outputs),
        "bert_precision": P.mean().item() if P is not None else None,
        "bert_recall": R.mean().item() if R is not None else None,
        "bert_f1": F1.mean().item() if F1 is not None else None,
        "embedding_similarity": np.mean(similarities) if similarities is not None else None,
        "examples": [
            {
                "puzzle": dataset.iloc[i].get("puzzle", "") or dataset.iloc[i].get("problem_statement", ""),
                "reference": reference_outputs[i],
                "model_output": model_outputs[i],
                "bert_f1": F1[i].item() if F1 is not None else None,
                "similarity": similarities[i] if similarities is not None else None
            }
            for i in range(len(model_outputs))
        ]
    }
    
    # Save results if requested
    if save_results:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"eval_{model_type}_{timestamp}.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_path}")
    
    return results

def compare_models(num_examples: int = 20) -> Dict[str, Any]:
    """
    Compare base and fine-tuned models on LOGIC-701 dataset
    
    Args:
        num_examples: Number of examples to evaluate
        
    Returns:
        Dictionary with comparison results
    """
    # Evaluate base model
    base_results = evaluate_model("base", num_examples)
    
    # Evaluate fine-tuned model
    finetuned_results = evaluate_model("finetuned", num_examples)
    
    # Combine results
    comparison = {
        "num_examples": num_examples,
        "base_model": {
            "bert_precision": base_results.get("bert_precision"),
            "bert_recall": base_results.get("bert_recall"),
            "bert_f1": base_results.get("bert_f1"),
            "embedding_similarity": base_results.get("embedding_similarity")
        },
        "finetuned_model": {
            "bert_precision": finetuned_results.get("bert_precision"),
            "bert_recall": finetuned_results.get("bert_recall"),
            "bert_f1": finetuned_results.get("bert_f1"),
            "embedding_similarity": finetuned_results.get("embedding_similarity")
        },
        "improvement": {
            "bert_precision": finetuned_results.get("bert_precision") - base_results.get("bert_precision") 
                if finetuned_results.get("bert_precision") and base_results.get("bert_precision") else None,
            "bert_recall": finetuned_results.get("bert_recall") - base_results.get("bert_recall")
                if finetuned_results.get("bert_recall") and base_results.get("bert_recall") else None,
            "bert_f1": finetuned_results.get("bert_f1") - base_results.get("bert_f1")
                if finetuned_results.get("bert_f1") and base_results.get("bert_f1") else None,
            "embedding_similarity": finetuned_results.get("embedding_similarity") - base_results.get("embedding_similarity")
                if finetuned_results.get("embedding_similarity") and base_results.get("embedding_similarity") else None
        }
    }
    
    return comparison

def plot_comparison(comparison_results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot comparison between base and fine-tuned models
    
    Args:
        comparison_results: Results from compare_models function
        save_path: Path to save the plot (optional)
    """
    if not comparison_results:
        logger.error("No comparison results to plot")
        return
    
    # Prepare data for plotting
    metrics = ["bert_precision", "bert_recall", "bert_f1", "embedding_similarity"]
    base_values = [comparison_results["base_model"].get(m, 0) for m in metrics]
    finetuned_values = [comparison_results["finetuned_model"].get(m, 0) for m in metrics]
    
    # Set up plot style
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, base_values, width, label='Base Model')
    plt.bar(x + width/2, finetuned_values, width, label='Fine-tuned Model')
    
    # Add labels and legend
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Model Comparison: Base vs. Fine-tuned')
    plt.xticks(x, ['BERT Precision', 'BERT Recall', 'BERT F1', 'Embedding Similarity'])
    plt.legend()
    plt.grid(True)
    
    # Add improvement percentages
    for i, metric in enumerate(metrics):
        if base_values[i] > 0:
            improvement = ((finetuned_values[i] - base_values[i]) / base_values[i]) * 100
            plt.text(i, max(base_values[i], finetuned_values[i]) + 0.02, 
                     f"{improvement:.1f}%", 
                     ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved comparison plot to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Test the evaluation functionality
    print("Testing evaluation module...")
    
    # Run a small evaluation (5 examples) on both models
    comparison = compare_models(5)
    
    # Print results
    print("\nComparison Results:")
    print(f"Base model BERT F1: {comparison['base_model']['bert_f1']:.4f}")
    print(f"Fine-tuned model BERT F1: {comparison['finetuned_model']['bert_f1']:.4f}")
    print(f"Improvement: {comparison['improvement']['bert_f1']:.4f}")
    
    # Plot comparison
    plot_comparison(comparison)