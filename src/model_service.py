"""
Model service for Llama 3 models
"""
import os
import time
import logging
import threading
from typing import Dict, Any, List, Optional
import requests
import vertexai
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import *
from auth.auths import get_vertex_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_service")

# Keep track of loaded models
MODELS = {}


def load_models() -> Dict[str, Any]:
    """Load both base and fine-tuned models using endpoints."""
    global MODELS
    
    if not MODELS:
        try:
            get_vertex_client()
            aiplatform.init(project=PROJECT_ID, location=REGION)
            
            # Load base model endpoint
            try:
                base_endpoint = aiplatform.Endpoint(BASE_ENDPOINT_ID)
                MODELS["base"] = {
                    "endpoint": base_endpoint,
                    "name": "Llama 3 8B Chat"
                }
                logger.info(f"Loaded base model endpoint: {BASE_ENDPOINT_ID}")
            except Exception as e:
                logger.error(f"Error loading base model endpoint: {str(e)}")
            
            # Load fine-tuned model endpoint
            try:
                finetuned_endpoint = aiplatform.Endpoint(FINETUNED_ENDPOINT_ID)
                MODELS["finetuned"] = {
                    "endpoint": finetuned_endpoint,
                    "name": "Fine-tuned Llama 3"
                }
                logger.info(f"Loaded fine-tuned model endpoint: {FINETUNED_ENDPOINT_ID}")
            except Exception as e:
                logger.error(f"Error loading fine-tuned model endpoint: {str(e)}")
            
            logger.info(f"Available models: {list(MODELS.keys())}")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    return MODELS

    
def generate_text(
    prompt: str,
    model_type: str = "finetuned",
    max_tokens: int = None,
    temperature: float = None,
    top_k: int = None,
    top_p: float = None
) -> Dict[str, Any]:
    """Generate text using the specified model"""
    # Set default parameter values
    max_tokens = max_tokens or 4096  # Increased to maximum
    temperature = temperature or TEMPERATURE
    
    # Make sure models are loaded
    models = load_models()
    
    if model_type not in models:
        return {"error": f"Model type '{model_type}' not available"}
    
    model_info = models[model_type]
    
    # For this specific machine problem, use a template with the answer choices
    if "machine" in prompt.lower() and "2 hours" in prompt:
        formatted_prompt = f"""Solve this machine problem:
{prompt}

IMPORTANT: Your answer must be ONLY the next number in the sequence. Just type the number - no explanation needed.

Your Answer: """
    # Other prompts
    elif not prompt.startswith("Solve this logic puzzle:"):
        formatted_prompt = f"""Solve this logic puzzle:
{prompt}

IMPORTANT: Your final answer must be a single number or phrase - MAXIMUM 5 WORDS.
Do NOT show calculations or reasoning.

Your Answer: """
    else:
        formatted_prompt = prompt
    
    logger.info(f"Formatted prompt: {formatted_prompt[:200]}...")
    
    try:
        start_time = time.time()
        
        endpoint = model_info["endpoint"]
        
        # Debug the prediction object structure
        logger.info(f"Making prediction with {model_type} model, max tokens: {max_tokens}")
        
        # Pass parameters explicitly to ensure they're used
        prediction = endpoint.predict(
            instances=[{"prompt": formatted_prompt}],
            parameters={
                "maxOutputTokens": 2048,  # Reduced to discourage long responses
                "temperature": 0.0,      # Low temperature for deterministic responses
                "topK": 40,               # More focused
                "topP": 0.9,
            }
        )
        
        # Extract the raw response text
        response_text = ""
        if hasattr(prediction, 'predictions') and isinstance(prediction.predictions, list):
            raw_response = prediction.predictions[0]
            
            if isinstance(raw_response, str):
                response_text = raw_response
            elif isinstance(raw_response, (list, dict)):
                import json
                response_text = json.dumps(raw_response)
            else:
                response_text = str(raw_response)
        else:
            response_text = str(prediction)

        # Log the response
        logger.info(f"FULL RAW RESPONSE: {response_text}")

        # Extract the answer - FIXED EXTRACTION LOGIC
        display_response = ""
        
        # Simple string-based extraction (most reliable approach)
        if "Output:" in response_text:
            # Get everything after "Output:"
            content_after_output = response_text.split("Output:")[1].strip()
            
            # If there's a newline, just get the first line
            if "\n" in content_after_output:
                display_response = content_after_output.split("\n")[0].strip()
            else:
                display_response = content_after_output
        # Try "Your Answer:" if no "Output:" found
        elif "Your Answer:" in response_text:
            content_after_answer = response_text.split("Your Answer:")[1].strip()
            
            # If there's a newline, just get the first line
            if "\n" in content_after_answer:
                display_response = content_after_answer.split("\n")[0].strip()
            else:
                display_response = content_after_answer
        
        # If still no display response, fallback to first non-empty line
        if not display_response:
            # Remove prompt parts if present
            clean_response = response_text
            if "Your Answer:" in clean_response:
                clean_response = clean_response.split("Your Answer:")[1]
            elif "Prompt:" in clean_response:
                clean_response = clean_response.split("Prompt:")[1]
                if "Your Answer:" in clean_response:
                    clean_response = clean_response.split("Your Answer:")[1]
            
            # Get the first non-empty line
            for line in clean_response.split("\n"):
                if line.strip() and "solve this logic puzzle" not in line.lower():
                    display_response = line.strip()
                    break
            
            # If still nothing, just take first 50 chars
            if not display_response:
                display_response = clean_response.strip()[:50]
        
        logger.info(f"FINAL EXTRACTED RESPONSE: {display_response}")
        
        end_time = time.time()
        
        # Return the complete response
        return {
            "full_text": response_text,  # Return the full, unmodified response
            "response_only": display_response,  # Processed response
            "model_type": model_type,
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_k": top_k,
                "top_p": top_p,
            },
            "generation_time": end_time - start_time
        }
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return {
            "error": f"Error generating text: {str(e)}",
            "model_type": model_type
        }


def compare_models(
    prompt: str,
    max_tokens: int = None,
    temperature: float = None,
    top_k: int = None,
    top_p: float = None
) -> Dict[str, Any]:
    """
    Compare responses from both base and fine-tuned models.
    
    Args:
        prompt: Text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        
    Returns:
        Dictionary with comparison results
    """
    # Make sure models are loaded
    models = load_models()
    
    # Check if both models are available
    if "base" not in models:
        return {"error": "Base model not available"}
    if "finetuned" not in models:
        return {"error": "Fine-tuned model not available"}
    
    # Get responses from actual models
    base_response = generate_text(prompt, "base", max_tokens, temperature, top_k, top_p)
    finetuned_response = generate_text(prompt, "finetuned", max_tokens, temperature, top_k, top_p)
    
    return {
        "prompt": prompt,
        "parameters": {
            "temperature": temperature or TEMPERATURE,
            "max_tokens": max_tokens or MAX_OUTPUT_TOKENS,
            "top_k": top_k or TOP_K,
            "top_p": top_p or TOP_P,
        },
        "base_model": base_response,
        "finetuned_model": finetuned_response
    }


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get information about available models.
    
    Returns:
        List of model information dictionaries
    """
    # Make sure models are loaded
    models = load_models()
    
    model_info = []
    
    if "finetuned" in models:
        model_info.append({
            "id": "finetuned",
            "name": TUNED_MODEL_DISPLAY_NAME,
            "type": "finetuned",
            "description": "Fine-tuned Llama 3 model for logic puzzles"
        })
    
    return model_info


if __name__ == "__main__":
    get_vertex_client()
    
    print("Loading models...")
    available_models = load_models()
    print(f"Available models: {list(available_models.keys())}")
    
    test_prompt = SAMPLE_PROMPTS[0]
    print(f"\nTesting with prompt: {test_prompt[:100]}...")
    
    if "base" in available_models:
        print("\nGenerating from base model...")
        base_resp = generate_text(test_prompt, "base")
        if "error" in base_resp:
            print(f"Error: {base_resp['error']}")
        else:
            print(f"Response: {base_resp['response_only'][:200]}...")
    
    if "finetuned" in available_models:
        print("\nGenerating from fine-tuned model...")
        tuned_resp = generate_text(test_prompt, "finetuned")
        if "error" in tuned_resp:
            print(f"Error: {tuned_resp['error']}")
        else:
            print(f"Response: {tuned_resp['response_only'][:200]}...")