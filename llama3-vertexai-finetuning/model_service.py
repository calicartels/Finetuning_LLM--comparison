"""
Model serving utilities for Llama 3
"""
import os
import time
import logging
import threading
from typing import Dict, Any, List, Optional

import vertexai
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel

import config
from auths import setup_google_auth

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
            setup_google_auth()
            aiplatform.init(project=config.PROJECT_ID, location=config.REGION)
            
            # Load base model endpoint
            try:
                base_endpoint = aiplatform.Endpoint(config.BASE_ENDPOINT_ID)
                MODELS["base"] = {
                    "endpoint": base_endpoint,
                    "name": "Llama 3 8B Chat"
                }
                logger.info(f"Loaded base model endpoint: {config.BASE_ENDPOINT_ID}")
            except Exception as e:
                logger.error(f"Error loading base model endpoint: {str(e)}")
            
            # Load fine-tuned model endpoint
            try:
                finetuned_endpoint = aiplatform.Endpoint(config.FINETUNED_ENDPOINT_ID)
                MODELS["finetuned"] = {
                    "endpoint": finetuned_endpoint,
                    "name": "Fine-tuned Llama 3"
                }
                logger.info(f"Loaded fine-tuned model endpoint: {config.FINETUNED_ENDPOINT_ID}")
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
    max_tokens = max_tokens or 2048  # Increased from default
    temperature = temperature or config.TEMPERATURE
    
    # Make sure models are loaded
    models = load_models()
    
    if model_type not in models:
        return {"error": f"Model type '{model_type}' not available"}
    
    model_info = models[model_type]
    
    # Format the prompt - simplified for LOGIC-701 format
    formatted_prompt = prompt
    if not prompt.startswith("Solve this logic puzzle:"):
        formatted_prompt = f"Solve this logic puzzle:\n{prompt}"
    
    logger.info(f"Formatted prompt: {formatted_prompt[:200]}...")
    
    try:
        start_time = time.time()
        
        endpoint = model_info["endpoint"]
        
        # Pass parameters explicitly to ensure they're used
        prediction = endpoint.predict(
            instances=[{"prompt": formatted_prompt}],
            parameters={
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topK": top_k or config.TOP_K,
                "topP": top_p or config.TOP_P
            }
        )
        
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
        
        # Log the complete raw response
        logger.info(f"COMPLETE RESPONSE: {response_text}")
        
        # Process response - extract content after "Output:" if present
        display_response = response_text
        if "Output:" in display_response:
            try:
                output_parts = display_response.split("Output:")
                if len(output_parts) > 1:
                    display_response = output_parts[1].strip()
            except Exception as e:
                logger.error(f"Error extracting output: {str(e)}")
        
        end_time = time.time()
        
        return {
            "full_text": formatted_prompt + "\n\nOutput: " + display_response,
            "response_only": display_response,
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
            "temperature": temperature or config.TEMPERATURE,
            "max_tokens": max_tokens or config.MAX_OUTPUT_TOKENS,
            "top_k": top_k or config.TOP_K,
            "top_p": top_p or config.TOP_P,
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
            "name": config.TUNED_MODEL_DISPLAY_NAME,
            "type": "finetuned",
            "description": "Fine-tuned Llama 3 model for logic puzzles"
        })
    
    return model_info


if __name__ == "__main__":
    setup_google_auth()
    
    print("Loading models...")
    available_models = load_models()
    print(f"Available models: {list(available_models.keys())}")
    
    test_prompt = config.SAMPLE_PROMPTS[0]
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