"""
Model serving utilities for Llama 3
"""
import os
import time
import logging
import threading
from typing import Dict, Any, List, Optional

import vertexai
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


def load_models() -> Dict[str, TextGenerationModel]:
    """
    Load both base and fine-tuned models.
    
    Returns:
        Dictionary of model_type -> model
    """
    global MODELS
    
    if not MODELS:
        try:
            # Make sure we're authenticated
            setup_google_auth()
            
            # Load base model
            logger.info(f"Loading base model: {config.BASE_MODEL_ID}")
            base_model = TextGenerationModel.from_pretrained(config.BASE_MODEL_ID)
            MODELS["base"] = base_model
            
            # Try to load fine-tuned model
            logger.info(f"Attempting to load fine-tuned model: {config.TUNED_MODEL_DISPLAY_NAME}")
            tuned_models = TextGenerationModel.list_tuned_models(
                base_model=config.BASE_MODEL_ID, 
                filter=f"display_name={config.TUNED_MODEL_DISPLAY_NAME}"
            )
            
            if tuned_models:
                tuned_model = TextGenerationModel.get_tuned_model(tuned_models[0].name)
                MODELS["finetuned"] = tuned_model
                logger.info(f"Loaded fine-tuned model: {tuned_model.name}")
            else:
                logger.warning(f"No fine-tuned model found with name: {config.TUNED_MODEL_DISPLAY_NAME}")
                logger.info("Only the base model will be available")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    return MODELS


def generate_text(
    prompt: str,
    model_type: str = "base",
    max_tokens: int = None,
    temperature: float = None,
    top_k: int = None,
    top_p: float = None
) -> Dict[str, Any]:
    """
    Generate text from the specified model.
    
    Args:
        prompt: Text prompt
        model_type: "base" or "finetuned"
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        
    Returns:
        Dictionary with generation results
    """
    # Set default parameter values
    max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
    temperature = temperature or config.TEMPERATURE
    top_k = top_k or config.TOP_K
    top_p = top_p or config.TOP_P
    
    # Make sure models are loaded
    models = load_models()
    
    if model_type not in models:
        return {
            "error": f"Model type '{model_type}' not available. Available models: {list(models.keys())}"
        }
    
    model = models[model_type]
    
    # Format prompt for logic puzzles
    if "logic puzzle" in prompt.lower() and not prompt.startswith("Solve this logic puzzle:"):
        formatted_prompt = f"Solve this logic puzzle: {prompt}"
    else:
        formatted_prompt = prompt
    
    try:
        # Generate response
        start_time = time.time()
        
        response = model.predict(
            formatted_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
        )
        
        end_time = time.time()
        
        return {
            "full_text": formatted_prompt + response.text,
            "response_only": response.text,
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
        return {
            "warning": "Fine-tuned model not available. Showing only base model response.",
            "prompt": prompt,
            "base_model": generate_text(prompt, "base", max_tokens, temperature, top_k, top_p),
            "finetuned_model": {"error": "Model not available"}
        }
    
    # Generate responses from both models
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
    
    if "base" in models:
        model_info.append({
            "id": "base",
            "name": config.BASE_MODEL_DISPLAY_NAME,
            "type": "base",
            "description": "Original Llama 3 8B model"
        })
    
    if "finetuned" in models:
        model_info.append({
            "id": "finetuned",
            "name": config.TUNED_MODEL_DISPLAY_NAME,
            "type": "finetuned",
            "description": "Fine-tuned Llama 3 8B for logic puzzles"
        })
    
    return model_info


if __name__ == "__main__":
    # Simple test of the model service
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