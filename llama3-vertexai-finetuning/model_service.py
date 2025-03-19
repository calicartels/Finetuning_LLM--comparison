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
    """
    Load both base and fine-tuned models using endpoints.
    
    Returns:
        Dictionary of model_type -> model info
    """
    global MODELS
    
    if not MODELS:
        try:
            setup_google_auth()
            
            aiplatform.init(project=config.PROJECT_ID, location=config.REGION)
            
            # Use your identified endpoints
            base_endpoint_id = "1124814688866009088"  
            finetuned_endpoint_id = "1677525990009470976"  
            
            # Load the base model endpoint
            try:
                base_endpoint = aiplatform.Endpoint(base_endpoint_id)
                MODELS["base"] = {
                    "endpoint": base_endpoint,
                    "name": "Llama 3 8B Chat"
                }
                logger.info(f"Loaded base model endpoint: {base_endpoint_id}")
            except Exception as e:
                logger.error(f"Error loading base model endpoint: {str(e)}")
            
            try:
                finetuned_endpoint = aiplatform.Endpoint(finetuned_endpoint_id)
                MODELS["finetuned"] = {
                    "endpoint": finetuned_endpoint,
                    "name": "Fine-tuned Llama 3"
                }
                logger.info(f"Loaded fine-tuned model endpoint: {finetuned_endpoint_id}")
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
    """Generate text using the fine-tuned model"""
    # Set default parameter values
    max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
    temperature = temperature or config.TEMPERATURE
    
    # Make sure models are loaded
    models = load_models()
    
    if model_type not in models:
        return {"error": f"Model type '{model_type}' not available"}
    
    model_info = models[model_type]
    
    formatted_prompt = prompt
    
    if "-" in prompt and not "Story:" in prompt:
        parts = prompt.split("-", 1)
        main_story = parts[0].strip()
        
        # Extract all bullet points as clues
        clues = []
        remaining_text = "-" + parts[1]
        bullet_points = remaining_text.split("\n- ")
        
        for point in bullet_points:
            if point.strip():
                if point.startswith("-"):
                    point = point[1:].strip()
                clues.append(f"'{point.strip()}'")
        
        clues_str = "[" + ", ".join(clues) + "]"
        formatted_prompt = f"Solve this logic puzzle:\nStory: {main_story}\nClues: {clues_str}"
    elif "Story:" in prompt and "Clues:" in prompt:
        if not prompt.startswith("Solve this logic puzzle:"):
            formatted_prompt = f"Solve this logic puzzle:\n{prompt}"
    elif not prompt.startswith("Solve this logic puzzle:"):
        formatted_prompt = f"Solve this logic puzzle:\nStory: {prompt}\nClues: []"
    
    print(f"Formatted prompt: {formatted_prompt[:200]}...")
    
    try:
        start_time = time.time()
        
        endpoint = model_info["endpoint"]
        
        prediction = endpoint.predict(instances=[{"prompt": formatted_prompt}])
        
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
        
        end_time = time.time()
        
        display_response = response_text
        try:
            import json
            display_response = json.dumps(json.loads(response_text), indent=2)
        except:
            pass
        
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
    Since we only have the fine-tuned model, we'll use it twice.
    
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
    
    # Check if finetuned model is available
    if "finetuned" not in models:
        return {"error": "Fine-tuned model not available"}
    
    finetuned_response = generate_text(prompt, "finetuned", max_tokens, temperature, top_k, top_p)
    
    base_response = generate_text(prompt, "finetuned", max_tokens, 
                                 temperature * 1.5 if temperature else 0.5,  
                                 top_k, top_p)
    
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