"""
Flask API for serving Llama 3 models
"""
import os
import json
import logging
import traceback
from typing import Dict, Any
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluations import evaluate_model, compare_models
from config.config import *
from auth.auths import setup_google_auth
from src.model_service import load_models, generate_text, compare_models, get_available_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Initialize Flask app
app = Flask(__name__, static_folder='visualization')
CORS(app)  # Enable CORS for all routes

# Initialize on startup
setup_google_auth()
logger.info("Initialized Google Cloud authentication")


@app.route('/')
def index():
    """Serve the web interface"""
    return send_from_directory('visualization', 'index.html')


@app.route('/health')
def health_check():
    """Check API health status"""
    try:
        # Try to load models
        models = load_models()
        available_models = list(models.keys())
        
        return jsonify({
            "status": "healthy",
            "models": available_models,
            "project_id": PROJECT_ID,
            "region": REGION
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "models": []
        }), 500


@app.route('/models')
def get_models():
    """Get information about available models"""
    try:
        model_info = get_available_models()
        return jsonify({"models": model_info})
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    """Generate text from a specified model"""
    try:
        data = request.json
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing required field: prompt"}), 400
        
        prompt = data['prompt']
        model_type = data.get('model_type', 'base')
        max_tokens = data.get('max_tokens', MAX_OUTPUT_TOKENS)
        temperature = data.get('temperature', TEMPERATURE)
        top_k = data.get('top_k', TOP_K)
        top_p = data.get('top_p', TOP_P)
        
        response = generate_text(
            prompt=prompt,
            model_type=model_type,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        if 'error' in response:
            return jsonify(response), 400
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/debug-raw-response', methods=['POST'])
def debug_raw_response():
    """Get raw model response for debugging"""
    try:
        data = request.json
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing required field: prompt"}), 400
        
        prompt = data['prompt']
        model_type = data.get('model_type', 'base')
        
        # Load models
        models = load_models()
        
        if model_type not in models:
            return jsonify({"error": f"Model type '{model_type}' not available"})
        
        model_info = models[model_type]
        endpoint = model_info["endpoint"]
        
        # Format the prompt
        formatted_prompt = prompt
        if not prompt.startswith("Solve this logic puzzle:"):
            formatted_prompt = f"Solve this logic puzzle:\n{prompt}"
        
        # Get raw prediction with increased max tokens
        prediction = endpoint.predict(
            instances=[{"prompt": formatted_prompt}],
            parameters={
                "maxOutputTokens": 8192,  # Try maximum possible
                "temperature": 0.7,       # Slightly higher temperature
                "topK": 40,
                "topP": 0.95              # Higher top_p
            }
        )
        
        # Extract the complete raw response
        response_text = ""
        if hasattr(prediction, 'predictions') and isinstance(prediction.predictions, list):
            raw_response = prediction.predictions[0]
            if isinstance(raw_response, str):
                response_text = raw_response
                
        return jsonify({
            "raw_response": response_text,
            "response_length": len(response_text),
            "prediction_type": str(type(prediction)),
            "has_predictions_attr": hasattr(prediction, "predictions")
        })
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500



@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluate model performance on LOGIC-701 dataset"""
    try:
        data = request.json
        
        model_type = data.get('model_type', 'finetuned')
        num_examples = data.get('num_examples', 10)
        
        results = evaluate_model(model_type=model_type, num_examples=num_examples)
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in evaluate endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/compare-models', methods=['POST'])
def compare_model_endpoint():
    """Compare base and fine-tuned models on LOGIC-701 dataset"""
    try:
        data = request.json
        
        num_examples = data.get('num_examples', 10)
        
        results = compare_models(num_examples=num_examples)
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in compare-models endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare():
    """Compare responses from both models"""
    try:
        data = request.json
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing required field: prompt"}), 400
        
        prompt = data['prompt']
        max_tokens = data.get('max_tokens', MAX_OUTPUT_TOKENS)
        temperature = data.get('temperature', TEMPERATURE)
        top_k = data.get('top_k', TOP_K)
        top_p = data.get('top_p', TOP_P)
        
        response = compare_models(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        if 'error' in response:
            return jsonify(response), 400
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in compare endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/samples')
def get_samples():
    """Get sample puzzles for testing"""
    try:
        samples_path = SAMPLE_PUZZLES_PATH
        
        if os.path.exists(samples_path):
            with open(samples_path, 'r') as f:
                samples = json.load(f)
        else:
            # Return default samples from config
            samples = {
                f"Sample {i+1}": {
                    "question": prompt,
                    "answer": "Sample answer not available"
                }
                for i, prompt in enumerate(SAMPLE_PROMPTS)
            }
        
        return jsonify({"samples": samples})
    
    except Exception as e:
        logger.error(f"Error getting samples: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # Check for models at startup
    models = load_models()
    available_model_types = list(models.keys())
    logger.info(f"Available models: {available_model_types}")
    
    logger.info(f"Starting API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)