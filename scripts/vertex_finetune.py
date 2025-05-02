"""
Fine-tuning script for Llama 3 using Vertex AI
"""
import os
import time
import argparse
import logging
from typing import Dict, Any, Optional

import vertexai
from vertexai.language_models import TextGenerationModel
from google.cloud import aiplatform

import config
from auths import setup_google_auth
from data_utils import prepare_data_for_vertex_ai, load_logic_puzzle_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vertex_finetune")


def create_tuning_job(
    train_data_uri: str,
    eval_data_uri: str,
    job_display_name: Optional[str] = None
) -> Any:  # Changed return type to Any to avoid the error
    """
    Create and start a Vertex AI tuning job for Llama 3.
    
    Args:
        train_data_uri: GCS URI for training data
        eval_data_uri: GCS URI for evaluation data
        job_display_name: Display name for the tuning job (optional)
        
    Returns:
        The Vertex AI tuning job object
    """
    if job_display_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_display_name = f"llama3_logic_puzzle_{timestamp}"
    
    logger.info(f"Creating tuning job: {job_display_name}")
    
    # Load the base model
    base_model = TextGenerationModel.from_pretrained(config.BASE_MODEL_ID)
    logger.info(f"Loaded base model: {config.BASE_MODEL_ID}")
    
    # Create the tuning job
    tuning_job = base_model.tune_model(
        training_data=train_data_uri,
        validation_data=eval_data_uri,
        # Training parameters
        train_steps=config.TRAIN_STEPS,
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        # Output model details
        tuned_model_display_name=config.TUNED_MODEL_DISPLAY_NAME,
    )
    
    logger.info(f"Tuning job created: {tuning_job.resource_name}")
    
    return tuning_job


def monitor_tuning_job(job_id: str) -> Dict[str, Any]:
    """
    Monitor a Vertex AI tuning job until completion.
    
    Args:
        job_id: The Vertex AI tuning job ID
        
    Returns:
        Dictionary with job status information
    """
    logger.info(f"Monitoring tuning job: {job_id}")
    
    client = aiplatform.gapic.JobServiceClient(
        client_options={"api_endpoint": f"{config.REGION}-aiplatform.googleapis.com"}
    )
    
    name = f"projects/{config.PROJECT_ID}/locations/{config.REGION}/tuningJobs/{job_id}"
    
    while True:
        response = client.get_tuning_job(name=name)
        state = response.state.name
        
        logger.info(f"Job state: {state}")
        
        if state == "JOB_STATE_SUCCEEDED":
            logger.info("Tuning job completed successfully!")
            return {
                "status": "completed",
                "model_resource_name": response.tuned_model,
                "training_steps_completed": response.training_steps_completed,
                "evaluation_metrics": response.evaluation_metrics,
            }
        
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
            logger.error(f"Tuning job failed or was cancelled: {response.error.message}")
            return {
                "status": "failed",
                "error": response.error.message,
            }
        
        # Wait before checking again
        time.sleep(60)  # Check every minute


def get_tuned_model(model_name: Optional[str] = None) -> TextGenerationModel:
    """
    Get a tuned model by name or use the default from config.
    
    Args:
        model_name: Name of the tuned model (optional, uses config if None)
        
    Returns:
        Vertex AI TextGenerationModel
    """
    model_name = model_name or config.TUNED_MODEL_DISPLAY_NAME
    
    logger.info(f"Loading tuned model: {model_name}")
    
    try:
        # Try to load by model name (display name)
        model = TextGenerationModel.list_tuned_models(
            base_model=config.BASE_MODEL_ID, 
            filter=f"display_name={model_name}"
        )
        
        if model:
            tuned_model = TextGenerationModel.get_tuned_model(model[0].name)
            logger.info(f"Found tuned model: {tuned_model.name}")
            return tuned_model
        else:
            logger.warning(f"No tuned model found with name: {model_name}")
            return None
    except Exception as e:
        logger.error(f"Error getting tuned model: {str(e)}")
        raise


def test_model(model: TextGenerationModel, prompt: str) -> str:
    """
    Test a model with a sample prompt.
    
    Args:
        model: Vertex AI TextGenerationModel
        prompt: Text prompt
        
    Returns:
        Generated response
    """
    logger.info(f"Testing model with prompt: {prompt[:50]}...")
    
    try:
        response = model.predict(
            prompt,
            temperature=config.TEMPERATURE,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
            top_k=config.TOP_K,
            top_p=config.TOP_P,
        )
        
        logger.info(f"Generated response: {response.text[:100]}...")
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}"


def fine_tune_model() -> Dict[str, Any]:
    """
    Run the complete fine-tuning workflow.
    
    Returns:
        Dictionary with information about the fine-tuning process
    """
    # Ensure we're authenticated
    setup_google_auth()
    
    # Prepare data
    train_data_uri, eval_data_uri = prepare_data_for_vertex_ai()
    
    # Create and run tuning job
    tuning_job = create_tuning_job(train_data_uri, eval_data_uri)
    
    # Get job ID from resource name
    # Format: projects/{project}/locations/{location}/tuningJobs/{tuning_job}
    job_id = tuning_job.resource_name.split('/')[-1]
    
    # Monitor job until completion
    job_result = monitor_tuning_job(job_id)
    
    return {
        "job_id": job_id,
        "status": job_result.get("status"),
        "model_name": config.TUNED_MODEL_DISPLAY_NAME,
        "train_data_uri": train_data_uri,
        "eval_data_uri": eval_data_uri,
    }


def main(args):
    """Main function to run fine-tuning based on command-line arguments"""
    # Ensure we're authenticated
    setup_google_auth()
    
    if args.prepare_only:
        # Just prepare the data
        logger.info("Preparing data only...")
        train_data_uri, eval_data_uri = prepare_data_for_vertex_ai()
        logger.info(f"Data prepared: {train_data_uri}, {eval_data_uri}")
    
    elif args.test_model:
        # Test an existing model
        if args.use_base_model:
            # Test base model
            logger.info(f"Testing base model: {config.BASE_MODEL_ID}")
            model = TextGenerationModel.from_pretrained(config.BASE_MODEL_ID)
        else:
            # Test tuned model
            model = get_tuned_model()
            
        if model:
            test_prompt = args.prompt or config.SAMPLE_PROMPTS[0]
            response = test_model(model, test_prompt)
            print("\n" + "=" * 80)
            print(f"Prompt: {test_prompt}")
            print("-" * 80)
            print(f"Response: {response}")
            print("=" * 80)
    
    else:
        # Run full fine-tuning
        logger.info("Starting fine-tuning process...")
        result = fine_tune_model()
        
        if result.get("status") == "completed":
            logger.info(f"Fine-tuning completed successfully. Model: {result['model_name']}")
        else:
            logger.error(f"Fine-tuning failed: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3 on Vertex AI")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare data without fine-tuning")
    parser.add_argument("--test-model", action="store_true", help="Test an existing model")
    parser.add_argument("--use-base-model", action="store_true", help="Test the base model instead of tuned model")
    parser.add_argument("--prompt", type=str, help="Test prompt for model testing")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Error in fine-tuning process: {str(e)}", exc_info=True)