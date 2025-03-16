"""
Google Cloud authentication utilities for Llama 3 fine-tuning
"""
import os
import json
import logging
from typing import Optional, Dict, Any

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import vertexai

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("auth_utils")


def save_credentials_file(credentials_data: Dict[str, Any], output_path: str) -> None:
    """
    Save credentials data to a JSON file.
    
    Args:
        credentials_data: Dictionary containing service account information
        output_path: Path to save the credentials file
    """
    with open(output_path, 'w') as f:
        json.dump(credentials_data, f, indent=2)
    
    logger.info(f"Credentials saved to {output_path}")


def setup_google_auth() -> Optional[Credentials]:
    """
    Set up Google Cloud authentication using credentials from the credentials folder.
    
    Returns:
        Google credentials object or None if application default
    """
    # Create credentials directory if it doesn't exist
    os.makedirs(config.CREDENTIALS_FOLDER, exist_ok=True)
    
    # Path for the service account key
    service_account_path = os.path.join(config.CREDENTIALS_FOLDER, "service-account.json")
    
    # Check if the provided dummy key needs to be saved to the credentials folder
    if not os.path.exists(service_account_path):
        logger.info(f"No service account key found at {service_account_path}")
        logger.info("If you have a JSON service account key, please save it to this location")
    
    # Try to find credentials
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        key_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        logger.info(f"Using credentials from environment variable: {key_path}")
    elif config.DEFAULT_CREDENTIALS_PATH:
        key_path = config.DEFAULT_CREDENTIALS_PATH
        # Set environment variable for other libraries that check it
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        logger.info(f"Using credentials from credentials folder: {key_path}")
    elif os.path.exists(service_account_path):
        key_path = service_account_path
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        logger.info(f"Using service account key: {key_path}")
    else:
        logger.warning("No explicit credentials found. Will attempt to use application default credentials.")
        key_path = None
    
    try:
        # If key path is provided, create credentials from file
        if key_path:
            # Read the file and check if it has required fields
            try:
                with open(key_path, 'r') as f:
                    creds_data = json.load(f)
                
                # Check for required fields
                required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
                missing_fields = [field for field in required_fields if field not in creds_data]
                
                if missing_fields:
                    logger.error(f"Credentials file missing required fields: {', '.join(missing_fields)}")
                    raise ValueError(f"Credentials file at {key_path} is missing required fields")
                
                # Create credentials
                credentials = Credentials.from_service_account_file(
                    key_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                
                if credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                
                # Update project ID in config if not set
                if config.PROJECT_ID == "your-project-id":
                    config.PROJECT_ID = creds_data['project_id']
                    logger.info(f"Updated project ID from credentials: {config.PROJECT_ID}")
                
                # Initialize Vertex AI with credentials
                vertexai.init(project=config.PROJECT_ID, location=config.REGION, credentials=credentials)
                logger.info(f"Authenticated using service account: {credentials.service_account_email}")
                return credentials
                
            except json.JSONDecodeError:
                logger.error(f"Credentials file at {key_path} is not valid JSON")
                raise
        else:
            # Use application default credentials
            vertexai.init(project=config.PROJECT_ID, location=config.REGION)
            logger.info("Authenticated using application default credentials")
            return None
            
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        
        # Provide more helpful information
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ and not config.DEFAULT_CREDENTIALS_PATH:
            logger.error(
                "No credentials found. Please either:\n"
                "1. Set the GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                "2. Place a service account key in the 'credentials' folder\n"
                "3. Or use application default credentials by running 'gcloud auth application-default login'"
            )
        
        raise


def verify_vertex_ai_access() -> bool:
    """
    Verify access to Vertex AI services.
    
    Returns:
        Boolean indicating if Vertex AI is accessible
    """
    try:
        # Try to list models to test access
        from vertexai.language_models import TextGenerationModel
        models = TextGenerationModel.list()
        logger.info(f"Successfully accessed Vertex AI. Found {len(models)} text generation models.")
        return True
    except Exception as e:
        logger.error(f"Error accessing Vertex AI: {str(e)}")
        return False


if __name__ == "__main__":
    # Simple test
    try:
        credentials = setup_google_auth()
        if credentials:
            print(f"Authenticated as: {credentials.service_account_email}")
        else:
            print("Using application default credentials")
        
        if verify_vertex_ai_access():
            print("Successfully verified Vertex AI access")
        else:
            print("Failed to access Vertex AI services")
    except Exception as e:
        print(f"Authentication failed: {str(e)}")