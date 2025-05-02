# LLMs Finetuning Project

A Flask-based API service for fine-tuning and serving LLM models for logic puzzle solving.

## Project Structure

```
LLMs_Finetuning/
├── src/                # Main application code
│   ├── app.py         # Flask API endpoints
│   ├── model_service.py # Model serving utilities
│   └── evaluations.py # Model evaluation code
├── tests/             # Test files
├── scripts/           # Utility scripts
├── docs/              # Documentation
├── data/              # Data files
├── config/            # Configuration files
├── auth/              # Authentication related files
├── evaluation_results/# Model evaluation results
├── credentials/       # API credentials (gitignored)
└── visualization/     # Frontend visualization files
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google Cloud credentials in the credentials directory

3. Run the application:
```bash
python src/app.py
```

## API Endpoints

- `/` - Web interface
- `/health` - Health check
- `/models` - List available models
- `/generate` - Generate text from model
- `/evaluate` - Evaluate model performance
- `/compare-models` - Compare model performance
- `/samples` - Get sample puzzles

## Development

- Use `scripts/upload_dataset.py` to upload new training data
- Use `scripts/download_dataset.py` to download training data
- Use `vertex_finetune.py` for model fine-tuning