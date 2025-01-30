import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Together AI API Key (set as an environment variable)
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

# Index path
INDEX_PATH = os.environ.get("INDEX_PATH", "data_out")

#LLM model name
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

# Output directory
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data_out")

# Logging level
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Set up basic logging configuration
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Log to file
        logging.StreamHandler()          # Log to console
    ]
)