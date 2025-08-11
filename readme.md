pip install virtualenv 
virtualenv my_env # create a virtual environment named my_env
source my_env/bin/activate # activate my_env


README = """
# Local GPT-2 Text Generator with Gradio

This project provides a simple local text generation tool using Hugging Face's GPT-2 model and a Gradio web interface. It automatically installs the required Python packages if they are missing.

## Features

- Uses GPT-2 for text generation locally (no cloud or API keys required)
- Auto-installs dependencies: transformers, gradio, and torch
- Easy-to-use web UI powered by Gradio
- Generates text up to 50 tokens based on your input prompt

## Requirements

- Python 3.7 or higher
- Internet connection (for first-time package installation and model download)

## Usage

1. Clone the repository or download the script.
2. Run the script:
   ```bash
   python gradio_llmbot.py
