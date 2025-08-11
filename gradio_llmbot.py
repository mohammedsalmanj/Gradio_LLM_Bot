import subprocess  # To run system commands like pip install
import sys         # To access the current Python interpreter path

def install(package):
    """
    Install a Python package using pip.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    # Runs: python -m pip install <package> using current Python interpreter

packages = ["transformers", "gradio", "torch"]  # List of required packages

for pkg in packages:
    try:
        __import__(pkg)  # Try to import the package
    except ImportError:
        print(f"Package {pkg} not found. Installing...")
        install(pkg)    # If not found, install it using pip

from transformers import pipeline  # Import text generation pipeline from transformers
import gradio as gr                 # Import Gradio for UI

generator = pipeline('text-generation', model='gpt2')
# Initialize GPT-2 text generation model pipeline

def generate_text(prompt):
    outputs = generator(prompt, max_length=50, num_return_sequences=1)
    # Generate text with max length 50 tokens, single output
    return outputs[0]['generated_text']  # Return generated text string

iface = gr.Interface(
    fn=generate_text,                      # Function called on user input
    inputs=gr.Textbox(lines=2, placeholder="Type your prompt here..."),  # Text input box
    outputs="text",                        # Output displayed as text
    title="Local GPT-2 Text Generator",  # Title of the UI
    description="Generates text locally without any account."  # Short description
)

iface.launch()  # Start Gradio web interface (opens in browser)

# You can now access the Gradio interface at http://127.0.0.1:7860

