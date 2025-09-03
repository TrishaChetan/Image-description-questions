# Image Summary and Question Regarding the Image
This project is a Streamlit application that allows you to:
- Upload an image
- Get a detailed 2-paragraph summary of the image (at least 500 words)
- Ask questions about the image and receive accurate answers

It uses the Ollama platform with vision-capable models (`llava-phi3`, `moondream`, or `qwen2.5vl`) to analyze the uploaded image and generate responses.

## How to Run
1. Clone this repository or download the files.
2. Install the required dependencies:pip install -r requirements.txt
3. Make sure Ollama is installed and running on your system.
4. Pull a vision-capable model (recommended: `llava-phi3`):ollama pull llava-phi3
5. Start the application:streamlit run app.py

