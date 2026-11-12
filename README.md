# Image Summary and Question Regarding the Image
This project is a Streamlit application that allows you to:
- Upload an image
- Get a detailed 2-paragraph summary of the image (at least 500 words)
- Ask questions about the image and receive accurate answers
- 📷 **Camera mode** — snap a photo with your phone or webcam instead of uploading (great for signs, menus and documents)
- 🔤 **Extract text** from photos of signs, receipts, notes or screenshots — copy it or download it as a .txt file
- 💡 Let the model **suggest questions** about the image — click one to get the answer
- Friendly error messages when Ollama isn't running or a model isn't pulled

It uses the Ollama platform with vision-capable models (`llava-phi3`, `moondream`, or `qwen2.5vl`) to analyze the uploaded image and generate responses.

## How to Run
1. Clone this repository or download the files.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure Ollama is installed and running on your system.
4. Pull a vision-capable model (recommended: `llava-phi3`):
   ```bash
   ollama pull llava-phi3
   ```
5. Start the application:
   ```bash
   streamlit run app.py
   ```

## Screenshots

See the `Sample image/` folder for example runs.

