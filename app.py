import base64
import hashlib
import re

import requests
import streamlit as st

OLLAMA_API = "http://localhost:11434/api/generate"

MODEL_CHOICES = ["llava-phi3", "moondream", "qwen2.5vl"]

SUMMARY_PROMPT = (
    "You are an expert vision describer. Look ONLY at the image. "
    "Write a detailed description in EXACTLY TWO PARAGRAPHS totaling AT LEAST 500 words. "
    "Be precise about objects, layout, colors, lighting, background, relationships, and possible context. "
    "Avoid speculation beyond what is visible. Do not include lists or headings; use natural prose."
)

OCR_PROMPT = (
    "Transcribe ALL text visible in the image exactly as written — signs, labels, "
    "documents, screens, handwriting. Keep the original line breaks and reading order. "
    "Do not translate, summarise or describe the image. "
    "If there is no readable text, reply with exactly: NO_TEXT"
)

TRANSLATE_LANGUAGES = ["English", "Hindi", "Kannada", "Tamil", "Telugu", "Malayalam", "Marathi", "Bengali",
                       "Gujarati", "Spanish", "French", "German", "Japanese", "Chinese (Simplified)", "Arabic"]


def translate_prompt(text: str, language: str) -> str:
    return (f"Translate the following text into {language}. Keep the original line breaks, numbers, prices, "
            f"names and codes unchanged. Output ONLY the translation, written in {language} script, "
            f"with no notes or explanations.\n\nTEXT:\n{text}")


SUGGEST_PROMPT = (
    "Look at the image and write 5 short, specific questions a curious person might ask about it. "
    "Each question must be answerable from the image alone. "
    "Output one question per line, with no numbering and no extra text."
)


def query_ollama(model: str, prompt: str, image_bytes: bytes | None, num_predict: int = 900, timeout=300):
    payload = {
        "model": model,
        "prompt": prompt,
        # text-only requests (e.g. translation) send no image
        "images": [base64.b64encode(image_bytes).decode("utf-8")] if image_bytes else [],
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0.4,
        },
    }
    resp = requests.post(OLLAMA_API, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def safe_query(model, prompt, image_bytes, num_predict):
    try:
        return query_ollama(model, prompt, image_bytes, num_predict=num_predict)
    except requests.ConnectionError:
        st.error("Can't reach Ollama at localhost:11434. Start it with `ollama serve`.")
    except requests.Timeout:
        st.error("The model timed out. Try a smaller image or fewer max tokens.")
    except requests.HTTPError as e:
        st.error(f"Ollama error — is `{model}` pulled? (`ollama pull {model}`)\n\n{e}")
    return None


def parse_questions(text: str, limit: int = 5) -> list:
    """Turn the model's raw output into a clean list of questions."""
    questions = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if len(line) > 5 and line not in questions:
            questions.append(line if line.endswith("?") else line + "?")
    return questions[:limit]


def answer(model, question, image_bytes, max_tokens):
    with st.spinner("Answering..."):
        result = safe_query(
            model,
            f"Answer clearly and concisely based ONLY on the image. Question: {question}",
            image_bytes,
            min(400, max_tokens),
        )
    if result:
        st.session_state.answers.insert(0, (question, result))


st.set_page_config(page_title="Image Summary and question regarding the image", layout="centered")
st.title("Image Summary and question regarding the image")

with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Model", MODEL_CHOICES, index=0)
    max_tokens = st.slider("Max tokens (higher = longer, slower)", 200, 1400, 900, step=50)
    st.caption("Tip: For speed, use smaller images and reduce max tokens.")

source = st.radio("Image source", ["📁 Upload", "📷 Camera"], horizontal=True, label_visibility="collapsed")
if source == "📁 Upload":
    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
else:
    # Works on phones and laptops; the browser asks for camera permission.
    # Handy for signs, menus, labels and documents in front of you.
    uploaded = st.camera_input("Take a photo")
    if uploaded is not None and not getattr(uploaded, "name", None):
        uploaded.name = "camera.jpg"

if not uploaded:
    st.info("Upload an image or take a photo to begin.")
    st.stop()

image_bytes = uploaded.getvalue()  # getvalue() is safe across reruns; read() can return b"" the second time
image_id = hashlib.sha1(image_bytes).hexdigest()
if st.session_state.get("image_id") != image_id:
    st.session_state.image_id = image_id
    st.session_state.suggestions = []
    st.session_state.answers = []
    st.session_state.ocr_text = None
    st.session_state.ocr_translation = None

st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

# --- Summary ---
if st.button("📝 Summarize (2 paragraphs, ≥500 words)"):
    with st.spinner("Summarizing..."):
        summary = safe_query(model, SUMMARY_PROMPT, image_bytes, max_tokens)
    if summary:
        st.subheader("Summary")
        st.write(summary)

# --- Text extraction (+ translation) ---
# Kept in session state so it survives the reruns caused by the translate controls.
if st.button("🔤 Extract text from image"):
    with st.spinner("Reading text..."):
        text = safe_query(model, OCR_PROMPT, image_bytes, max_tokens)
    if text is not None:
        cleaned = text.strip().strip("`").strip()
        st.session_state.ocr_text = "" if (not cleaned or cleaned.upper().startswith("NO_TEXT")) else cleaned
        st.session_state.ocr_translation = None

ocr_text = st.session_state.get("ocr_text")
if ocr_text == "":
    st.info("No readable text found in this image.")
elif ocr_text:
    stem = uploaded.name.rsplit(".", 1)[0]
    st.subheader("Extracted text")
    st.code(ocr_text, language=None)  # code block has a built-in copy button
    st.download_button("⬇️ Download as .txt", ocr_text, file_name=f"{stem}-text.txt", mime="text/plain")
    st.caption("Read by the vision model — double-check numbers and names. "
               "qwen2.5vl is the most accurate for small or dense text.")

    t1, t2 = st.columns([2, 1])
    target = t1.selectbox("Translate to", TRANSLATE_LANGUAGES, label_visibility="collapsed")
    if t2.button("🌐 Translate", use_container_width=True):
        with st.spinner(f"Translating to {target}..."):
            out = safe_query(model, translate_prompt(ocr_text, target), None, max(400, len(ocr_text) // 2))
        if out:
            st.session_state.ocr_translation = (target, out.strip())
    tr = st.session_state.get("ocr_translation")
    if tr:
        lang, translated = tr
        st.subheader(f"Translation · {lang}")
        st.code(translated, language=None)
        st.download_button(f"⬇️ Download {lang} .txt", translated,
                           file_name=f"{stem}-{lang.split()[0].lower()}.txt", mime="text/plain")

st.divider()

# --- Suggested questions ---
st.subheader("💡 Not sure what to ask?")
if st.button("Suggest questions about this image"):
    with st.spinner("Looking for interesting questions..."):
        raw = safe_query(model, SUGGEST_PROMPT, image_bytes, 200)
    if raw:
        st.session_state.suggestions = parse_questions(raw)
        if not st.session_state.suggestions:
            st.warning("The model didn't return usable questions — try again or switch models.")

for i, suggestion in enumerate(st.session_state.suggestions):
    if st.button(suggestion, key=f"suggest-{i}"):
        answer(model, suggestion, image_bytes, max_tokens)

# --- Free-form question ---
st.subheader("🔎 Ask your own question")
q = st.text_input("Ask a question about the image:")
if st.button("Get Answer", disabled=not q.strip()):
    answer(model, q, image_bytes, max_tokens)

for question, reply in st.session_state.answers:
    st.markdown(f"**Q: {question}**")
    st.write(reply)
