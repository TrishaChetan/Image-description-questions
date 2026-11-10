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

SUGGEST_PROMPT = (
    "Look at the image and write 5 short, specific questions a curious person might ask about it. "
    "Each question must be answerable from the image alone. "
    "Output one question per line, with no numbering and no extra text."
)


def query_ollama(model: str, prompt: str, image_bytes: bytes, num_predict: int = 900, timeout=300):
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image_bytes).decode("utf-8")],
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

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.info("Upload an image to begin.")
    st.stop()

image_bytes = uploaded.read()
image_id = hashlib.sha1(image_bytes).hexdigest()
if st.session_state.get("image_id") != image_id:
    st.session_state.image_id = image_id
    st.session_state.suggestions = []
    st.session_state.answers = []

st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

# --- Summary ---
if st.button("📝 Summarize (2 paragraphs, ≥500 words)"):
    with st.spinner("Summarizing..."):
        summary = safe_query(model, SUMMARY_PROMPT, image_bytes, max_tokens)
    if summary:
        st.subheader("Summary")
        st.write(summary)

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
