from pathlib import Path
import json
import pickle
import re
import shutil
import string
import tempfile

import contractions
import nltk
import gradio as gr

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


BASE_DIR = Path(__file__).resolve().parent


# ── NLTK + model loading ──────────────────────────────────────────────────────

def load_nltk_resources():
    """Download and load NLTK resources"""
    resources = {
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for package, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except (LookupError, OSError):
            nltk.download(package, quiet=True)

    return set(stopwords.words("english")), WordNetLemmatizer()


def normalize_keras_h5_config(obj):
    """Normalize Keras H5 config for compatibility"""
    drop_keys = {"optional", "quantization_config"}
    if isinstance(obj, dict):
        if obj.get("class_name") == "DTypePolicy":
            return obj.get("config", {}).get("name", "float32")
        config = obj.get("config")
        if obj.get("class_name") == "InputLayer" and isinstance(config, dict):
            if "batch_shape" in config:
                config["batch_input_shape"] = config.pop("batch_shape")
        return {k: normalize_keras_h5_config(v) for k, v in obj.items() if k not in drop_keys}
    if isinstance(obj, list):
        return [normalize_keras_h5_config(v) for v in obj]
    return obj


def make_compatible_h5_copy(model_path):
    """Create a compatible copy of the H5 model"""
    import h5py
    version_key = int(model_path.stat().st_mtime)
    compat_path = Path(tempfile.gettempdir()) / f"{model_path.stem}_tfkeras_{version_key}.h5"
    if compat_path.exists():
        return compat_path
    shutil.copy2(model_path, compat_path)
    with h5py.File(compat_path, "r+") as h5_file:
        raw_config = h5_file.attrs["model_config"]
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")
        model_config = normalize_keras_h5_config(json.loads(raw_config))
        h5_file.attrs.modify("model_config", json.dumps(model_config))
    return compat_path


def load_best_model(model_path):
    """Load the best model with compatibility handling"""
    try:
        return load_model(str(model_path), compile=False)
    except TypeError as error:
        message = str(error)
        if "batch_shape" not in message and "quantization_config" not in message:
            raise
        compatible_model_path = make_compatible_h5_copy(model_path)
        return load_model(str(compatible_model_path), compile=False)


class TokenizerUnpickler(pickle.Unpickler):
    """Custom unpickler for Keras tokenizer"""
    def find_class(self, module, name):
        if module == "keras.src.legacy.preprocessing.text" and name == "Tokenizer":
            return Tokenizer
        return super().find_class(module, name)


def load_saved_artifacts():
    """Load model, tokenizer, and max_length"""
    # Try different model file names
    model_files = ["good_sarcasm_model.h5"]
    model = None
    for model_file in model_files:
        model_path = BASE_DIR / model_file
        if model_path.exists():
            model = load_best_model(model_path)
            print(f"✓ Loaded model: {model_file}")
            break
    
    if model is None:
        raise FileNotFoundError("No model file found. Looking for: " + ", ".join(model_files))
    
    with open(BASE_DIR / "tokenizer.pkl", "rb") as f:
        tokenizer = TokenizerUnpickler(f).load()
    with open(BASE_DIR / "max_length.pkl", "rb") as f:
        max_length = pickle.load(f)
    
    return model, tokenizer, max_length


# Load resources
print("Loading NLTK resources...")
stop_words, lemmatizer = load_nltk_resources()
print("Loading model and artifacts...")
model, tokenizer, max_length = load_saved_artifacts()
print("✓ All resources loaded successfully!")


# ── Text processing ───────────────────────────────────────────────────────────

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = word_tokenize(text)
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned_words)


def predict_sarcasm(text):
    """Predict if text is sarcastic"""
    if not text or not text.strip():
        return {
            "error": "Please enter some text before running the model.",
            "prediction": "",
            "confidence": "",
            "raw_score": "",
            "cleaned_text": ""
        }
    
    cleaned = clean_text(text)
    
    if not cleaned:
        return {
            "error": "No tokens remained after preprocessing. Try different text.",
            "prediction": "",
            "confidence": "",
            "raw_score": "",
            "cleaned_text": ""
        }
    
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
    raw = model.predict(padded, verbose=0)[0][0]
    
    label = "Sarcastic" if raw >= 0.5 else "Not Sarcastic"
    confidence = raw if raw >= 0.5 else 1 - raw
    
    return {
        "error": "",
        "prediction": label,
        "confidence": f"{confidence * 100:.1f}%",
        "raw_score": f"{raw:.4f}",
        "cleaned_text": cleaned
    }


def analyze_text(text):
    """Main function for Gradio interface"""
    result = predict_sarcasm(text)
    
    if result["error"]:
        return result["error"], "", "", "", ""
    
    return (
        result["prediction"],
        result["confidence"],
        result["raw_score"],
        result["cleaned_text"],
        ""  # Clear error
    )


# ── Example sentences ─────────────────────────────────────────────────────────

EXAMPLES = [
    ["Local man thrilled to spend weekend resetting all his passwords."],
    ["The city council approved a new public library downtown."],
    ["Great, another meeting that definitely could not have been an email."],
    ["Scientists discover new species of butterfly in the Amazon rainforest."],
    ["Oh wonderful, my phone died right before my important presentation."],
]


# ── Gradio Interface ──────────────────────────────────────────────────────────

# Custom CSS for styling
custom_css = """
#main-container {
    max-width: 900px;
    margin: 0 auto;
}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

#title {
    text-align: center;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

#subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

#badge {
    text-align: center;
    color: #a78bfa;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
"""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("### LSTM · Word2Vec · NLP")
    gr.Markdown("# Sarcasm Detection")
    gr.Markdown(
        "Paste any headline, tweet, or sentence and let the model decide "
        "whether it's dripping with sarcasm — or perfectly sincere."
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Enter text to classify",
                placeholder="e.g. 'Oh great, the printer is out of ink again — what a surprise.'",
                lines=5
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Analyse Text", variant="primary")
            
            error_output = gr.Textbox(label="Status", visible=False)
    
    with gr.Row():
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction", interactive=False)
            confidence_output = gr.Textbox(label="Model Confidence", interactive=False)
        
        with gr.Column():
            raw_score_output = gr.Textbox(label="Raw Sarcasm Score", interactive=False)
            gr.Markdown("*Threshold: 0.5000 — scores above this are classified as sarcastic*")
    
    with gr.Row():
        cleaned_output = gr.Textbox(
            label="Cleaned & Tokenised Input",
            interactive=False,
            lines=3
        )
    
    gr.Markdown("---")
    gr.Markdown("### Try these examples:")
    gr.Examples(
        examples=EXAMPLES,
        inputs=input_text,
        label="Click an example to load it"
    )
    
    gr.Markdown("---")
    gr.Markdown(
        "<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>"
        "Text is cleaned, tokenised, and padded before being passed to an LSTM model<br/>"
        "trained on the News Headlines Dataset for Sarcasm Detection."
        "</div>"
    )
    
    # Event handlers
    submit_btn.click(
        fn=analyze_text,
        inputs=input_text,
        outputs=[prediction_output, confidence_output, raw_score_output, cleaned_output, error_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "", "", ""),
        inputs=None,
        outputs=[input_text, prediction_output, confidence_output, raw_score_output, cleaned_output]
    )


# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
