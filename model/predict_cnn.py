import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocess import clean_text
from rules.rule_engine import get_engine

MAX_LEN = 200
MODEL_PATH = "model/cnn_model.h5"
TOKENIZER_PATH = "model/tokenizer.pkl"

# Lazy loading — chỉ load khi cần
_model = None
_tokenizer = None


def _load_model():
    """Load model và tokenizer nếu chưa load."""
    global _model, _tokenizer
    if _model is None:
        _model = load_model(MODEL_PATH, compile=False)
    if _tokenizer is None:
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)


def predict_email(text, sender_email=None, use_rules=True):
    if use_rules:
        engine = get_engine()

        # Tách subject từ text nếu có
        parts = text.split("\n", 1)
        subject = parts[0] if len(parts) > 1 else ""
        body = parts[1] if len(parts) > 1 else text

        rule_result = engine.classify(
            subject=subject,
            body=body,
            sender_email=sender_email or "",
        )

        # Nếu rule đã quyết định → trả về luôn
        if rule_result["label"] is not None:
            return {
                "label": rule_result["label"],
                "confidence": rule_result["confidence"],
                "display": f"{rule_result['label']} ({rule_result['confidence']:.1%})",
                "method": rule_result["method"],
                "matched_rules": rule_result["matched_rules"],
                "spam_score": rule_result["spam_score"],
                "details": rule_result["details"],
            }

    _load_model()

    clean = clean_text(text)
    seq = _tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    prob = _model.predict(padded, verbose=0)[0][0]

    if prob > 0.5:
        label = "Spam"
        confidence = prob
    else:
        label = "Normal"
        confidence = 1 - prob

    return {
        "label": label,
        "confidence": float(confidence),
        "display": f"{label} ({confidence:.1%})",
        "method": "model_cnn",
        "matched_rules": [],
        "spam_score": 0.0,
        "details": "Phân loại bằng CNN model.",
    }