import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocess import clean_text, _is_vietnamese
from rules.rule_engine import get_engine

MAX_LEN = 200
MODEL_PATH = "model/fasttext_model.h5"
TOKENIZER_PATH = "model/fasttext_tokenizer.pkl"

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

    # Model fastText được huấn luyện trên tập SpamAssassin (tiếng Anh).
    # Nếu text là tiếng Việt và rule engine không quyết định được,
    # mặc định trả về Normal vì model không đáng tin cậy cho tiếng Việt.
    if _is_vietnamese(text):
        return {
            "label": "Normal",
            "confidence": 0.6,
            "display": "Normal (60.0%)",
            "method": "model_fasttext",
            "matched_rules": [],
            "spam_score": 0.0,
            "details": "Text tieng Viet — Rule Engine khong phat hien spam, mac dinh Normal.",
        }

    _load_model()

    clean = clean_text(text)
    seq = _tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    prob = _model.predict(padded, verbose=0)[0][0]

    THRESHOLD = 0.7  # Ngưỡng cao để đạt precision tốt cho Spam

    if prob > THRESHOLD:
        label = "Spam"
        confidence = prob
    else:
        label = "Normal"
        confidence = 1 - prob

    return {
        "label": label,
        "confidence": float(confidence),
        "display": f"{label} ({confidence:.1%})",
        "method": "model_fasttext",
        "matched_rules": [],
        "spam_score": 0.0,
        "details": "Phan loai bang fastText model.",
    }
