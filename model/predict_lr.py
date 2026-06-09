import pickle
import numpy as np
from utils.preprocess import clean_text, _is_vietnamese
from rules.rule_engine import get_engine

MODEL_PATH = "model/lr_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

# Lazy loading — chỉ load khi cần
_model = None
_vectorizer = None


def _load_model():
    """Load model và vectorizer nếu chưa load."""
    global _model, _vectorizer
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    if _vectorizer is None:
        with open(VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)


def predict_email(text, sender_email=None, use_rules=True):
    """
    Phân loại email: Normal hoặc Spam.
    Kết hợp rule-based check (nếu có sender info) + Logistic Regression model.

    Args:
        text: Nội dung email (subject + body)
        sender_email: Địa chỉ email người gửi (optional)
        use_rules: Có dùng rule-based check trước không (default: True)

    Returns: dict với keys:
        - label: "Spam" / "Normal"
        - confidence: float (0-1)
        - display: str hiển thị
        - method: "rule_whitelist" / "rule_keyword" / "model_lr"
        - matched_rules: list (nếu dùng rules)
        - spam_score: float (nếu dùng rules)
        - details: str chi tiết
    """
    # ─── Step 1: Rule-based check ───
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

    # ─── Step 2: Kiểm tra ngôn ngữ ───
    # Model LR được huấn luyện trên tập SpamAssassin (tiếng Anh).
    # Nếu text là tiếng Việt và rule engine không quyết định được,
    # mặc định trả về Normal vì model không đáng tin cậy cho tiếng Việt.
    if _is_vietnamese(text):
        return {
            "label": "Normal",
            "confidence": 0.6,
            "display": "Normal (60.0%)",
            "method": "model_lr",
            "matched_rules": [],
            "spam_score": 0.0,
            "details": "Text tieng Viet — Rule Engine khong phat hien spam, mac dinh Normal.",
        }

    # ─── Step 3: Fallback sang Logistic Regression model ───
    _load_model()

    clean = clean_text(text)

    # TF-IDF Vectorizer
    vect_text = _vectorizer.transform([clean])

    # Predict probability
    prob = _model.predict_proba(vect_text)[0][1]

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
        "method": "model_lr",
        "matched_rules": [],
        "spam_score": 0.0,
        "details": "Phan loai bang Logistic Regression model.",
    }
