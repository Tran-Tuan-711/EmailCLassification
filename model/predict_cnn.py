import pickle
import inspect
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from utils.preprocess import clean_text
from rules.rule_engine import get_engine

MAX_LEN = 200
MODEL_PATH = "model/cnn_model.h5"
TOKENIZER_PATH = "model/tokenizer.pkl"
LR_MODEL_PATH = "model/lr_model.pkl"
LR_VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

# Ngưỡng tối thiểu để phân loại Spam — nếu xác suất spam < ngưỡng này
# thì coi như model không đủ tự tin → mặc định Normal.
# Giải quyết vấn đề: text rỗng/không nhận dạng được (tiếng Việt, gibberish)
# tạo ra TF-IDF vector gần bằng 0 → bias model thiên nhẹ về spam (~51.4%)
# → phân loại sai thành Spam dù model thực chất "đoán bừa".
SPAM_CONFIDENCE_THRESHOLD = 0.6

# Lazy loading — chỉ load khi cần
_model = None
_tokenizer = None
_lr_model = None
_lr_vectorizer = None

# Xác định các thuộc tính deprecated dựa trên phiên bản sklearn hiện tại
_lr_supported_params = set(inspect.signature(LogisticRegression.__init__).parameters.keys())


def _load_model():
    """Load model và tokenizer nếu chưa load."""
    global _model, _tokenizer
    if _model is None:
        _model = load_model(MODEL_PATH, compile=False)
    if _tokenizer is None:
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)


def _load_lr_model():
    """Load Logistic Regression model và TF-IDF vectorizer nếu chưa load."""
    global _lr_model, _lr_vectorizer
    if _lr_model is None:
        with open(LR_MODEL_PATH, "rb") as f:
            _lr_model = pickle.load(f)
        # Xử lý tương thích phiên bản scikit-learn:
        # sklearn mới xóa 'multi_class' khỏi __init__ nhưng nội bộ predict
        # vẫn truy cập self.multi_class. Đảm bảo thuộc tính luôn tồn tại
        # với giá trị mặc định "auto" để chạy được trên MỌI phiên bản.
        if not hasattr(_lr_model, "multi_class"):
            _lr_model.multi_class = "auto"
    if _lr_vectorizer is None:
        with open(LR_VECTORIZER_PATH, "rb") as f:
            _lr_vectorizer = pickle.load(f)


def predict_email(text, sender_email=None, use_rules=True, model_type="cnn"):
    """
    Phân loại email: Normal hoặc Spam.
    Kết hợp rule-based check (nếu có sender info) + CNN hoặc LR model.

    Args:
        text: Nội dung email (subject + body)
        sender_email: Địa chỉ email người gửi (optional)
        use_rules: Có dùng rule-based check trước không (default: True)
        model_type: Loại mô hình sử dụng ("cnn" hoặc "lr")

    Returns: dict với keys:
        - label: "Spam" / "Normal"
        - confidence: float (0-1)
        - display: str hiển thị
        - method: "rule_whitelist" / "rule_keyword" / "model_cnn" / "model_lr"
        - matched_rules: list (nếu dùng rules)
        - spam_score: float (nếu dùng rules)
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

    # ─── Step 2: Fallback sang Machine Learning Model ───
    clean = clean_text(text)

    if model_type == "lr":
        _load_lr_model()
        
        # Transform và dự đoán với LR
        X_lr = _lr_vectorizer.transform([clean])
        prob = _lr_model.predict_proba(X_lr)[0][1]  # xác suất spam
        
        # Dùng ngưỡng SPAM_CONFIDENCE_THRESHOLD thay vì 0.5
        # Nếu prob nằm trong vùng không chắc chắn (0.5 ~ threshold)
        # → mặc định Normal vì model "đoán bừa"
        if prob >= SPAM_CONFIDENCE_THRESHOLD:
            label = "Spam"
            confidence = prob
            details = "Phân loại bằng mô hình Logistic Regression."
        else:
            label = "Normal"
            confidence = 1 - prob
            if prob > 0.5:
                details = (f"Mô hình LR không đủ tự tin để phân loại Spam "
                           f"(xác suất {prob:.1%} < ngưỡng {SPAM_CONFIDENCE_THRESHOLD:.0%}). "
                           f"Mặc định: Normal.")
            else:
                details = "Phân loại bằng mô hình Logistic Regression."

        return {
            "label": label,
            "confidence": float(confidence),
            "display": f"{label} ({confidence:.1%})",
            "method": "model_lr",
            "matched_rules": [],
            "spam_score": 0.0,
            "details": details,
        }
    else:
        _load_model()

        seq = _tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        prob = _model.predict(padded, verbose=0)[0][0]

        # Dùng ngưỡng SPAM_CONFIDENCE_THRESHOLD thay vì 0.5
        if prob >= SPAM_CONFIDENCE_THRESHOLD:
            label = "Spam"
            confidence = prob
            details = "Phân loại bằng mô hình CNN."
        else:
            label = "Normal"
            confidence = 1 - prob
            if prob > 0.5:
                details = (f"Mô hình CNN không đủ tự tin để phân loại Spam "
                           f"(xác suất {prob:.1%} < ngưỡng {SPAM_CONFIDENCE_THRESHOLD:.0%}). "
                           f"Mặc định: Normal.")
            else:
                details = "Phân loại bằng mô hình CNN."

        return {
            "label": label,
            "confidence": float(confidence),
            "display": f"{label} ({confidence:.1%})",
            "method": "model_cnn",
            "matched_rules": [],
            "spam_score": 0.0,
            "details": details,
        }