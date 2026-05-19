"""
Train Logistic Regression model cho phân loại email Spam/Normal.

Pipeline:
  1. Load dataset merged (spam_clean_merged.csv)
  2. TF-IDF Vectorization
  3. Train Logistic Regression
  4. Đánh giá trên test set
  5. Lưu model + vectorizer
  6. Lưu biểu đồ

Sử dụng:
  python -m model.train_lr
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)

from utils.logger import setup_logger

logger = setup_logger(log_file="logs/train_lr.log")

# Paths
DATA_PATH = "data/spam_clean_merged.csv"
MODEL_PATH = "model/lr_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"


def train():
    """Train Logistic Regression model."""

    # =====================
    # 1. Load dataset
    # =====================
    logger.info("=" * 60)
    logger.info("LOGISTIC REGRESSION — TRAINING")
    logger.info("=" * 60)

    logger.info(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Đảm bảo có cột clean_text
    if 'clean_text' not in df.columns:
        logger.error("Dataset thiếu cột 'clean_text'! Hãy chạy prepare.py trước.")
        return

    # Loại bỏ NaN
    df = df.dropna(subset=['clean_text', 'label'])
    df['clean_text'] = df['clean_text'].astype(str)

    texts = df['clean_text']
    labels = df['label'].astype(int)

    logger.info(f"Dataset size: {len(df)}")
    logger.info(f"Label distribution:\n{labels.value_counts().to_string()}")

    # =====================
    # 2. TF-IDF Vectorization
    # =====================
    logger.info("Applying TF-IDF Vectorization (max_features=10000)...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),     # Unigram + Bigram
        min_df=2,               # Bỏ từ xuất hiện < 2 lần
        max_df=0.95,            # Bỏ từ xuất hiện > 95% documents
        sublinear_tf=True,      # Logarithmic TF scaling
    )

    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    logger.info(f"TF-IDF matrix shape: {X.shape}")

    # =====================
    # 3. Train/Test Split
    # =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # =====================
    # 4. Train Logistic Regression
    # =====================
    logger.info("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,                  # Regularization strength
        solver='lbfgs',         # Tốt cho dataset vừa
        class_weight='balanced', # Cân bằng class tự động
        random_state=42,
        verbose=1,
    )

    model.fit(X_train, y_train)
    logger.info("Training completed!")

    # =====================
    # 5. Evaluation
    # =====================
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    logger.info("\n" + "=" * 40)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 40)
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Spam'])
    logger.info("\n" + report)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC-AUC:  {roc_auc:.4f}")

    logger.info("\nCONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n{cm}")

    # =====================
    # 6. Save model & vectorizer
    # =====================
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {MODEL_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Vectorizer saved to {VECTORIZER_PATH}")

    # =====================
    # 7. Save charts
    # =====================
    os.makedirs("logs/charts", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix heatmap
    ax1 = axes[0]
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title("Confusion Matrix — Logistic Regression", fontsize=12)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Normal', 'Spam'])
    ax1.set_yticklabels(['Normal', 'Spam'])
    # Annotate cells
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14)
    fig.colorbar(im, ax=ax1)

    # Top 20 important features
    ax2 = axes[1]
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    top_positive = np.argsort(coef)[-10:]  # Top spam indicators
    top_negative = np.argsort(coef)[:10]    # Top normal indicators
    top_indices = np.concatenate([top_negative, top_positive])
    top_features = [feature_names[i] for i in top_indices]
    top_coefs = [coef[i] for i in top_indices]

    colors = ['#22c55e' if c < 0 else '#ef4444' for c in top_coefs]
    ax2.barh(range(len(top_features)), top_coefs, color=colors)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features, fontsize=8)
    ax2.set_title("Top 20 Important Features (Green=Normal, Red=Spam)", fontsize=11)
    ax2.set_xlabel("Coefficient Weight")

    plt.tight_layout()
    chart_path = "logs/charts/lr_training.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info(f"Charts saved to {chart_path}")

    logger.info("\n" + "=" * 60)
    logger.info("LOGISTIC REGRESSION TRAINING COMPLETE!")
    logger.info("=" * 60)

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "report": report,
    }


if __name__ == "__main__":
    train()
