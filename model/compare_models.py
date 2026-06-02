"""
So sánh hiệu suất giữa 2 model: CNN vs Logistic Regression.

Sử dụng:
  python -m model.compare_models

Output:
  - Bảng so sánh accuracy, precision, recall, F1, ROC-AUC
  - Biểu đồ so sánh lưu vào logs/charts/
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.logger import setup_logger

logger = setup_logger(log_file="logs/compare.log")

# Paths
DATA_PATH = "data/spam_clean.csv"
CNN_MODEL_PATH = "model/cnn_model.h5"
CNN_TOKENIZER_PATH = "model/tokenizer.pkl"
LR_MODEL_PATH = "model/lr_model.pkl"
LR_VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

MAX_LEN = 200


def compare():
    """So sánh CNN vs Logistic Regression trên cùng test set."""

    logger.info("=" * 60)
    logger.info("SO SÁNH CNN vs LOGISTIC REGRESSION")
    logger.info("=" * 60)

    # =====================
    # 1. Load dataset
    # =====================
    logger.info(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['clean_text', 'label'])
    df['clean_text'] = df['clean_text'].astype(str)

    texts = df['clean_text']
    labels = df['label'].astype(int)

    logger.info(f"Dataset size: {len(df)}")
    logger.info(f"Label distribution:\n{labels.value_counts().to_string()}")

    # Split — CÙNG random_state để đảm bảo cùng test set
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info(f"Train: {len(texts_train)}, Test: {len(texts_test)}")

    # =====================
    # 2. CNN Prediction
    # =====================
    logger.info("\n--- CNN Model ---")

    cnn_model = load_model(CNN_MODEL_PATH, compile=False)
    with open(CNN_TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    seq_test = tokenizer.texts_to_sequences(texts_test)
    X_test_cnn = pad_sequences(seq_test, maxlen=MAX_LEN)

    cnn_prob = cnn_model.predict(X_test_cnn, verbose=0).flatten()
    cnn_pred = (cnn_prob > 0.5).astype(int)

    cnn_acc = accuracy_score(y_test, cnn_pred)
    try:
        cnn_auc = roc_auc_score(y_test, cnn_prob)
    except ValueError:
        cnn_auc = 0.0

    cnn_report = classification_report(y_test, cnn_pred,
                                        target_names=['Normal', 'Spam'],
                                        output_dict=True)
    cnn_cm = confusion_matrix(y_test, cnn_pred)

    logger.info(f"CNN Accuracy: {cnn_acc:.4f}")
    logger.info(f"CNN ROC-AUC:  {cnn_auc:.4f}")
    logger.info(f"CNN Confusion Matrix:\n{cnn_cm}")

    # =====================
    # 3. Logistic Regression Prediction
    # =====================
    logger.info("\n--- Logistic Regression ---")

    with open(LR_MODEL_PATH, "rb") as f:
        lr_model = pickle.load(f)
    with open(LR_VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    X_test_lr = vectorizer.transform(texts_test)

    lr_pred = lr_model.predict(X_test_lr)
    lr_prob = lr_model.predict_proba(X_test_lr)[:, 1]

    lr_acc = accuracy_score(y_test, lr_pred)
    try:
        lr_auc = roc_auc_score(y_test, lr_prob)
    except ValueError:
        lr_auc = 0.0

    lr_report = classification_report(y_test, lr_pred,
                                       target_names=['Normal', 'Spam'],
                                       output_dict=True)
    lr_cm = confusion_matrix(y_test, lr_pred)

    logger.info(f"LR Accuracy: {lr_acc:.4f}")
    logger.info(f"LR ROC-AUC:  {lr_auc:.4f}")
    logger.info(f"LR Confusion Matrix:\n{lr_cm}")

    # =====================
    # 4. Bảng so sánh
    # =====================
    logger.info("\n" + "=" * 60)
    logger.info("BẢNG SO SÁNH")
    logger.info("=" * 60)

    header = f"{'Metric':<25} {'CNN':>12} {'LR':>12}"
    logger.info(header)
    logger.info("-" * 50)
    logger.info(f"{'Accuracy':<25} {cnn_acc:>12.4f} {lr_acc:>12.4f}")
    logger.info(f"{'ROC-AUC':<25} {cnn_auc:>12.4f} {lr_auc:>12.4f}")
    logger.info(f"{'Precision (Normal)':<25} {cnn_report['Normal']['precision']:>12.4f} {lr_report['Normal']['precision']:>12.4f}")
    logger.info(f"{'Recall (Normal)':<25} {cnn_report['Normal']['recall']:>12.4f} {lr_report['Normal']['recall']:>12.4f}")
    logger.info(f"{'F1 (Normal)':<25} {cnn_report['Normal']['f1-score']:>12.4f} {lr_report['Normal']['f1-score']:>12.4f}")
    logger.info(f"{'Precision (Spam)':<25} {cnn_report['Spam']['precision']:>12.4f} {lr_report['Spam']['precision']:>12.4f}")
    logger.info(f"{'Recall (Spam)':<25} {cnn_report['Spam']['recall']:>12.4f} {lr_report['Spam']['recall']:>12.4f}")
    logger.info(f"{'F1 (Spam)':<25} {cnn_report['Spam']['f1-score']:>12.4f} {lr_report['Spam']['f1-score']:>12.4f}")

    fp_cnn = cnn_cm[0][1]
    fn_cnn = cnn_cm[1][0]
    fp_lr = lr_cm[0][1]
    fn_lr = lr_cm[1][0]
    total_err_cnn = fp_cnn + fn_cnn
    total_err_lr = fp_lr + fn_lr

    logger.info(f"{'False Positive':<25} {fp_cnn:>12d} {fp_lr:>12d}")
    logger.info(f"{'False Negative':<25} {fn_cnn:>12d} {fn_lr:>12d}")
    logger.info(f"{'Total Errors':<25} {total_err_cnn:>12d} {total_err_lr:>12d}")

    # =====================
    # 5. Biểu đồ so sánh
    # =====================
    os.makedirs("logs/charts", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Chart 1: Accuracy & AUC bar chart
    ax1 = axes[0]
    metrics = ['Accuracy', 'ROC-AUC']
    cnn_vals = [cnn_acc, cnn_auc]
    lr_vals = [lr_acc, lr_auc]
    x = np.arange(len(metrics))
    width = 0.3

    bars1 = ax1.bar(x - width/2, cnn_vals, width, label='CNN',
                    color='#00d4aa', alpha=0.85)
    bars2 = ax1.bar(x + width/2, lr_vals, width, label='LR',
                    color='#ffa502', alpha=0.85)

    ax1.set_ylim(0.95, 1.005)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_title("Accuracy & ROC-AUC", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.bar_label(bars1, fmt='%.4f', fontsize=8)
    ax1.bar_label(bars2, fmt='%.4f', fontsize=8)

    # Chart 2: CNN Confusion Matrix
    ax2 = axes[1]
    im2 = ax2.imshow(cnn_cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax2.set_title("Confusion Matrix — CNN", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Normal', 'Spam'])
    ax2.set_yticklabels(['Normal', 'Spam'])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cnn_cm[i, j]), ha="center", va="center",
                     color="white" if cnn_cm[i, j] > cnn_cm.max() / 2 else "black",
                     fontsize=14, fontweight='bold')
    fig.colorbar(im2, ax=ax2)

    # Chart 3: LR Confusion Matrix
    ax3 = axes[2]
    im3 = ax3.imshow(lr_cm, interpolation='nearest', cmap=plt.cm.Oranges)
    ax3.set_title("Confusion Matrix — Logistic Regression", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Normal', 'Spam'])
    ax3.set_yticklabels(['Normal', 'Spam'])
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(lr_cm[i, j]), ha="center", va="center",
                     color="white" if lr_cm[i, j] > lr_cm.max() / 2 else "black",
                     fontsize=14, fontweight='bold')
    fig.colorbar(im3, ax=ax3)

    plt.tight_layout()
    chart_path = "logs/charts/model_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info(f"\nBiểu đồ so sánh đã lưu: {chart_path}")

    # =====================
    # 6. Kết luận
    # =====================
    winner = "CNN" if cnn_acc > lr_acc else "Logistic Regression" if lr_acc > cnn_acc else "Ngang nhau"
    logger.info(f"\n{'=' * 60}")
    logger.info(f"KẾT LUẬN: {winner} có accuracy cao hơn.")
    if lr_auc > cnn_auc:
        logger.info(f"Tuy nhiên, LR có ROC-AUC ({lr_auc:.4f}) cao hơn CNN ({cnn_auc:.4f}).")
    logger.info(f"{'=' * 60}")

    return {
        "cnn": {"accuracy": cnn_acc, "roc_auc": cnn_auc, "report": cnn_report, "cm": cnn_cm},
        "lr": {"accuracy": lr_acc, "roc_auc": lr_auc, "report": lr_report, "cm": lr_cm},
    }


if __name__ == "__main__":
    compare()
