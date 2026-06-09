# 📧 Email Spam Classification — Deep Learning + Rule-based

Hệ thống phân loại email **Spam / Normal** sử dụng kết hợp **mô hình CNN (Deep Learning)** và **bộ luật phát hiện spam/phishing (Rule-based)**, hỗ trợ cả email tiếng Anh và tiếng Việt.

## 🏗 Kiến trúc hệ thống

```
Email Input
    │
    ▼
┌─────────────────────────────────────────┐
│  TẦNG 1: RULE ENGINE (Tiếng Việt)       │
│  ├─ Trusted sender patterns  ──→ Normal │
│  ├─ Suspicious domain check  ──→ +score │
│  ├─ Keyword matching (6 nhóm)──→ +score │
│  ├─ Suspicious sender check  ──→ +score │
│  ├─ score ≥ threshold?       ──→ Spam   │
│  └─ Trusted domain + score=0 ──→ Normal │
└──────────────┬──────────────────────────┘
               │ (nếu không quyết định được)
               ▼
┌─────────────────────────────────────────┐
│  TẦNG 2: AI MODELS (SpamAssassin)       │
│  ├─ CNN Model: Embedding -> Conv1D      │
│  │  -> GlobalMaxPool -> Dense -> Sigmoid│
│  │                                      │
│  └─ fastText Model: Embedding           │
│     -> GlobalAveragePool -> Dense       │
└─────────────────────────────────────────┘
               │
               ▼
         Kết quả: Spam / Normal
         + Confidence + Method
```

## 📁 Cấu trúc dự án

```
EmailClassification/
├── main.py                     # Entry point — khởi chạy GUI
├── test.py                     # Test phân loại với email mẫu
├── requirements.txt            # Danh sách thư viện
├── data/
│   ├── spam_assassin.csv       # Dataset gốc SpamAssassin
│   └── spam_clean.csv          # Dataset đã tiền xử lý
├── model/
│   ├── train_cnn.py            # Huấn luyện model CNN
│   ├── train_fasttext.py       # Huấn luyện model fastText
│   ├── predict_cnn.py          # Phân loại email (Rule Engine + CNN)
│   ├── predict_fasttext.py     # Phân loại email (Rule Engine + fastText)
│   ├── compare_models.py       # So sánh CNN vs fastText
│   ├── cnn_model.h5            # Model CNN đã huấn luyện
│   ├── tokenizer.pkl           # Tokenizer (CNN)
│   ├── fasttext_model.h5       # Model fastText đã huấn luyện
│   └── fasttext_tokenizer.pkl  # Tokenizer (fastText)
├── GUI/
│   └── app.py                  # Giao diện Tkinter (dark theme)
├── email_reader/
│   └── imap_reader.py          # Đọc email từ IMAP (Gmail, Outlook, Yahoo...)
├── rules/
│   ├── rule_engine.py          # Bộ phát hiện spam dựa trên rules
│   └── vietnam_spam_rules.py   # Từ khóa spam tiếng Việt (6 nhóm)
├── utils/
│   ├── preprocess.py           # Tiền xử lý text (EN + VN)
│   ├── clean_dataset.py        # Làm sạch dataset SpamAssassin
│   ├── save_clean_data.py      # Lưu dataset đã clean
│   └── logger.py               # Logging utility
└── logs/
    ├── train.log               # Log huấn luyện CNN
    ├── train_lr.log            # Log huấn luyện LR
    └── charts/                 # Biểu đồ training & so sánh
```

## 🔧 Cài đặt

```bash
# Clone repository
git clone <repo-url>
cd EmailClassification

# Cài đặt thư viện
pip install -r requirements.txt
```

## 🚀 Sử dụng

### 1. Tiền xử lý dataset (nếu chưa có `spam_clean.csv`)
```bash
python -m utils.save_clean_data
```

### 2. Huấn luyện model CNN
```bash
python -m model.train_cnn
```

### 3. Huấn luyện model fastText
```bash
python -m model.train_fasttext
```

### 4. So sánh 2 model
```bash
python -m model.compare_models
```

### 5. Chạy GUI
```bash
python main.py
```

### 6. Test nhanh qua CLI
```bash
python test.py
```

## 📊 Kết quả

### So sánh Models

| Metric      | CNN          | fastText            |
|-------------|--------------|---------------------|
| **Accuracy**| **~99.1%**   | **~99.1%**          |
| ROC-AUC     | ~0.9995      | ~0.9996             |
| Dataset     | SpamAssassin | SpamAssassin        |
| Features    | Tokenizer    | Tokenizer           |

### Rule Engine (Tiếng Việt)

| Nhóm từ khóa               | Số keywords | Trọng số |
|-----------------------------|-------------|----------|
| Giả mạo ngân hàng          | 33          | 2.0      |
| Tiền bạc / Khuyến mãi      | 26          | 1.5      |
| Giả mạo dịch vụ            | 15          | 1.8      |
| Đầu tư / Lừa đảo tài chính | 22          | 1.7      |
| Việc làm giả               | 11          | 1.3      |
| Thông báo giả mạo          | 10          | 1.2      |

## 🔑 Lưu ý sử dụng Gmail IMAP

Để đọc email từ Gmail qua IMAP, bạn cần:

1. Bật **2-Step Verification** trong cài đặt Google Account
2. Tạo **App Password** tại: https://myaccount.google.com/apppasswords
3. Sử dụng App Password (không phải mật khẩu thường) trong GUI

## 🛠 Công nghệ

- **Python 3.11.x**
- **TensorFlow/Keras** — Mô hình CNN, Mô hình fastText
- **scikit-learn** — Metrics & train/test split
- **NLTK** — Tiền xử lý ngôn ngữ tự nhiên
- **Tkinter** — Giao diện người dùng
- **imaplib** — Đọc email qua IMAP
- **matplotlib** — Biểu đồ training & so sánh
