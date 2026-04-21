# 📧 Email Classification using CNN — Deep Learning

Đồ án phân loại email **Normal / Spam** sử dụng mô hình **Convolutional Neural Network (CNN)**.

## 📁 Cấu trúc dự án

```
EmailClassification/
├── main.py                     # Entry point — khởi chạy GUI
├── test_predict.py             # Test phân loại với email mẫu
├── requirements.txt            # Danh sách thư viện
├── data/
│   ├── spam.csv                # Dataset gốc
│   └── spam_clean.csv          # Dataset đã tiền xử lý
├── model/
│   ├── train_cnn.py            # Script huấn luyện model CNN
│   ├── predict_cnn.py          # Hàm phân loại email
│   ├── cnn_model.h5            # Model đã huấn luyện
│   └── tokenizer.pkl           # Tokenizer đã huấn luyện
├── GUI/
│   └── app.py                  # Giao diện Tkinter
├── email_reader/
│   └── imap_reader.py          # Đọc email từ Gmail qua IMAP
├── utils/
│   ├── preprocess.py           # Tiền xử lý text
│   ├── clean_dataset.py        # Làm sạch dataset
│   ├── save_clean_data.py      # Lưu dataset đã clean
│   └── logger.py               # Logging utility
└── logs/
    ├── train.log               # Log huấn luyện
    ├── fit/                    # TensorBoard logs
    └── charts/                 # Biểu đồ training
```

## 🔧 Cài đặt

```bash
# Cài đặt thư viện
pip install -r requirements.txt
```

## 🚀 Sử dụng

### 1. Tiền xử lý dataset (nếu chưa có `spam_clean.csv`)
```bash
python -m utils.save_clean_data
```

### 2. Huấn luyện model
```bash
python -m model.train_cnn
```

### 3. Chạy GUI
```bash
python main.py
```

### 4. Test nhanh qua CLI
```bash
python test_predict.py
```

## 📊 Kết quả

- **Model**: CNN (Embedding → Conv1D → GlobalMaxPooling1D → Dense → Dropout → Sigmoid)
- **Dataset**: ~5,300 email samples
- **Accuracy**: ~99%
- **Phân loại**: Normal (Ham) / Spam

## 🔑 Lưu ý sử dụng Gmail IMAP

Để đọc email từ Gmail, bạn cần:

1. Bật **2-Step Verification** trong cài đặt Google Account
2. Tạo **App Password** tại: https://myaccount.google.com/apppasswords
3. Sử dụng App Password (không phải mật khẩu thường) trong GUI

## 🛠 Công nghệ

- **Python 3.11.x**
- **TensorFlow/Keras** — Mô hình CNN
- **NLTK** — Tiền xử lý ngôn ngữ tự nhiên
- **scikit-learn** — Đánh giá model
- **Tkinter** — Giao diện người dùng
- **imaplib** — Đọc email qua IMAP
