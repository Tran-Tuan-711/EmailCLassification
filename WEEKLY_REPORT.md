# 📋 BÁO CÁO TIẾN ĐỘ HÀNG TUẦN — Email Classification using CNN

> **Đề tài:** Phân loại Email Spam/Normal sử dụng Deep Learning (CNN) kết hợp Rule Engine
> **Thời gian thực hiện:** 6 tuần (14/04/2026 – 25/05/2026)
> **Công nghệ:** Python 3.11, TensorFlow/Keras, scikit-learn, NLTK, Tkinter

## 👥 Thành viên nhóm

| STT | Họ và Tên         | MSSV       | Vai trò                          |
|-----|-------------------|------------|----------------------------------|
| 1   | Thành viên 1      | XXXXXXXX   | Trưởng nhóm – Huấn luyện Model  |
| 2   | Thành viên 2      | XXXXXXXX   | Tiền xử lý dữ liệu & Dataset   |
| 3   | Thành viên 3      | XXXXXXXX   | Rule Engine & Phát hiện Phishing |
| 4   | Thành viên 4      | XXXXXXXX   | Giao diện GUI & Tích hợp hệ thống|

---

## 📅 TUẦN 1 (14/04 – 20/04/2026)

### 🎯 Mục tiêu: Khảo sát đề tài, thu thập dữ liệu, thiết kế kiến trúc

#### Thành viên 1 — Trưởng nhóm / Model
- Nghiên cứu các phương pháp phân loại email: Naive Bayes, SVM, CNN, Logistic Regression
- Quyết định chọn **CNN (Convolutional Neural Network)** làm mô hình chính
- Thiết kế kiến trúc tổng thể dự án (phân chia module)
- Tạo repository GitHub, thiết lập cấu trúc thư mục:
  ```
  EmailClassification/
  ├── model/       # Huấn luyện & dự đoán
  ├── utils/       # Tiền xử lý & tiện ích
  ├── GUI/         # Giao diện
  ├── email_reader/# Đọc email IMAP
  ├── rules/       # Rule Engine phát hiện spam
  ├── data/        # Dataset
  └── logs/        # Log & biểu đồ
  ```

#### Thành viên 2 — Dữ liệu
- Tìm kiếm và thu thập dataset **SpamAssassin** (~24MB, raw email format)
- Phân tích cấu trúc dataset: cột `text`, `target`, raw email headers
- Nghiên cứu các phương pháp tiền xử lý văn bản NLP (tokenization, stemming, stopwords)
- Tạo file `requirements.txt` với các thư viện cần thiết

#### Thành viên 3 — Rule Engine
- Nghiên cứu các mẫu phishing email phổ biến tại Việt Nam
- Thu thập danh sách từ khóa spam tiếng Việt theo nhóm (ngân hàng, tiền bạc, đầu tư)
- Phân loại các nhóm lừa đảo: giả mạo ngân hàng, khuyến mãi, việc làm giả
- Thu thập danh sách domain đáng tin cậy (whitelist)

#### Thành viên 4 — GUI
- Nghiên cứu Tkinter cho giao diện desktop Python
- Nghiên cứu giao thức IMAP để đọc email từ Gmail
- Thiết kế mockup giao diện người dùng
- Tìm hiểu cách tạo App Password Gmail cho IMAP

### 📊 Kết quả Tuần 1
- ✅ Hoàn thành thiết kế kiến trúc dự án
- ✅ Thu thập dataset SpamAssassin (24MB)
- ✅ Xác định công nghệ và phân chia công việc

---

## 📅 TUẦN 2 (21/04 – 27/04/2026)

### 🎯 Mục tiêu: Xây dựng pipeline tiền xử lý, huấn luyện model CNN v1

#### Thành viên 1 — Model CNN
- Xây dựng kiến trúc CNN cho phân loại email:
  ```
  Embedding(10000, 128) → Conv1D(128, 5, relu) → GlobalMaxPooling1D
  → Dense(64, relu) → Dropout(0.5) → Dense(1, sigmoid)
  ```
- Cấu hình training: `epochs=10`, `batch_size=32`, `validation_split=0.2`
- Sử dụng `EarlyStopping(patience=3)` và `TensorBoard` logging
- **Kết quả huấn luyện lần 1:**

| Metric      | Normal | Spam   | Overall |
|-------------|--------|--------|---------|
| Precision   | 0.99   | 0.99   | —       |
| Recall      | 0.99   | 0.98   | —       |
| F1-Score    | 0.99   | 0.98   | —       |
| **Accuracy**| —      | —      | **0.99**|

- **Confusion Matrix (lần 1):**

|                | Predicted Normal | Predicted Spam |
|----------------|-----------------|----------------|
| Actual Normal  | 737             | 4              |
| Actual Spam    | 7               | 322            |

- Tổng: **11 lỗi / 1070 mẫu** → Tỷ lệ lỗi: **1.03%**

#### Thành viên 2 — Tiền xử lý
- Viết module `utils/preprocess.py`:
  - `clean_text_en()`: Lowercase, xóa HTML/URL/email/số, bỏ stopwords EN, stemming (Porter)
  - `clean_text_vi()`: Giữ dấu Unicode tiếng Việt, bỏ stopwords VN, không stem
  - `_is_vietnamese()`: Tự động phát hiện ngôn ngữ (>= 2 ký tự dấu VN)
  - `clean_text()`: Auto-detect ngôn ngữ → chọn pipeline phù hợp
- Viết module `utils/clean_dataset.py`:
  - `load_and_clean()`: Hỗ trợ nhiều format dataset (SpamAssassin, Enron, CSV cơ bản)
  - `_extract_body_from_raw()`: Trích xuất body từ raw email
  - `_extract_subject_from_raw()`: Trích xuất subject từ header
  - Cân bằng dataset: sample normal ≤ 3× spam

- **Thống kê dataset sau clean:**

| Metric                | Giá trị |
|-----------------------|---------|
| Tổng mẫu             | 5,348   |
| Normal (label=0)      | 3,648   |
| Spam (label=1)        | 1,700   |
| Train set             | 4,278   |
| Test set              | 1,070   |
| MAX_WORDS (Tokenizer) | 10,000  |
| MAX_LEN (Padding)     | 200     |

#### Thành viên 3 — Rule Engine
- Xây dựng bộ từ khóa spam 6 nhóm trong `rules/vietnam_spam_rules.py`:

| Nhóm                   | Số từ khóa | Trọng số |
|------------------------|-----------|----------|
| Giả mạo ngân hàng     | 33        | 2.0      |
| Tiền bạc / Khuyến mãi | 26        | 1.5      |
| Giả mạo dịch vụ       | 15        | 1.8      |
| Đầu tư / Lừa đảo TC   | 22        | 1.7      |
| Việc làm giả           | 11        | 1.3      |
| Thông báo giả mạo      | 10        | 1.2      |

- Thu thập 50+ domain đáng tin cậy (Google, Microsoft, ngân hàng VN, e-commerce)
- Định nghĩa 16 pattern domain đáng ngờ (TLD rẻ: .xyz, .top, .buzz, ...)
- Định nghĩa 12 pattern sender đáng ngờ

#### Thành viên 4 — GUI & Email Reader
- Nghiên cứu module `imaplib` để đọc email Gmail
- Thiết kế cấu trúc file `email_reader/imap_reader.py`
- Bắt đầu thiết kế layout Tkinter cho GUI

### 📊 Kết quả Tuần 2
- ✅ Model CNN v1 đạt accuracy **99%**
- ✅ Pipeline tiền xử lý song ngữ EN/VN hoàn chỉnh
- ✅ Bộ từ khóa spam tiếng Việt 117 keywords, 6 nhóm

---

## 📅 TUẦN 3 (28/04 – 04/05/2026)

### 🎯 Mục tiêu: Xây dựng Rule Engine, bắt đầu GUI

#### Thành viên 1 — Tối ưu Model
- Huấn luyện lại model CNN (lần 2):

| Metric      | Normal | Spam   | Overall |
|-------------|--------|--------|---------|
| Precision   | 0.99   | 0.98   | —       |
| Recall      | 0.99   | 0.99   | —       |
| F1-Score    | 0.99   | 0.98   | —       |
| **Accuracy**| —      | —      | **0.99**|

- **Confusion Matrix (lần 2):**

|                | Predicted Normal | Predicted Spam |
|----------------|-----------------|----------------|
| Actual Normal  | 735             | 6              |
| Actual Spam    | 4               | 325            |

- Tổng lỗi: **10 / 1070** → Cải thiện 1 mẫu so với lần 1
- Lưu model `cnn_model.h5` và `tokenizer.pkl`
- Export biểu đồ training (Loss & Accuracy) vào `logs/charts/`

#### Thành viên 2 — Dataset & Tiền xử lý
- Viết `utils/save_clean_data.py` — script tiền xử lý SpamAssassin dataset
- Kiểm tra chất lượng `spam_clean.csv`:
  - Loại bỏ duplicates, NaN, text quá ngắn
  - Verify label distribution cân bằng hợp lý
- Viết hàm `_looks_like_raw_email()` phát hiện email có header

#### Thành viên 3 — Rule Engine Core
- Xây dựng `rules/rule_engine.py` — class `PhishingRuleEngine`:
  - **Step 1:** Trusted sender patterns → Normal (email dịch vụ chính thức)
  - **Step 2:** Thu thập spam signals (domain, keywords, sender)
  - **Step 3:** Nếu `spam_score >= 1.5` → Spam (kể cả trusted domain)
  - **Step 4:** Trusted domain + score = 0 → Normal
  - **Step 5:** Nếu không đủ → fallback sang CNN model
- Implement singleton pattern `get_engine()` cho hiệu suất
- **Logic đặc biệt:** Trusted domain không return Normal ngay — phải check nội dung trước

#### Thành viên 4 — GUI Tkinter
- Thiết kế giao diện chính với Tkinter dark theme
- Tạo form nhập email (subject + body)
- Tạo nút phân loại và hiển thị kết quả
- Thiết kế khung hiển thị chi tiết (method, confidence, matched rules)

### 📊 Kết quả Tuần 3
- ✅ Model CNN v2 ổn định, accuracy 99%
- ✅ Dataset SpamAssassin đã clean và validate
- ✅ Rule Engine hoàn chỉnh với 5-step pipeline
- ✅ GUI prototype cơ bản

---

## 📅 TUẦN 4 (05/05 – 11/05/2026)

### 🎯 Mục tiêu: Tích hợp hệ thống, test end-to-end

#### Thành viên 1 — Predict Module
- Viết `model/predict_cnn.py`:
  - Lazy loading model (chỉ load khi gọi predict lần đầu)
  - Hàm `predict_email()` kết hợp **Rule Engine + CNN**:
    - Nếu `use_rules=True` → Rule check trước
    - Nếu Rule không quyết định → fallback CNN
  - Return dict: `label`, `confidence`, `display`, `method`, `matched_rules`, `spam_score`
- Viết `test.py` — 10 test case (spam EN, spam VN, normal VN, mixed, trusted sender):

| Test Case | Nội dung | Expected | Method |
|-----------|----------|----------|--------|
| 1 | Win a FREE iPhone now!!! | Spam | model_cnn |
| 2 | Trúng thưởng 1 chiếc xe SH | Spam | rule_keyword |
| 3 | Tài khoản bị khóa, xác minh ngay | Spam | rule_keyword |
| 4 | Vay tiền nhanh lãi suất thấp | Spam | rule_keyword |
| 5 | Ê tối nay đi ăn không? | Normal | model_cnn |
| 6 | Mai nhớ nộp bài deadline | Normal | model_cnn |
| 7 | Gửi em tài liệu học tập | Normal | model_cnn |
| 8 | Bạn đã nhận tài liệu chưa? | Normal | model_cnn |
| 9 | Click nhận ưu đãi cực lớn!!! | Spam | rule_keyword |
| 10 | New login from Chrome (noreply@github.com) | Normal | rule_whitelist |

#### Thành viên 2 — Validate Dataset
- Chạy pipeline clean: `spam_assassin.csv` → `spam_clean.csv` (5,348 mẫu)
- Validate: kiểm tra label distribution, kiểm tra text quality
- Kiểm tra không còn reference tới dataset VN tự generate

#### Thành viên 3 — Tối ưu Rule Engine
- Xóa `gmail.com`, `outlook.com`, `hotmail.com`, `live.com`, `icloud.com` khỏi TRUSTED_DOMAINS
  - Lý do: ai cũng đăng ký được → spammer cũng dùng
- Sửa logic: trusted domain + nội dung spam → vẫn phân loại Spam
- Thêm pattern trusted sender: `*.edu.vn`, `*.gov.vn`, GitHub, Microsoft
- Thêm pattern suspicious domain: `-verify`, `-secure`, `-confirm`, banking-*
- Test rule engine: trusted domain + spam keywords → Spam ✅

#### Thành viên 4 — Tích hợp GUI
- Kết nối GUI với `predict_email()` function
- Hiển thị kết quả phân loại: label, confidence, method
- Hiển thị chi tiết matched rules nếu dùng rule-based
- Tạo `main.py` entry point → chạy GUI

### 📊 Kết quả Tuần 4
- ✅ Hệ thống tích hợp Rule Engine + CNN hoạt động end-to-end
- ✅ 10/10 test case cho kết quả chính xác
- ✅ Logic whitelist đã tối ưu — không auto-trust email công cộng

---

## 📅 TUẦN 5 (12/05 – 18/05/2026)

### 🎯 Mục tiêu: Thêm model Logistic Regression, so sánh hiệu suất, xây dựng IMAP reader

#### Thành viên 1 — So sánh Models
- Viết `model/compare_models.py` — so sánh CNN vs LR trên cùng test set
- So sánh kết quả 2 model:

| Metric      | CNN    | Logistic Regression |
|-------------|--------|---------------------|
| Accuracy    | 0.9900 | 0.9879              |
| ROC-AUC     | —      | 0.9994              |
| Precision (Normal) | 0.99 | 0.99           |
| Recall (Normal)    | 0.99 | 0.99           |
| Precision (Spam)   | 0.98 | 0.98           |
| Recall (Spam)      | 0.98 | 0.99           |
| F1 (Spam)          | 0.98 | 0.98           |
| Tổng lỗi   | 13/1070| 13/1070             |

- Kết luận: CNN nhỉnh hơn 0.21% accuracy, LR có ROC-AUC rất cao (0.9994)
- Export biểu đồ so sánh vào `logs/charts/model_comparison.png`

#### Thành viên 2 — Huấn luyện Logistic Regression
- Viết script `model/train_lr.py`:
  - TF-IDF Vectorization: `max_features=10000`, unigram + bigram
  - Train/Test split: 80/20, `random_state=42`, stratified
- **Kết quả LR:**

| Metric      | Normal | Spam   |
|-------------|--------|--------|
| Precision   | 0.99   | 0.98   |
| Recall      | 0.99   | 0.99   |
| F1-Score    | 0.99   | 0.98   |

- **Accuracy: 0.9879 | ROC-AUC: 0.9994**
- **Confusion Matrix (LR):**

|                | Predicted Normal | Predicted Spam |
|----------------|-----------------|----------------|
| Actual Normal  | 722             | 8              |
| Actual Spam    | 5               | 335            |

- Lưu: `lr_model.pkl`, `tfidf_vectorizer.pkl`
- Export biểu đồ: `logs/charts/lr_training.png`

#### Thành viên 3 — IMAP Reader
- Xây dựng `email_reader/imap_reader.py`:
  - Auto-detect IMAP server từ email domain (Gmail, Outlook, Yahoo, iCloud...)
  - Hàm `detect_imap_server()` mapping domain → server
  - Class `IMAPEmailReader`: login, fetch, parse email
  - Hỗ trợ MIME multipart, encoding handling, HTML-to-text fallback
- Test đọc email thật từ Gmail qua IMAP ✅

#### Thành viên 4 — Hoàn thiện GUI
- Hoàn thiện giao diện Tkinter dark theme, 2 tab:
  - **Tab 1:** Phân loại thủ công — nhập sender + nội dung → kết quả
  - **Tab 2:** IMAP Reader — nhập email + app password + số email → phân loại batch
- Auto-detect server khi nhập email (hiển thị real-time)
- Treeview hiển thị kết quả batch với màu sắc (xanh=Normal, đỏ=Spam)
- Click email trong treeview → xem chi tiết ở Tab 1
- Threading để không block UI khi đọc IMAP

### 📊 Kết quả Tuần 5
- ✅ Model Logistic Regression đạt accuracy **98.79%**, ROC-AUC **0.9994**
- ✅ So sánh CNN vs LR hoàn tất với biểu đồ
- ✅ IMAP Reader hoạt động với Gmail, Outlook
- ✅ GUI hoàn thiện với 2 tab

---

## 📅 TUẦN 6 (19/05 – 25/05/2026)

### 🎯 Mục tiêu: Hoàn thiện, viết tài liệu, chuẩn bị báo cáo

#### Thành viên 1 — Tổng kết & Documentation
- Viết `README.md` hoàn chỉnh:
  - Kiến trúc hệ thống (Rule Engine tầng 1 → CNN tầng 2)
  - Hướng dẫn cài đặt, sử dụng
  - Bảng so sánh CNN vs LR
  - Hướng dẫn Gmail IMAP + App Password
- Tổng hợp kết quả cuối cùng:

| Model               | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| **CNN**              | **99.0%** | 0.99     | 0.99   | 0.99     |
| Logistic Regression  | 98.8%    | 0.99      | 0.99   | 0.99     |
| Rule Engine          | —        | —         | —      | —        |

- Review và merge tất cả code

#### Thành viên 2 — Cleanup & Validation
- Clean code, thêm docstring cho tất cả functions
- Validate lại dataset cuối cùng: chỉ dùng SpamAssassin, không còn dataset VN tự generate
- Kiểm tra logging (`utils/logger.py`) hoạt động đúng
- Cập nhật `.gitignore`

#### Thành viên 3 — Test Rule Engine
- Chạy test suite với tất cả test cases
- Verify trusted sender patterns hoạt động đúng (noreply@github.com → Normal)
- Verify keyword detection với email phishing VN
- Verify logic mới: gmail.com + nội dung spam → Spam ✅
- Viết documentation cho bộ rules

#### Thành viên 4 — Demo & Slides
- Test GUI end-to-end: nhập email → phân loại → hiển thị kết quả
- Test đọc email thật từ Gmail
- Chuẩn bị slides báo cáo đồ án
- Quay video demo

### 📊 Kết quả Tuần 6
- ✅ Hệ thống hoàn chỉnh, sẵn sàng demo
- ✅ Documentation đầy đủ
- ✅ Tất cả test cases pass

---

## 📈 TỔNG KẾT DỰ ÁN

### Thống kê Model Training

| Thông số               | CNN (Final)   | Logistic Regression |
|------------------------|---------------|---------------------|
| Dataset                | SpamAssassin (5,348 mẫu) | SpamAssassin (5,348 mẫu) |
| Train/Test split       | 80% / 20%    | 80% / 20%          |
| Train size             | 4,278         | 4,278               |
| Test size              | 1,070         | 1,070               |
| Epochs                 | 10 (EarlyStopping) | —              |
| Batch size             | 32            | —                   |
| Feature extraction     | Tokenizer (10K words) | TF-IDF (10K features) |
| Max sequence length    | 200           | —                   |
| **Accuracy**           | **~99.0%**    | **~98.8%**          |
| **ROC-AUC**            | —             | **~0.999**          |
| Tổng lỗi phân loại    | ~13/1070      | ~13/1070            |
| Thời gian train        | ~40 giây      | ~0.2 giây           |

### Kiến trúc hệ thống cuối cùng

```
Email Input
    │
    ▼
┌─────────────────────────────┐
│  TẦNG 1: RULE ENGINE (VN)   │
│  ├─ Trusted sender patterns │──→ Normal (95% confidence)
│  ├─ Suspicious domain       │──→ +3.0 score
│  ├─ Keyword matching        │──→ +score (6 nhóm)
│  ├─ Suspicious sender       │──→ +1.5 score
│  ├─ score ≥ 1.5?            │──→ Spam (kể cả trusted domain)
│  └─ Trusted domain+score=0  │──→ Normal (90% confidence)
└─────────┬───────────────────┘
          │ (nếu không quyết định được)
          ▼
┌─────────────────────────────┐
│  TẦNG 2: CNN MODEL          │
│  (trained on SpamAssassin)  │
│  Embedding → Conv1D         │
│  → GlobalMaxPool            │
│  → Dense → Dropout          │
│  → Sigmoid                  │
└─────────────────────────────┘
          │
          ▼
    Kết quả: Spam / Normal
    + Confidence + Method
```

### Đóng góp từng thành viên

| Thành viên   | Module chính                      | Files          |
|-------------|-----------------------------------|----------------|
| Thành viên 1 | Model CNN, Predict, So sánh, Docs | `train_cnn.py`, `predict_cnn.py`, `compare_models.py`, `README.md` |
| Thành viên 2 | Tiền xử lý, Dataset, LR          | `preprocess.py`, `clean_dataset.py`, `train_lr.py` |
| Thành viên 3 | Rule Engine, Spam Rules, IMAP     | `rule_engine.py`, `vietnam_spam_rules.py`, `imap_reader.py` |
| Thành viên 4 | GUI Tkinter, Tích hợp, Demo      | `app.py`, `main.py`, `test.py` |
