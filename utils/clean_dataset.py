import pandas as pd
import re
from utils.preprocess import clean_text


def _extract_body_from_raw(raw_message):
    """
    Trích xuất body từ raw email message (bỏ header).
    Email header kết thúc sau dòng trống đầu tiên.
    """
    if not isinstance(raw_message, str):
        return ""

    # Email header ends at first blank line
    parts = raw_message.split("\n\n", 1)
    if len(parts) > 1:
        body = parts[1]
    else:
        body = raw_message

    # Remove forwarded/replied headers inside body
    body = re.sub(r'-{3,}\s*Original Message\s*-{3,}.*', '', body, flags=re.DOTALL)
    body = re.sub(r'-{3,}\s*Forwarded.*?-{3,}', '', body, flags=re.DOTALL)

    return body.strip()


def _extract_subject_from_raw(raw_message):
    """Trích xuất Subject từ raw email header."""
    if not isinstance(raw_message, str):
        return ""
    match = re.search(r'^Subject:\s*(.*)$', raw_message, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _looks_like_raw_email(text):
    """
    Kiểm tra text có phải raw email (chứa header) không.
    Dựa trên sự xuất hiện của các header phổ biến.
    """
    if not isinstance(text, str) or len(text) < 50:
        return False

    headers = ['From:', 'Return-Path:', 'Received:', 'Date:', 'Subject:', 'To:',
               'Content-Type:', 'MIME-Version:', 'Delivered-To:']
    header_count = sum(1 for h in headers if h in text[:500])
    return header_count >= 2


def load_and_clean(path, encoding='latin-1'):
    """
    Load dataset CSV và thực hiện cleaning.
    Được tinh giản để tập trung xử lý tập dữ liệu SpamAssassin.
    """
    df = pd.read_csv(path, encoding=encoding)
    print(f"Raw dataset: {len(df)} rows, columns: {list(df.columns)}")

    # Tự động nhận diện cột text và target/label
    text_candidates = ['text', 'message', 'email', 'content', 'body', 'sms']
    text_col = None
    for col in text_candidates:
        matches = [c for c in df.columns if col in c.lower()]
        if matches:
            text_col = matches[0]
            break

    label_candidates = ['target', 'label', 'class', 'spam', 'category', 'v1']
    label_col = None
    for col in label_candidates:
        matches = [c for c in df.columns if col in c.lower()]
        if matches:
            label_col = matches[0]
            break

    if text_col is None or label_col is None:
        print(f"Available columns: {list(df.columns)}")
        raise Exception(
            f"Không tìm thấy cột text (tìm: {text_candidates}) "
            f"hoặc cột label (tìm: {label_candidates})!"
        )

    print(f"Detected columns — text: '{text_col}', label: '{label_col}'")

    df = df[[text_col, label_col]]
    df.columns = ['text', 'label']

    # Chuyển label text sang số nếu cần
    if df['label'].dtype == 'object':
        label_map = {'ham': 0, 'spam': 1, 'normal': 0}
        df['label'] = df['label'].str.lower().map(label_map)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

    # Kiểm tra nếu text chứa raw email (có header) — trích xuất body
    sample_text = str(df['text'].iloc[0]) if len(df) > 0 else ""
    if _looks_like_raw_email(sample_text):
        print("Detected: Raw email format (text contains headers) -> extracting body...")
        df['subject'] = df['text'].apply(_extract_subject_from_raw)
        df['body'] = df['text'].apply(_extract_body_from_raw)
        df['text'] = df['subject'] + " " + df['body']

    # Cleaning chung
    df = df[['text', 'label']]
    df = df.dropna()
    df = df.drop_duplicates(subset=['text'])

    print("Applying text preprocessing...")
    df['clean_text'] = df['text'].apply(clean_text)

    # Loại bỏ text quá ngắn sau khi clean
    df = df[df['clean_text'].str.len() > 10]

    print(f"Dataset size after clean (before balancing): {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    # Cân bằng dataset: downsample lớp đa số (ham) để khớp với lớp thiểu số (spam)
    df_normal = df[df['label'] == 0]
    df_spam = df[df['label'] == 1]

    if len(df_normal) > len(df_spam):
        df_normal = df_normal.sample(n=len(df_spam), random_state=42)
        print(f"\nBalancing: Downsampled Normal tu {len(df[df['label'] == 0])} -> {len(df_normal)} "
              f"de khop voi Spam ({len(df_spam)})")
    elif len(df_spam) > len(df_normal):
        df_spam = df_spam.sample(n=len(df_normal), random_state=42)
        print(f"\nBalancing: Downsampled Spam tu {len(df[df['label'] == 1])} -> {len(df_spam)} "
              f"de khop voi Normal ({len(df_normal)})")

    df = pd.concat([df_normal, df_spam], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    print(f"\nDataset size after balancing: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df