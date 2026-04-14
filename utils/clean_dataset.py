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


def load_and_clean(path, encoding='latin-1'):
    """
    Load dataset CSV và thực hiện cleaning.
    Hỗ trợ 3 format:
      - Format 1 (cũ): cột 'text'/'message' + 'target'/'label' (đã có nhãn)
      - Format 2 (Enron): cột 'file' + 'message' (gán nhãn từ folder path)
      - Format 3 (SpamAssassin): cột 'text' + 'target' (raw email có header)
    """
    df = pd.read_csv(path, encoding=encoding)
    print(f"Raw dataset: {len(df)} rows, columns: {list(df.columns)}")

    # =====================
    # Detect format
    # =====================
    if 'file' in df.columns and 'message' in df.columns:
        # === FORMAT ENRON ===
        print("Detected: Enron email format (file + message)")

        # Gán nhãn dựa trên folder path
        file_lower = df['file'].str.lower()
        df['label'] = 0  # Default: Normal

        # Các folder chứa spam/junk
        spam_mask = file_lower.str.contains('junk|spam|bulk', na=False)
        df.loc[spam_mask, 'label'] = 1  # Spam

        spam_count = df['label'].sum()
        normal_count = len(df) - spam_count
        print(f"Label from folder path — Spam: {spam_count}, Normal: {normal_count}")

        # Trích xuất subject + body
        df['subject'] = df['message'].apply(_extract_subject_from_raw)
        df['body'] = df['message'].apply(_extract_body_from_raw)
        df['text'] = df['subject'] + " " + df['body']

        # Cân bằng dataset: sample normal xuống (lấy tối đa 3x spam)
        df_spam = df[df['label'] == 1]
        df_normal = df[df['label'] == 0]

        sample_size = min(len(df_normal), len(df_spam) * 3)
        df_normal_sampled = df_normal.sample(n=sample_size, random_state=42)

        df = pd.concat([df_normal_sampled, df_spam], ignore_index=True)
        print(f"After balancing — Normal: {len(df_normal_sampled)}, Spam: {len(df_spam)}")

    else:
        # === FORMAT CŨ (text + label/target) hoặc SpamAssassin ===
        # Auto-detect cột text
        text_candidates = ['text', 'message', 'email', 'content', 'body', 'sms']
        text_col = None
        for col in text_candidates:
            matches = [c for c in df.columns if col in c.lower()]
            if matches:
                text_col = matches[0]
                break

        # Auto-detect cột label
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

    # =====================
    # Cleaning chung
    # =====================
    df = df[['text', 'label']]
    df = df.dropna()
    df = df.drop_duplicates(subset=['text'])

    print("Applying text preprocessing...")
    df['clean_text'] = df['text'].apply(clean_text)

    # Loại bỏ text quá ngắn sau khi clean
    df = df[df['clean_text'].str.len() > 10]

    print(f"Dataset size after clean: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


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


def load_and_clean_vn(path):
    """
    Load và clean dataset tiếng Việt.
    Dataset VN dùng encoding utf-8, text đã sạch (không có header email).
    """
    df = pd.read_csv(path, encoding='utf-8')
    print(f"\n[VN Dataset] Raw: {len(df)} rows, columns: {list(df.columns)}")

    # Chuẩn hóa cột
    if 'label' not in df.columns and 'target' in df.columns:
        df = df.rename(columns={'target': 'label'})

    df = df[['text', 'label']]
    df = df.dropna()
    df = df.drop_duplicates(subset=['text'])

    # Chuyển label text sang số nếu cần
    if df['label'].dtype == 'object':
        label_map = {'ham': 0, 'spam': 1, 'normal': 0}
        df['label'] = df['label'].str.lower().map(label_map)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

    print("Applying Vietnamese text preprocessing...")
    df['clean_text'] = df['text'].apply(clean_text)

    # Loại bỏ text quá ngắn
    df = df[df['clean_text'].str.len() > 5]

    print(f"[VN Dataset] After clean: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


def merge_datasets(df_en, df_vn):
    """
    Merge dataset English + Vietnamese.
    Cân bằng lại nếu cần để tránh mất cân bằng quá nhiều.
    """
    print(f"\n{'='*50}")
    print("MERGING DATASETS")
    print(f"{'='*50}")
    print(f"English dataset: {len(df_en)} rows")
    print(f"Vietnamese dataset: {len(df_vn)} rows")

    # Thêm cột source để track nguồn
    df_en = df_en.copy()
    df_vn = df_vn.copy()
    df_en['source'] = 'spam_assassin'
    df_vn['source'] = 'dataset_vn'

    # Merge
    df_merged = pd.concat([df_en, df_vn], ignore_index=True)

    # Shuffle
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nMerged dataset: {len(df_merged)} rows")
    print(f"Label distribution:")
    print(f"  Normal (0): {len(df_merged[df_merged['label'] == 0])}")
    print(f"  Spam   (1): {len(df_merged[df_merged['label'] == 1])}")
    print(f"\nSource distribution:")
    print(df_merged['source'].value_counts().to_string())

    return df_merged