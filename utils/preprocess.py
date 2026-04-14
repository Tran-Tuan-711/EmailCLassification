import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords quietly, chỉ khi chưa có
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words_en = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Stopwords tiếng Việt cơ bản
stop_words_vi = set([
    'và', 'của', 'là', 'có', 'cho', 'với', 'các', 'được', 'trong',
    'này', 'đã', 'để', 'từ', 'một', 'những', 'không', 'khi', 'tại',
    'hay', 'hoặc', 'nhưng', 'cũng', 'vì', 'nếu', 'thì', 'mà',
    'đang', 'sẽ', 'rất', 'vẫn', 'đều', 'theo', 'về', 'do',
    'bị', 'ra', 'lên', 'đến', 'nên', 'sau', 'trước', 'bạn',
    'tôi', 'anh', 'chị', 'em', 'ông', 'bà', 'họ', 'chúng',
])


def _is_vietnamese(text):
    """
    Phát hiện text có phải tiếng Việt không.
    Dựa trên sự xuất hiện của ký tự Unicode đặc trưng cho tiếng Việt.
    """
    # Các ký tự dấu đặc trưng tiếng Việt
    vi_pattern = re.compile(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ'
                            r'ìíịỉĩòóọỏõôồốộổỗơờớợởỡ'
                            r'ùúụủũưừứựửữỳýỵỷỹđ'
                            r'ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄ'
                            r'ÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ'
                            r'ÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]')
    matches = vi_pattern.findall(text)
    # Nếu có >= 2 ký tự tiếng Việt → coi là tiếng Việt
    return len(matches) >= 2


def clean_text_en(text):
    """
    Tiền xử lý text email tiếng Anh:
    - Lowercase
    - Xóa HTML tags, URLs, email addresses, số, ký tự đặc biệt
    - Loại bỏ stopwords tiếng Anh
    - Stemming (Porter Stemmer)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)        # HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)  # URLs
    text = re.sub(r'\S+@\S+', ' ', text)      # Email addresses
    text = re.sub(r'\d+', ' ', text)           # Numbers
    text = re.sub(r'[^a-z\s]', ' ', text)     # Special characters

    words = text.split()
    words = [w for w in words if w not in stop_words_en]
    words = [stemmer.stem(w) for w in words]

    return " ".join(words)


def clean_text_vi(text):
    """
    Tiền xử lý text email tiếng Việt:
    - Lowercase
    - Xóa HTML tags, URLs, email addresses, số
    - Giữ lại ký tự tiếng Việt (dấu Unicode)
    - Loại bỏ stopwords tiếng Việt cơ bản
    - KHÔNG stem (tiếng Việt không dùng stemming)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)        # HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)  # URLs
    text = re.sub(r'\S+@\S+', ' ', text)      # Email addresses
    text = re.sub(r'\d+', ' ', text)           # Numbers
    # Giữ lại chữ cái Latin + Unicode Vietnamese + khoảng trắng
    text = re.sub(r'[^\w\s]', ' ', text)
    # Loại bỏ underscore (vì \w bao gồm _)
    text = text.replace('_', ' ')

    words = text.split()
    words = [w for w in words if w not in stop_words_vi]

    return " ".join(words)


def clean_text(text):
    """
    Tiền xử lý text tự động — phát hiện ngôn ngữ và chọn pipeline phù hợp.
    - Tiếng Việt: giữ dấu, bỏ stopwords VN, không stem
    - Tiếng Anh: bỏ stopwords EN, stemming
    """
    if not isinstance(text, str):
        return ""

    if _is_vietnamese(text):
        return clean_text_vi(text)
    else:
        return clean_text_en(text)