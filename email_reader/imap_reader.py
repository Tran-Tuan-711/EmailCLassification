"""
IMAPEmailReader — Module đọc email từ IMAP server.

Hỗ trợ:
  - Gmail (imap.gmail.com) — cần App Password
  - Outlook (outlook.office365.com)
  - Yahoo (imap.mail.yahoo.com)
  - Custom IMAP server

Sử dụng thư viện built-in: imaplib, email.
"""

import imaplib
import email
from email.header import decode_header
from email.utils import parseaddr
import re


# ─── IMAP Server auto-detection ──────────────────────────────────────────────

IMAP_SERVERS = {
    "gmail.com": ("imap.gmail.com", 993, "Gmail"),
    "googlemail.com": ("imap.gmail.com", 993, "Gmail"),
    "outlook.com": ("outlook.office365.com", 993, "Outlook"),
    "hotmail.com": ("outlook.office365.com", 993, "Outlook"),
    "live.com": ("outlook.office365.com", 993, "Outlook"),
    "yahoo.com": ("imap.mail.yahoo.com", 993, "Yahoo"),
    "yahoo.co.jp": ("imap.mail.yahoo.co.jp", 993, "Yahoo Japan"),
    "icloud.com": ("imap.mail.me.com", 993, "iCloud"),
    "me.com": ("imap.mail.me.com", 993, "iCloud"),
    "aol.com": ("imap.aol.com", 993, "AOL"),
    "zoho.com": ("imap.zoho.com", 993, "Zoho"),
    "protonmail.com": ("127.0.0.1", 1143, "ProtonMail Bridge"),
}


def detect_imap_server(email_addr):
    """
    Tự động nhận diện IMAP server từ địa chỉ email.

    Args:
        email_addr: Địa chỉ email (ví dụ: user@gmail.com)

    Returns:
        tuple: (server, port, display_name)
    """
    if not email_addr or "@" not in email_addr:
        return None, None, None

    domain = email_addr.split("@")[-1].strip().lower()

    if domain in IMAP_SERVERS:
        return IMAP_SERVERS[domain]

    # Fallback: thử imap.<domain>
    return (f"imap.{domain}", 993, domain.split(".")[0].capitalize())


# ─── IMAP Email Reader ───────────────────────────────────────────────────────

class IMAPEmailReader:
    """
    Đọc email từ IMAP server.

    Usage:
        reader = IMAPEmailReader("imap.gmail.com", 993)
        reader.login("user@gmail.com", "app_password")
        emails = reader.fetch_emails(count=10)
        reader.logout()
    """

    def __init__(self, server, port=993):
        self.server = server
        self.port = port
        self.mail = None

    def login(self, email_addr, password):
        """Kết nối và đăng nhập IMAP server."""
        try:
            self.mail = imaplib.IMAP4_SSL(self.server, self.port)
            self.mail.login(email_addr, password)
            return True, "Đăng nhập thành công!"
        except imaplib.IMAP4.error as e:
            return False, f"Lỗi đăng nhập: {str(e)}"
        except Exception as e:
            return False, f"Lỗi kết nối: {str(e)}"

    def fetch_emails(self, count=10, folder="INBOX"):
        """
        Đọc email từ folder chỉ định.

        Args:
            count: Số email muốn đọc (mới nhất)
            folder: Tên folder (default: INBOX)

        Returns:
            list of dict: [{subject, sender, sender_email, date, body}, ...]
        """
        if not self.mail:
            return []

        try:
            self.mail.select(folder)
            _, data = self.mail.search(None, "ALL")
            email_ids = data[0].split()

            if not email_ids:
                return []

            # Lấy N email mới nhất (từ cuối)
            latest_ids = email_ids[-count:]
            latest_ids.reverse()  # Mới nhất trước

            emails = []
            for eid in latest_ids:
                try:
                    _, msg_data = self.mail.fetch(eid, "(RFC822)")
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)

                    parsed = self._parse_email(msg)
                    if parsed:
                        emails.append(parsed)
                except Exception:
                    continue

            return emails

        except Exception as e:
            print(f"Lỗi đọc email: {e}")
            return []

    def _parse_email(self, msg):
        """Parse email message thành dict."""
        try:
            # Subject
            subject = self._decode_header(msg.get("Subject", ""))

            # Sender
            sender_raw = msg.get("From", "")
            sender_name, sender_email = parseaddr(sender_raw)
            sender_name = self._decode_header(sender_name) if sender_name else sender_email

            # Date
            date = msg.get("Date", "")

            # Body
            body = self._get_body(msg)

            return {
                "subject": subject,
                "sender": sender_name,
                "sender_email": sender_email,
                "date": date,
                "body": body,
            }
        except Exception:
            return None

    def _decode_header(self, header_value):
        """Decode email header (có thể encoded)."""
        if not header_value:
            return ""
        try:
            parts = decode_header(header_value)
            decoded = []
            for part, charset in parts:
                if isinstance(part, bytes):
                    charset = charset or "utf-8"
                    try:
                        decoded.append(part.decode(charset, errors="replace"))
                    except (LookupError, UnicodeDecodeError):
                        decoded.append(part.decode("utf-8", errors="replace"))
                else:
                    decoded.append(str(part))
            return " ".join(decoded)
        except Exception:
            return str(header_value)

    def _get_body(self, msg):
        """Trích xuất body text từ email (ưu tiên plain text, fallback HTML)."""
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disp = str(part.get("Content-Disposition", ""))

                # Bỏ qua attachment
                if "attachment" in content_disp:
                    continue

                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            body = payload.decode(charset, errors="replace")
                        except (LookupError, UnicodeDecodeError):
                            body = payload.decode("utf-8", errors="replace")
                        break  # Ưu tiên plain text

                elif content_type == "text/html" and not body:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            html = payload.decode(charset, errors="replace")
                        except (LookupError, UnicodeDecodeError):
                            html = payload.decode("utf-8", errors="replace")
                        body = self._html_to_text(html)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    body = payload.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    body = payload.decode("utf-8", errors="replace")

                if msg.get_content_type() == "text/html":
                    body = self._html_to_text(body)

        return body.strip()

    def _html_to_text(self, html):
        """Chuyển HTML thành plain text (đơn giản)."""
        # Xóa script và style
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Thay <br> và <p> bằng newline
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</?p[^>]*>', '\n', text, flags=re.IGNORECASE)
        # Xóa tất cả tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
        text = text.replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'")
        # Clean whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def logout(self):
        """Đóng kết nối IMAP."""
        if self.mail:
            try:
                self.mail.close()
                self.mail.logout()
            except Exception:
                pass
            self.mail = None