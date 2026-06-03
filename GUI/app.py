"""
Email Spam Classifier — GUI Application
Giao diện đồ họa phân loại email spam sử dụng tkinter.

Tính năng:
  - Tab 1: Phân loại email thủ công (nhập sender + nội dung)
  - Tab 2: Đọc email từ IMAP server và phân loại batch

Kiến trúc: Rule Engine (VN) + CNN Model (SpamAssassin)
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model.predict_cnn import predict_email
from email_reader.imap_reader import IMAPEmailReader, detect_imap_server


# ═══════════════════════════════════════════════════════════════════════════
# COLOR PALETTE — Premium Light Theme
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg_dark": "#f1f5f9",         # Nền ứng dụng chính (Slate 100)
    "bg_card": "#ffffff",          # Nền của các thẻ (Card) chứa nội dung
    "bg_input": "#ffffff",         # Nền của các ô nhập liệu
    "bg_input_focus": "#f8fafc",
    "accent": "#0f172a",           # Màu đen Slate 900 (Premium/Corporate)
    "accent_hover": "#334155",     # Slate 700 khi di chuột qua
    "accent_dim": "#f1f5f9",       # Nền xám Slate 100 khi chọn dòng
    "danger": "#991b1b",           # Màu đỏ trầm (Muted Red) cho Spam
    "danger_hover": "#7f1d1d",
    "warning": "#d97706",          # Màu Amber/Vàng hổ phách khi đang tải
    "success": "#166534",          # Màu xanh trầm (Muted Green) cho Normal
    "text_primary": "#0f172a",     # Chữ chính màu Slate 900
    "text_secondary": "#475569",   # Chữ phụ màu Slate 600
    "text_muted": "#94a3b8",       # Chữ mờ màu Slate 400
    "border": "#cbd5e1",           # Đường viền Slate 300
    "border_focus": "#475569",     # Đường viền khi click vào input
    "scrollbar": "#cbd5e1",        # Màu scrollbar
    "tag_spam_bg": "#fee2e2",      # Nhãn Spam nền hồng nhạt
    "tag_spam_fg": "#991b1b",      # Nhãn Spam chữ đỏ trầm
    "tag_normal_bg": "#dcfce7",    # Nhãn Normal nền xanh nhạt
    "tag_normal_fg": "#166534",    # Nhãn Normal chữ xanh trầm
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class EmailClassifierApp:
    """GUI Application — Email Spam Classifier."""

    def __init__(self, root):
        self.root = root
        self.root.title("Email Spam Classifier — CNN + Rule Engine")
        self.root.geometry("960x720")
        self.root.minsize(800, 600)
        self.root.configure(bg=COLORS["bg_dark"])

        self._setup_styles()
        self._build_ui()

    # ─── Style Configuration ─────────────────────────────────────────────

    def _setup_styles(self):
        """Configure ttk styles for light theme."""
        style = ttk.Style()
        style.theme_use("clam")

        # Notebook (tabs)
        style.configure("TNotebook", background=COLORS["bg_dark"], borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=COLORS["bg_card"],
                        foreground=COLORS["text_secondary"],
                        padding=[20, 8],
                        font=("Segoe UI", 10, "bold"),
                        borderwidth=1,
                        bordercolor=COLORS["border"])
        style.map("TNotebook.Tab",
                  background=[("selected", COLORS["accent"]), ("active", COLORS["accent_dim"])],
                  foreground=[("selected", "#ffffff"), ("active", COLORS["accent"])])

        # Frame
        style.configure("Dark.TFrame", background=COLORS["bg_dark"])
        style.configure("Card.TFrame", background=COLORS["bg_card"])

        # Combobox
        style.configure("TCombobox",
                        background=COLORS["bg_card"],
                        foreground=COLORS["text_primary"],
                        bordercolor=COLORS["border"],
                        darkcolor=COLORS["border"],
                        lightcolor=COLORS["border"],
                        arrowcolor=COLORS["accent"],
                        font=("Segoe UI", 10))

        # Label
        style.configure("Title.TLabel",
                        background=COLORS["bg_dark"],
                        foreground=COLORS["text_primary"],
                        font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel",
                        background=COLORS["bg_dark"],
                        foreground=COLORS["text_secondary"],
                        font=("Segoe UI", 10, "italic"))
        style.configure("Dark.TLabel",
                        background=COLORS["bg_dark"],
                        foreground=COLORS["text_primary"],
                        font=("Segoe UI", 10))
        style.configure("Card.TLabel",
                        background=COLORS["bg_card"],
                        foreground=COLORS["text_primary"],
                        font=("Segoe UI", 10))
        style.configure("Accent.TLabel",
                        background=COLORS["bg_dark"],
                        foreground=COLORS["accent"],
                        font=("Segoe UI", 10, "bold"))

        # Button
        style.configure("Accent.TButton",
                        background=COLORS["accent"],
                        foreground="#ffffff",
                        font=("Segoe UI", 11, "bold"),
                        padding=[20, 10])
        style.map("Accent.TButton",
                  background=[("active", COLORS["accent_hover"]),
                              ("disabled", COLORS["text_muted"])],
                  foreground=[("active", "#ffffff"),
                              ("disabled", "#ffffff")])

        style.configure("Danger.TButton",
                        background=COLORS["danger"],
                        foreground="#ffffff",
                        font=("Segoe UI", 10, "bold"),
                        padding=[15, 8])
        style.map("Danger.TButton",
                  background=[("active", COLORS["danger_hover"])])

        # Separator
        style.configure("Dark.TSeparator", background=COLORS["border"])

    # ─── Main UI Layout ──────────────────────────────────────────────────

    def _build_ui(self):
        """Build the main application UI."""
        # Header
        header = ttk.Frame(self.root, style="Dark.TFrame")
        header.pack(fill="x", padx=30, pady=(20, 5))

        ttk.Label(header, text="Email Filter Pro",
                  style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Hệ thống lọc và phân loại thư rác",
                  style="Subtitle.TLabel").pack(side="right", pady=(12, 0))

        # Separator
        ttk.Separator(self.root, orient="horizontal",
                       style="Dark.TSeparator").pack(fill="x", padx=30, pady=(5, 15))

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self._build_classify_tab()
        self._build_imap_tab()

    # ─── Tab 1: Phan loai thu cong ────────────────────────────────────────

    def _build_classify_tab(self):
        tab = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(tab, text="  Phân loại thủ công  ")

        # Create a PanedWindow to allow dragging/resizing the panels
        paned = ttk.Panedwindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=15, pady=15)

        # Left panel — Input container
        left_container = ttk.Frame(paned, style="Dark.TFrame")
        paned.add(left_container, weight=1)

        left = ttk.Frame(left_container, style="Dark.TFrame")
        left.pack(fill="both", expand=True, padx=(0, 10))

        # Sender email input
        ttk.Label(left, text="Email người gửi (tùy chọn)",
                  style="Dark.TLabel").pack(anchor="w", pady=(0, 5))

        self.sender_input = tk.Entry(left,
                                     font=("Segoe UI", 11),
                                     bg=COLORS["bg_input"],
                                     fg=COLORS["text_primary"],
                                     insertbackground=COLORS["accent"],
                                     relief="flat",
                                     highlightthickness=1,
                                     highlightcolor=COLORS["border_focus"],
                                     highlightbackground=COLORS["border"])
        self.sender_input.pack(fill="x", pady=(0, 15), ipady=8)
        self.sender_input.insert(0, "vi du: sender@gmail.com")
        self.sender_input.config(fg=COLORS["text_muted"])
        self.sender_input.bind("<FocusIn>", lambda e: self._on_focus_in(self.sender_input, "vi du: sender@gmail.com"))
        self.sender_input.bind("<FocusOut>", lambda e: self._on_focus_out(self.sender_input, "vi du: sender@gmail.com"))

        # Chọn thuật toán phân loại (Dropdown / Combobox)
        model_frame = ttk.Frame(left, style="Dark.TFrame")
        model_frame.pack(fill="x", pady=(0, 15))

        ttk.Label(model_frame, text="Thuật toán:", style="Dark.TLabel").pack(side="left", padx=(0, 10))

        self.model_combo = ttk.Combobox(model_frame, values=["CNN (Deep Learning)", "Logistic Regression (LR)"], state="readonly", font=("Segoe UI", 10))
        self.model_combo.set("CNN (Deep Learning)")
        self.model_combo.pack(side="left", fill="x", expand=True)

        # Email content input
        ttk.Label(left, text="Nội dung Email (Tiêu đề + Thân bài)",
                  style="Dark.TLabel").pack(anchor="w", pady=(0, 5))

        self.email_input = scrolledtext.ScrolledText(
            left,
            font=("Segoe UI", 11),
            bg=COLORS["bg_input"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["accent"],
            relief="flat",
            wrap="word",
            highlightthickness=1,
            highlightcolor=COLORS["border_focus"],
            highlightbackground=COLORS["border"],
            height=12,
            width=40,
        )
        self.email_input.pack(fill="both", expand=True, pady=(0, 15))

        # Buttons
        btn_frame = ttk.Frame(left, style="Dark.TFrame")
        btn_frame.pack(fill="x")

        ttk.Button(btn_frame, text="Phân tích email",
                   style="Accent.TButton",
                   command=self._on_classify).pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(btn_frame, text="Xóa sạch",
                   style="Danger.TButton",
                   command=self._on_clear).pack(side="right")

        # Right panel — Result container
        right_container = ttk.Frame(paned, style="Dark.TFrame")
        paned.add(right_container, weight=1)

        right = ttk.Frame(right_container, style="Dark.TFrame")
        right.pack(fill="both", expand=True, padx=(10, 0))

        ttk.Label(right, text="Kết quả phân tích",
                  style="Dark.TLabel").pack(anchor="w", pady=(0, 10))

        # Result card
        self.result_frame = tk.Frame(right, bg=COLORS["bg_card"],
                                     highlightthickness=1,
                                     highlightbackground=COLORS["border"])
        self.result_frame.pack(fill="both", expand=True)

        # Label result (badge)
        self.result_label = tk.Label(self.result_frame,
                                     text="---",
                                     font=("Segoe UI", 13, "bold"),
                                     bg=COLORS["bg_card"],
                                     fg=COLORS["text_muted"],
                                     padx=15, pady=6)
        self.result_label.pack(pady=(20, 10))

        # Confidence
        self.confidence_label = tk.Label(self.result_frame,
                                          text="",
                                          font=("Segoe UI", 11, "bold"),
                                          bg=COLORS["bg_card"],
                                          fg=COLORS["text_primary"])
        self.confidence_label.pack(pady=(0, 8))

        # Method
        self.method_label = tk.Label(self.result_frame,
                                     text="",
                                     font=("Segoe UI", 9),
                                     bg=COLORS["bg_card"],
                                     fg=COLORS["text_secondary"])
        self.method_label.pack(pady=(0, 10))

        # Details
        self.details_text = scrolledtext.ScrolledText(
            self.result_frame,
            font=("Segoe UI", 9),
            bg=COLORS["bg_input"],
            fg=COLORS["text_secondary"],
            relief="flat",
            wrap="word",
            height=8,
            width=40,
            state="disabled",
        )
        self.details_text.pack(fill="both", expand=True, padx=15, pady=(5, 15))

    # ─── Tab 2: IMAP Reader ────────────────────────────────────────────

    def _build_imap_tab(self):
        tab = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(tab, text="  Quét hòm thư (IMAP)  ")

        # Top — Connection form
        form = ttk.Frame(tab, style="Dark.TFrame")
        form.pack(fill="x", padx=15, pady=(15, 10))

        # Email
        row1 = ttk.Frame(form, style="Dark.TFrame")
        row1.pack(fill="x", pady=(0, 8))

        ttk.Label(row1, text="Địa chỉ Email:", style="Dark.TLabel",
                  width=16).pack(side="left")

        self.imap_email = tk.Entry(row1,
                                   font=("Segoe UI", 11),
                                   bg=COLORS["bg_input"],
                                   fg=COLORS["text_primary"],
                                   insertbackground=COLORS["accent"],
                                   relief="flat",
                                   highlightthickness=1,
                                   highlightcolor=COLORS["border_focus"],
                                   highlightbackground=COLORS["border"])
        self.imap_email.pack(side="left", fill="x", expand=True, ipady=6)
        self.imap_email.bind("<KeyRelease>", self._on_email_change)

        # Server auto-detect label
        self.server_label = tk.Label(row1, text="",
                                     font=("Segoe UI", 9),
                                     bg=COLORS["bg_dark"],
                                     fg=COLORS["accent"])
        self.server_label.pack(side="right", padx=(10, 0))

        # Password
        row2 = ttk.Frame(form, style="Dark.TFrame")
        row2.pack(fill="x", pady=(0, 8))

        ttk.Label(row2, text="App Password:", style="Dark.TLabel",
                  width=16).pack(side="left")

        self.imap_password = tk.Entry(row2,
                                      font=("Segoe UI", 11),
                                      bg=COLORS["bg_input"],
                                      fg=COLORS["text_primary"],
                                      insertbackground=COLORS["accent"],
                                      relief="flat",
                                      show="*",
                                      highlightthickness=1,
                                      highlightcolor=COLORS["border_focus"],
                                      highlightbackground=COLORS["border"])
        self.imap_password.pack(side="left", fill="x", expand=True, ipady=6)

        # Algorithm selection
        row_algo = ttk.Frame(form, style="Dark.TFrame")
        row_algo.pack(fill="x", pady=(0, 8))

        ttk.Label(row_algo, text="Thuật toán:", style="Dark.TLabel",
                  width=16).pack(side="left")

        self.imap_model_combo = ttk.Combobox(row_algo, values=["CNN (Deep Learning)", "Logistic Regression (LR)"], state="readonly", font=("Segoe UI", 10))
        self.imap_model_combo.set("CNN (Deep Learning)")
        self.imap_model_combo.pack(side="left", fill="x", expand=True)

        # Count
        row3 = ttk.Frame(form, style="Dark.TFrame")
        row3.pack(fill="x", pady=(0, 8))

        ttk.Label(row3, text="Số lượng đọc:", style="Dark.TLabel",
                  width=16).pack(side="left")

        self.imap_count = tk.Spinbox(row3,
                                     from_=1, to=50, value=10,
                                     font=("Segoe UI", 11),
                                     bg=COLORS["bg_input"],
                                     fg=COLORS["text_primary"],
                                     buttonbackground=COLORS["bg_card"],
                                     relief="flat",
                                     width=8,
                                     highlightthickness=1,
                                     highlightcolor=COLORS["border_focus"],
                                     highlightbackground=COLORS["border"])
        self.imap_count.pack(side="left", ipady=6)

        # Connect button
        self.connect_btn = ttk.Button(row3, text="Kết nối & Phân loại",
                                      style="Accent.TButton",
                                      command=self._on_connect_imap)
        self.connect_btn.pack(side="right", padx=(10, 0))

        # Status label
        self.imap_status = tk.Label(form, text="",
                                    font=("Segoe UI", 10),
                                    bg=COLORS["bg_dark"],
                                    fg=COLORS["text_secondary"])
        self.imap_status.pack(fill="x", pady=(5, 0))

        # Separator
        ttk.Separator(tab, orient="horizontal",
                       style="Dark.TSeparator").pack(fill="x", padx=15, pady=5)

        # Bottom — Email list results
        list_frame = ttk.Frame(tab, style="Dark.TFrame")
        list_frame.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        # Treeview for email list
        columns = ("sender", "subject", "label", "confidence", "method")
        self.email_tree = ttk.Treeview(list_frame, columns=columns,
                                        show="headings", height=12)

        self.email_tree.heading("sender", text="Người gửi")
        self.email_tree.heading("subject", text="Tiêu đề")
        self.email_tree.heading("label", text="Nhãn")
        self.email_tree.heading("confidence", text="Độ tin cậy")
        self.email_tree.heading("method", text="Phương pháp")

        self.email_tree.column("sender", width=200)
        self.email_tree.column("subject", width=300)
        self.email_tree.column("label", width=80, anchor="center")
        self.email_tree.column("confidence", width=100, anchor="center")
        self.email_tree.column("method", width=120, anchor="center")

        # Treeview styling
        style = ttk.Style()
        style.configure("Treeview",
                        background=COLORS["bg_card"],
                        foreground=COLORS["text_primary"],
                        fieldbackground=COLORS["bg_card"],
                        font=("Segoe UI", 10),
                        rowheight=35)
        style.configure("Treeview.Heading",
                        background=COLORS["accent_dim"],
                        foreground=COLORS["accent"],
                        font=("Segoe UI", 10, "bold"))
        style.map("Treeview.Heading",
                  background=[("active", COLORS["border"]),
                              ("!active", COLORS["accent_dim"])],
                  foreground=[("active", COLORS["accent"]),
                              ("!active", COLORS["accent"])])
        style.map("Treeview",
                  background=[("selected", COLORS["accent_dim"])],
                  foreground=[("selected", COLORS["accent"])])

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                                   command=self.email_tree.yview)
        self.email_tree.configure(yscrollcommand=scrollbar.set)

        self.email_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind click event
        self.email_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Tag colors for Spam/Normal
        self.email_tree.tag_configure("spam", foreground=COLORS["tag_spam_fg"], background=COLORS["tag_spam_bg"])
        self.email_tree.tag_configure("normal", foreground=COLORS["tag_normal_fg"], background=COLORS["tag_normal_bg"])

    # ─── Event Handlers ──────────────────────────────────────────────────

    def _on_focus_in(self, widget, placeholder):
        """Clear placeholder on focus."""
        if widget.get() == placeholder:
            widget.delete(0, "end")
            widget.config(fg=COLORS["text_primary"])

    def _on_focus_out(self, widget, placeholder):
        """Restore placeholder if empty."""
        if not widget.get().strip():
            widget.insert(0, placeholder)
            widget.config(fg=COLORS["text_muted"])

    def _on_email_change(self, event=None):
        """Update server auto-detect label when email changes."""
        email_addr = self.imap_email.get().strip()
        server, port, name = detect_imap_server(email_addr)
        if server and name:
            self.server_label.config(text=f"{name} ({server}:{port})")
        else:
            self.server_label.config(text="")

    def _on_classify(self):
        """Handle classify button click."""
        text = self.email_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Thieu noi dung", "Vui long nhap noi dung email!")
            return

        sender = self.sender_input.get().strip()
        if sender == "vi du: sender@gmail.com":
            sender = ""

        # Run prediction
        try:
            model_name = self.model_combo.get()
            model_type = "cnn" if "CNN" in model_name else "lr"
            result = predict_email(text, sender_email=sender, use_rules=True, model_type=model_type)
            self._display_result(result)
        except Exception as e:
            messagebox.showerror("Loi", f"Loi phan loai: {str(e)}")

    def _display_result(self, result):
        """Display classification result in the right panel."""
        label = result["label"]
        confidence = result["confidence"]
        method = result["method"]
        details = result.get("details", "")
        matched = result.get("matched_rules", [])

        # Color based on label
        if label == "Spam":
            color = COLORS["danger"]
            bg_color = COLORS["tag_spam_bg"]
            icon = "SPAM (Thư rác)"
        else:
            color = COLORS["success"]
            bg_color = COLORS["tag_normal_bg"]
            icon = "NORMAL (Thư thường)"

        # Cấu hình màu sắc của card kết quả (giữ nền trắng, chỉ đổi màu viền và màu badge)
        self.result_frame.config(bg=COLORS["bg_card"], highlightbackground=color, highlightthickness=2)
        self.result_label.config(text=icon, fg=color, bg=bg_color)
        self.confidence_label.config(text=f"Độ tin cậy: {confidence:.1%}", fg=COLORS["text_primary"], bg=COLORS["bg_card"])

        # Method display
        method_map = {
            "rule_whitelist": "Bộ luật kiểm tra (Danh sách tin cậy)",
            "rule_keyword": "Bộ luật kiểm tra (Từ khóa nhạy cảm)",
            "model_cnn": "Mô hình học máy Deep Learning (CNN)",
            "model_lr": "Mô hình học máy Logistic Regression (LR)",
        }
        self.method_label.config(text=f"Phương pháp: {method_map.get(method, method)}", fg=COLORS["text_secondary"], bg=COLORS["bg_card"])

        # Details
        detail_lines = [details]
        if matched:
            detail_lines.append("")
            detail_lines.append("-- Các luật vi phạm trùng khớp --")
            for m in matched:
                kws = ', '.join(m['matched_keywords'][:5])
                detail_lines.append(
                    f"* {m['group_name']} (Trọng số={m['weight']}, "
                    f"Điểm số nhóm={m['group_score']:.1f}): {kws}"
                )

        self.details_text.config(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", "\n".join(detail_lines))
        self.details_text.config(state="disabled")

    def _on_clear(self):
        """Clear all inputs and results."""
        self.email_input.delete("1.0", "end")
        self.sender_input.delete(0, "end")
        self.sender_input.insert(0, "vi du: sender@gmail.com")
        self.sender_input.config(fg=COLORS["text_muted"])
        
        # Reset card
        self.result_frame.config(bg=COLORS["bg_card"], highlightbackground=COLORS["border"], highlightthickness=1)
        self.result_label.config(text="---", fg=COLORS["text_muted"], bg=COLORS["bg_card"])
        self.confidence_label.config(text="", bg=COLORS["bg_card"])
        self.method_label.config(text="", bg=COLORS["bg_card"])
        
        self.details_text.config(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.config(state="disabled")

    def _on_connect_imap(self):
        """Handle IMAP connect button click."""
        email_addr = self.imap_email.get().strip()
        password = self.imap_password.get().strip()

        if not email_addr or not password:
            messagebox.showwarning("Thieu thong tin",
                                   "Vui long nhap email va app password!")
            return

        try:
            count = int(self.imap_count.get())
        except ValueError:
            count = 10

        server, port, name = detect_imap_server(email_addr)
        if not server:
            messagebox.showerror("Loi",
                                 "Khong the nhan dien IMAP server tu email!")
            return

        # Disable button and show status
        self.connect_btn.config(state="disabled")
        self.imap_status.config(text=f"Dang ket noi {name}...",
                                fg=COLORS["warning"])

        # Run in background thread
        thread = threading.Thread(
            target=self._fetch_and_classify,
            args=(server, port, email_addr, password, count),
            daemon=True
        )
        thread.start()

    def _fetch_and_classify(self, server, port, email_addr, password, count):
        """Fetch emails from IMAP and classify them (runs in background thread)."""
        try:
            reader = IMAPEmailReader(server, port)
            success, msg = reader.login(email_addr, password)

            if not success:
                self.root.after(0, lambda: self._imap_error(msg))
                return

            self.root.after(0, lambda: self.imap_status.config(
                text=f"Dang doc {count} email...", fg=COLORS["warning"]))

            emails = reader.fetch_emails(count=count)
            reader.logout()

            if not emails:
                self.root.after(0, lambda: self._imap_done([], "Khong tim thay email nao."))
                return

            # Classify each email
            model_name = self.imap_model_combo.get()
            model_type = "cnn" if "CNN" in model_name else "lr"
            results = []
            for i, em in enumerate(emails):
                self.root.after(0, lambda i=i, total=len(emails):
                    self.imap_status.config(
                        text=f"Phan loai {i+1}/{total}...",
                        fg=COLORS["warning"]))

                text = f"{em['subject']}\n{em['body']}"
                pred = predict_email(text,
                                     sender_email=em.get("sender_email", ""),
                                     use_rules=True,
                                     model_type=model_type)
                results.append({**em, **pred})

            self.root.after(0, lambda: self._imap_done(
                results,
                f"Da phan loai {len(results)} email!"))

        except Exception as e:
            self.root.after(0, lambda: self._imap_error(str(e)))

    def _imap_error(self, msg):
        """Handle IMAP error."""
        self.imap_status.config(text=f"LOI: {msg}", fg=COLORS["danger"])
        self.connect_btn.config(state="normal")

    def _imap_done(self, results, status_msg):
        """Handle IMAP fetch completion."""
        self.imap_status.config(text=status_msg, fg=COLORS["success"])
        self.connect_btn.config(state="normal")

        # Clear old data
        for item in self.email_tree.get_children():
            self.email_tree.delete(item)

        # Populate treeview
        spam_count = 0
        for r in results:
            tag = "spam" if r["label"] == "Spam" else "normal"
            if r["label"] == "Spam":
                spam_count += 1

            self.email_tree.insert("", "end", values=(
                r.get("sender_email", r.get("sender", "")),
                r.get("subject", "")[:80],
                r["label"],
                f"{r['confidence']:.1%}",
                r["method"],
            ), tags=(tag,))

        # Update status with stats
        normal_count = len(results) - spam_count
        self.imap_status.config(
            text=f"{len(results)} email - "
                 f"Normal: {normal_count}, Spam: {spam_count}")

        # Store results for detail view
        self._imap_results = results

    def _on_tree_select(self, event):
        """Handle treeview row selection — show email details."""
        selection = self.email_tree.selection()
        if not selection:
            return

        idx = self.email_tree.index(selection[0])
        if hasattr(self, '_imap_results') and idx < len(self._imap_results):
            r = self._imap_results[idx]

            # Switch to classify tab to show details
            self.notebook.select(0)

            # Fill in data
            self.sender_input.delete(0, "end")
            self.sender_input.insert(0, r.get("sender_email", ""))
            self.sender_input.config(fg=COLORS["text_primary"])

            self.email_input.delete("1.0", "end")
            body = f"{r.get('subject', '')}\n{r.get('body', '')}"
            self.email_input.insert("1.0", body[:3000])

            # Show result
            self._display_result(r)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app = EmailClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
