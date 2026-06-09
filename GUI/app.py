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

from model.predict_cnn import predict_email as predict_email_cnn
from model.predict_fasttext import predict_email as predict_email_fasttext
from email_reader.imap_reader import IMAPEmailReader, detect_imap_server


# ═══════════════════════════════════════════════════════════════════════════
# COLOR PALETTE — Dark Theme
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg_dark": "#0f0f1a",
    "bg_card": "#1a1a2e",
    "bg_input": "#16213e",
    "bg_input_focus": "#1a2744",
    "accent": "#00d4aa",
    "accent_hover": "#00f5c4",
    "accent_dim": "#007a63",
    "danger": "#ff4757",
    "danger_hover": "#ff6b7a",
    "warning": "#ffa502",
    "success": "#2ed573",
    "text_primary": "#e8e8f0",
    "text_secondary": "#8888aa",
    "text_muted": "#555577",
    "border": "#2a2a4a",
    "border_focus": "#00d4aa",
    "scrollbar": "#333355",
    "tag_spam_bg": "#3d1515",
    "tag_spam_fg": "#ff4757",
    "tag_normal_bg": "#153d2b",
    "tag_normal_fg": "#2ed573",
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class EmailClassifierApp:
    """GUI Application — Email Spam Classifier."""

    def __init__(self, root):
        self.root = root
        self.root.title("Email Spam Classifier — Multi-Model + Rule Engine")
        self.root.geometry("960x720")
        self.root.minsize(800, 600)
        self.root.configure(bg=COLORS["bg_dark"])

        self._setup_styles()
        self._build_ui()

    # ─── Style Configuration ─────────────────────────────────────────────

    def _setup_styles(self):
        """Configure ttk styles for dark theme."""
        style = ttk.Style()
        style.theme_use("clam")

        # Notebook (tabs)
        style.configure("TNotebook", background=COLORS["bg_dark"], borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=COLORS["bg_card"],
                        foreground=COLORS["text_secondary"],
                        padding=[20, 10],
                        font=("Segoe UI", 11))
        style.map("TNotebook.Tab",
                  background=[("selected", COLORS["bg_dark"])],
                  foreground=[("selected", COLORS["accent"])])

        # Frame
        style.configure("Dark.TFrame", background=COLORS["bg_dark"])
        style.configure("Card.TFrame", background=COLORS["bg_card"])

        # Label
        style.configure("Title.TLabel",
                        background=COLORS["bg_dark"],
                        foreground=COLORS["text_primary"],
                        font=("Segoe UI", 22, "bold"))
        style.configure("Subtitle.TLabel",
                        background=COLORS["bg_dark"],
                        foreground=COLORS["text_secondary"],
                        font=("Segoe UI", 10))
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
                        font=("Segoe UI", 10))

        # Button
        style.configure("Accent.TButton",
                        background=COLORS["accent"],
                        foreground="#000000",
                        font=("Segoe UI", 11, "bold"),
                        padding=[20, 10])
        style.map("Accent.TButton",
                  background=[("active", COLORS["accent_hover"]),
                              ("disabled", COLORS["text_muted"])])

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

        ttk.Label(header, text="Email Spam Classifier",
                  style="Title.TLabel").pack(side="left")
        # Model selection dropdown in header
        selector_frame = ttk.Frame(header, style="Dark.TFrame")
        selector_frame.pack(side="right", pady=(5, 0))

        ttk.Label(selector_frame, text="Mô hình:",
                  font=("Segoe UI", 10, "bold"),
                  foreground=COLORS["text_secondary"],
                  background=COLORS["bg_dark"]).pack(side="left", padx=(0, 8))
        
        self.model_var = tk.StringVar(value="fastText Model")
        self.model_combobox = ttk.Combobox(
            selector_frame,
            textvariable=self.model_var,
            values=["fastText Model", "CNN Model"],
            state="readonly",
            width=20,
            font=("Segoe UI", 10)
        )
        self.model_combobox.pack(side="left")

        # Separator
        ttk.Separator(self.root, orient="horizontal",
                       style="Dark.TSeparator").pack(fill="x", padx=30, pady=(5, 10))

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self._build_classify_tab()
        self._build_imap_tab()

    # ─── Tab 1: Phan loai thu cong ────────────────────────────────────────

    def _build_classify_tab(self):
        tab = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(tab, text="  Phan loai Email  ")

        # Left panel — Input
        left = ttk.Frame(tab, style="Dark.TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(15, 5), pady=15)

        # Sender email input
        ttk.Label(left, text="Email nguoi gui (khong bat buoc)",
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

        # Email content input
        ttk.Label(left, text="Noi dung Email (Subject + Body)",
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
        )
        self.email_input.pack(fill="both", expand=True, pady=(0, 15))

        # Buttons
        btn_frame = ttk.Frame(left, style="Dark.TFrame")
        btn_frame.pack(fill="x")

        ttk.Button(btn_frame, text="Phan loai",
                   style="Accent.TButton",
                   command=self._on_classify).pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(btn_frame, text="Xoa",
                   style="Danger.TButton",
                   command=self._on_clear).pack(side="right")

        # Right panel — Result
        right = ttk.Frame(tab, style="Dark.TFrame")
        right.pack(side="right", fill="both", expand=True, padx=(5, 15), pady=15)

        ttk.Label(right, text="Ket qua phan loai",
                  style="Dark.TLabel").pack(anchor="w", pady=(0, 10))

        # Result card
        self.result_frame = tk.Frame(right, bg=COLORS["bg_card"],
                                     highlightthickness=1,
                                     highlightbackground=COLORS["border"])
        self.result_frame.pack(fill="both", expand=True)

        # Label result (big)
        self.result_label = tk.Label(self.result_frame,
                                     text="---",
                                     font=("Segoe UI", 36, "bold"),
                                     bg=COLORS["bg_card"],
                                     fg=COLORS["text_muted"])
        self.result_label.pack(pady=(30, 5))

        # Confidence
        self.confidence_label = tk.Label(self.result_frame,
                                          text="",
                                          font=("Segoe UI", 14),
                                          bg=COLORS["bg_card"],
                                          fg=COLORS["text_secondary"])
        self.confidence_label.pack(pady=(0, 10))

        # Method
        self.method_label = tk.Label(self.result_frame,
                                     text="",
                                     font=("Segoe UI", 10),
                                     bg=COLORS["bg_card"],
                                     fg=COLORS["text_muted"])
        self.method_label.pack(pady=(0, 5))

        # Details
        self.details_text = scrolledtext.ScrolledText(
            self.result_frame,
            font=("Segoe UI", 9),
            bg=COLORS["bg_input"],
            fg=COLORS["text_secondary"],
            relief="flat",
            wrap="word",
            height=8,
            state="disabled",
        )
        self.details_text.pack(fill="both", expand=True, padx=15, pady=(5, 15))

    # ─── Tab 2: IMAP Reader ────────────────────────────────────────────

    def _build_imap_tab(self):
        tab = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(tab, text="  Doc Email (IMAP)  ")

        # Top — Connection form
        form = ttk.Frame(tab, style="Dark.TFrame")
        form.pack(fill="x", padx=15, pady=(15, 10))

        # Email
        row1 = ttk.Frame(form, style="Dark.TFrame")
        row1.pack(fill="x", pady=(0, 8))

        ttk.Label(row1, text="Email:", style="Dark.TLabel",
                  width=14).pack(side="left")

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
                  width=14).pack(side="left")

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

        # Count
        row3 = ttk.Frame(form, style="Dark.TFrame")
        row3.pack(fill="x", pady=(0, 8))

        ttk.Label(row3, text="So email:", style="Dark.TLabel",
                  width=14).pack(side="left")

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
        self.connect_btn = ttk.Button(row3, text="Ket noi & Phan loai",
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

        self.email_tree.heading("sender", text="Nguoi gui")
        self.email_tree.heading("subject", text="Tieu de")
        self.email_tree.heading("label", text="Nhan")
        self.email_tree.heading("confidence", text="Confidence")
        self.email_tree.heading("method", text="Phuong phap")

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
                        rowheight=30)
        style.configure("Treeview.Heading",
                        background=COLORS["bg_input"],
                        foreground=COLORS["accent"],
                        font=("Segoe UI", 10, "bold"))
        style.map("Treeview",
                  background=[("selected", COLORS["accent_dim"])],
                  foreground=[("selected", COLORS["text_primary"])])

        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                                   command=self.email_tree.yview)
        self.email_tree.configure(yscrollcommand=scrollbar.set)

        self.email_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind click event
        self.email_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Tag colors for Spam/Normal
        self.email_tree.tag_configure("spam", foreground=COLORS["danger"])
        self.email_tree.tag_configure("normal", foreground=COLORS["success"])

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
            model_type = self.model_var.get()
            if model_type == "CNN Model":
                result = predict_email_cnn(text, sender_email=sender, use_rules=True)
            else:
                result = predict_email_fasttext(text, sender_email=sender, use_rules=True)
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
            icon = "[SPAM]"
        else:
            color = COLORS["success"]
            icon = "[OK]"

        self.result_label.config(text=f"{icon} {label}", fg=color)
        self.confidence_label.config(text=f"Confidence: {confidence:.1%}")

        # Method display
        method_map = {
            "rule_whitelist": "Rule Engine (Whitelist)",
            "rule_keyword": "Rule Engine (Keywords)",
            "model_cnn": "CNN Model",
            "model_fasttext": "fastText Model",
        }
        self.method_label.config(text=method_map.get(method, method))

        # Details
        detail_lines = [details]
        if matched:
            detail_lines.append("")
            detail_lines.append("-- Matched Rules --")
            for m in matched:
                kws = ', '.join(m['matched_keywords'][:5])
                detail_lines.append(
                    f"* {m['group_name']} (weight={m['weight']}, "
                    f"score={m['group_score']:.1f}): {kws}"
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
        self.result_label.config(text="---", fg=COLORS["text_muted"])
        self.confidence_label.config(text="")
        self.method_label.config(text="")
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

        # Get model type
        model_type = self.model_var.get()

        # Run in background thread
        thread = threading.Thread(
            target=self._fetch_and_classify,
            args=(server, port, email_addr, password, count, model_type),
            daemon=True
        )
        thread.start()

    def _fetch_and_classify(self, server, port, email_addr, password, count, model_type):
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
            results = []
            for i, em in enumerate(emails):
                self.root.after(0, lambda i=i, total=len(emails):
                    self.imap_status.config(
                        text=f"Phan loai {i+1}/{total}...",
                        fg=COLORS["warning"]))

                text = f"{em['subject']}\n{em['body']}"
                if model_type == "CNN Model":
                    pred = predict_email_cnn(text,
                                             sender_email=em.get("sender_email", ""),
                                             use_rules=True)
                else:
                    pred = predict_email_fasttext(text,
                                                  sender_email=em.get("sender_email", ""),
                                                  use_rules=True)
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
