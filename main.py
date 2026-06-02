"""
Entry point — Chạy GUI Email Spam Classifier.
Usage: python main.py
"""

import sys
import os

# Đảm bảo import path đúng
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from GUI.app import main

if __name__ == "__main__":
    main()