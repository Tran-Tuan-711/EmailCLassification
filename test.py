"""
Test cases cho hệ thống phân loại email.
Chạy: python test.py
"""
import sys

# Khắc phục lỗi UnicodeEncodeError trên terminal Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from model.predict_cnn import predict_email as predict_email_cnn
from model.predict_lr import predict_email as predict_email_lr


def run_test():
    test_cases = [
        # ===== Spam (English) =====
        {
            "text": "Win a FREE iPhone now!!! Click here to claim your prize",
            "sender": "promo@randomspam.com",
            "expected": "Spam",
        },

        # ===== Spam (Vietnamese) =====
        {
            "text": "Chúc mừng bạn đã trúng thưởng 1 chiếc xe SH! Nhấn link để nhận ngay!!!",
            "sender": "khuyenmai@trungthuong.vn",
            "expected": "Spam",
        },
        {
            "text": "Tài khoản của bạn bị khóa, vui lòng xác minh ngay lập tức để tránh bị xóa.",
            "sender": "security@fakebank.vn",
            "expected": "Spam",
        },
        {
            "text": "Vay tiền nhanh lãi suất thấp, không cần thế chấp, đăng ký ngay!",
            "sender": "vaynhanh@loan247.vn",
            "expected": "Spam",
        },

        # ===== Normal (Vietnamese — gmail.com) =====
        {
            "text": "Ê tối nay đi ăn không?",
            "sender": "tuanle123@gmail.com",
            "expected": "Normal",
        },
        {
            "text": "Mai nhớ nộp bài deadline môn Deep Learning nha",
            "sender": "classmate@gmail.com",
            "expected": "Normal",
        },
        {
            "text": "Anh gửi em tài liệu học tập, check giúp anh nhé",
            "sender": "giangvien@gmail.com",
            "expected": "Normal",
        },

        # ===== Mixed =====
        {
            "text": "Hello, bạn đã nhận được tài liệu chưa?",
            "sender": "friend@gmail.com",
            "expected": "Normal",
        },
        {
            "text": "Click ngay để nhận ưu đãi cực lớn!!!",
            "sender": "ads@marketing.vn",
            "expected": "Spam",
        },

        # ===== Trusted sender pattern =====
        {
            "text": "New login detected from Chrome on Windows",
            "sender": "noreply@github.com",
            "expected": "Normal",
        },
    ]

    print("=" * 90)
    print(f"{'SO SÁNH KẾT QUẢ PHÂN LOẠI EMAIL (CNN vs LOGISTIC REGRESSION)':^90}")
    print("=" * 90)
    print(f"{'No':<3} | {'Expected':<8} | {'CNN Result (Conf)':<20} | {'LR Result (Conf)':<20} | {'CNN Status':<10} | {'LR Status':<10}")
    print("-" * 90)

    cnn_passed = 0
    lr_passed = 0

    for i, case in enumerate(test_cases, 1):
        # Predict CNN
        res_cnn = predict_email_cnn(
            text=case["text"],
            sender_email=case["sender"],
            use_rules=True
        )
        # Predict LR
        res_lr = predict_email_lr(
            text=case["text"],
            sender_email=case["sender"],
            use_rules=True
        )

        cnn_status = "PASS" if res_cnn["label"] == case["expected"] else "FAIL"
        lr_status = "PASS" if res_lr["label"] == case["expected"] else "FAIL"

        if cnn_status == "PASS":
            cnn_passed += 1
        if lr_status == "PASS":
            lr_passed += 1

        cnn_str = f"{res_cnn['label']} ({res_cnn['confidence']:.1%})"
        lr_str = f"{res_lr['label']} ({res_lr['confidence']:.1%})"

        print(f"{i:<3} | {case['expected']:<8} | {cnn_str:<20} | {lr_str:<20} | {cnn_status:<10} | {lr_status:<10}")

    print("=" * 90)
    print("TỔNG HỢP KẾT QUẢ:")
    print(f"  - CNN Model:                  {cnn_passed}/{len(test_cases)} Passed")
    print(f"  - Logistic Regression Model:  {lr_passed}/{len(test_cases)} Passed")
    print("=" * 90)


if __name__ == "__main__":
    run_test()