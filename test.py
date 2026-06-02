"""
Test cases cho hệ thống phân loại email.
Chạy: python test.py
"""
from model.predict_cnn import predict_email


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

    print("=" * 70)
    print("EMAIL CLASSIFICATION TEST")
    print("=" * 70)

    passed = 0
    failed = 0

    for i, case in enumerate(test_cases, 1):
        result = predict_email(
            text=case["text"],
            sender_email=case["sender"],
            use_rules=True
        )

        status = "PASS" if result["label"] == case["expected"] else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"\n--- Test {i} [{status}] ---")
        print(f"  Text:       {case['text'][:60]}...")
        print(f"  Sender:     {case['sender']}")
        print(f"  Expected:   {case['expected']}")
        print(f"  Got:        {result['label']} ({result['confidence']:.1%})")
        print(f"  Method:     {result['method']}")
        if result.get('details'):
            print(f"  Details:    {result['details'][:80]}")

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed} passed, {failed} failed, {len(test_cases)} total")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_test()