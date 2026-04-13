import random
import pandas as pd

spam_templates = [
    "Chúc mừng bạn đã trúng thưởng {amount} VNĐ",
    "Nhận quà miễn phí ngay hôm nay",
    "Tài khoản của bạn cần xác minh gấp",
    "Bạn được chọn nhận ưu đãi đặc biệt",
    "Click vào link để nhận thưởng ngay",
    "Khuyến mãi sốc giảm giá 90%",
    "Bạn đã trúng giải đặc biệt",
    "Cảnh báo bảo mật, vui lòng xác nhận tài khoản",
    "Nhận tiền hoàn ngay hôm nay",
    "Ưu đãi độc quyền dành riêng cho bạn"
]

normal_templates = [
    "Đơn hàng Shopee đã giao thành công",
    "Mã OTP của bạn là {code}",
    "Thông báo giao dịch thành công",
    "Xác nhận đăng ký tài khoản",
    "Lịch học ngày mai đã được cập nhật",
    "Thông báo từ GitHub về bảo mật",
    "Google cảnh báo đăng nhập mới",
    "Hóa đơn điện tử của bạn",
    "Cập nhật trạng thái đơn hàng",
    "Thông báo từ ngân hàng"
]

def generate_spam():
    text = random.choice(spam_templates)
    return text.format(
        amount=random.randint(1000000, 1000000000)
    )

def generate_normal():
    text = random.choice(normal_templates)
    return text.format(
        code=random.randint(100000, 999999)
    )

data = []

# 250 spam
for _ in range(250):
    data.append((generate_spam(), 1))

# 250 normal
for _ in range(250):
    data.append((generate_normal(), 0))

# shuffle
random.shuffle(data)

# save CSV
df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("dataset_vn.csv", index=False, encoding="utf-8")

print("Đã tạo dataset_vn.csv với 500 dòng!")