"""
Bộ từ khóa và rules phát hiện spam/phishing cho email tại Việt Nam.
Phân chia theo nhóm để dễ quản lý và cập nhật.

Cấu trúc:
- SPAM_KEYWORD_GROUPS: Từ khóa spam phân theo nhóm (ngân hàng, tiền bạc, ...)
- TRUSTED_DOMAINS: Domain đáng tin cậy (whitelist)
- SUSPICIOUS_DOMAIN_PATTERNS: Pattern domain đáng ngờ
- SUSPICIOUS_SENDER_PATTERNS: Pattern tên email đáng ngờ
- TRUSTED_SENDER_PATTERNS: Pattern email hợp lệ (GitHub, Google, ...)
"""

# ==============================================================================
# NHÓM 1: GIẢ MẠO NGÂN HÀNG (Bank Phishing)
# ==============================================================================
BANK_PHISHING_KEYWORDS = [
    # Hành động giả mạo
    "xác minh tài khoản ngân hàng", "cập nhật thông tin ngân hàng",
    "tài khoản bị khóa", "tài khoản bị tạm ngưng", "tài khoản bị đóng băng",
    "xác thực danh tính ngân hàng", "đăng nhập lại tài khoản ngân hàng",
    "xác nhận giao dịch đáng ngờ", "phát hiện hoạt động bất thường",
    "tài khoản có dấu hiệu bị xâm nhập", "cảnh báo bảo mật tài khoản",
    "khôi phục tài khoản ngân hàng", "mở khóa tài khoản",
    "cập nhật mật khẩu ngân hàng", "thay đổi pin thẻ", "thẻ tín dụng bị khóa",

    # Tên ngân hàng phổ biến (dùng trong context giả mạo)
    "vietcombank thông báo khẩn", "techcombank cảnh báo",
    "mbbank xác minh", "bidv yêu cầu", "agribank thông báo",
    "tpbank cập nhật", "vpbank bảo mật", "acb cảnh báo",
    "sacombank xác nhận", "hdbank yêu cầu", "shinhan bank thông báo",
    "vib cảnh báo", "oceanbank thông báo", "pvcombank xác minh",
    "lienvietpostbank cập nhật", "scb thông báo",

    # Giả mạo ví điện tử
    "momo xác minh tài khoản", "zalopay cảnh báo",
    "vnpay yêu cầu xác nhận", "shopeepay xác minh",
]

# ==============================================================================
# NHÓM 2: TIỀN BẠC / KHUYẾN MÃI (Money / Promotion Scam)
# ==============================================================================
MONEY_PROMO_KEYWORDS = [
    # Trúng thưởng
    "chúc mừng bạn đã trúng thưởng", "bạn đã trúng giải",
    "trúng giải đặc biệt", "trúng thưởng xổ số",
    "bạn là người may mắn", "giải thưởng trị giá",
    "nhận giải thưởng", "bạn được chọn ngẫu nhiên",

    # Khuyến mãi giả
    "giảm giá 90%", "giảm giá 80%", "giảm giá sốc",
    "ưu đãi sốc", "khuyến mãi khủng", "sale lớn nhất năm",
    "miễn phí hoàn toàn", "nhận quà miễn phí",
    "click nhận quà", "nhận ngay", "đăng ký nhận",
    "ưu đãi độc quyền", "ưu đãi đặc biệt dành riêng",
    "hoàn tiền ngay", "hoàn tiền 100%",
    "mua 1 tặng 10", "voucher giảm giá",

    # Tiền bạc
    "nhận tiền mặt", "chuyển tiền cho bạn",
    "rút tiền ngay", "nhận thưởng tiền mặt",
    "kiếm tiền dễ dàng", "thu nhập hàng ngày",
]

# ==============================================================================
# NHÓM 3: GIẢ MẠO DỊCH VỤ (Service Impersonation)
# ==============================================================================
SERVICE_PHISHING_KEYWORDS = [
    # Fake urgency
    "tài khoản sẽ bị xóa", "tài khoản sẽ bị khóa vĩnh viễn",
    "hết hạn trong 24 giờ", "hết hạn trong 24h",
    "xác nhận ngay kẻo mất", "còn 24 giờ để xác nhận",
    "hành động ngay", "yêu cầu khẩn cấp",
    "cần xác minh ngay lập tức", "không xác minh sẽ mất tài khoản",

    # Fake links
    "click vào link để xác minh", "nhấn vào đây để xác nhận",
    "click link bên dưới", "truy cập link sau",
    "đăng nhập lại ngay", "link xác minh",
    "xác nhận qua đường link",

    # Fake security
    "phát hiện virus", "máy tính bị nhiễm",
    "cài đặt phần mềm bảo mật", "tải app bảo mật",
    "cập nhật bảo mật khẩn cấp",
]

# ==============================================================================
# NHÓM 4: ĐẦU TƯ / LỪA ĐẢO TÀI CHÍNH (Investment Scam)
# ==============================================================================
INVESTMENT_SCAM_KEYWORDS = [
    "lợi nhuận 300%", "lợi nhuận 200%", "lợi nhuận 500%",
    "đầu tư bitcoin", "đầu tư crypto", "đầu tư tiền điện tử",
    "kiếm tiền online", "kiếm tiền tại nhà",
    "thu nhập thụ động", "thu nhập không giới hạn",
    "cơ hội vàng", "cơ hội đầu tư",
    "đầu tư forex", "sàn giao dịch",
    "lợi nhuận cam kết", "đảm bảo sinh lời",
    "nhân đôi tài khoản", "nhân ba tài khoản",
    "đầu tư chứng khoán dễ dàng",
    "tham gia ngay hôm nay", "đăng ký đầu tư",
    "đào coin", "mining bitcoin",
]

# ==============================================================================
# NHÓM 5: VIỆC LÀM GIẢ (Fake Job Scam)
# ==============================================================================
FAKE_JOB_KEYWORDS = [
    "tuyển dụng làm việc tại nhà", "việc nhẹ lương cao",
    "tuyển cộng tác viên online", "thu nhập 50 triệu",
    "không cần kinh nghiệm lương cao", "tuyển gấp",
    "làm việc 2-3 giờ mỗi ngày", "lương từ 20 triệu",
    "tuyển nhân viên part-time online",
    "việc làm online thu nhập cao",
    "đăng ký làm cộng tác viên",
]

# ==============================================================================
# NHÓM 6: THÔNG BÁO GIẢ (Fake Notification)
# ==============================================================================
FAKE_NOTIFICATION_KEYWORDS = [
    "bạn có bưu phẩm chưa nhận", "đơn hàng cần xác nhận thanh toán",
    "giao hàng thất bại cần xác nhận", "xác nhận nhận hàng",
    "bạn có tin nhắn chưa đọc", "ai đó đã gửi tin nhắn cho bạn",
    "bạn có yêu cầu kết bạn", "ai đó đã xem hồ sơ của bạn",
    "thông báo trúng thưởng từ shopee", "lazada thông báo trúng thưởng",
]

# ==============================================================================
# Tổng hợp tất cả nhóm keywords
# ==============================================================================
SPAM_KEYWORD_GROUPS = {
    "bank_phishing": {
        "name": "Giả mạo ngân hàng",
        "keywords": BANK_PHISHING_KEYWORDS,
        "weight": 2.0,  # Trọng số cao — nguy hiểm nhất
    },
    "money_promo": {
        "name": "Tiền bạc / Khuyến mãi lừa đảo",
        "keywords": MONEY_PROMO_KEYWORDS,
        "weight": 1.5,
    },
    "service_phishing": {
        "name": "Giả mạo dịch vụ",
        "keywords": SERVICE_PHISHING_KEYWORDS,
        "weight": 1.8,
    },
    "investment_scam": {
        "name": "Đầu tư / Lừa đảo tài chính",
        "keywords": INVESTMENT_SCAM_KEYWORDS,
        "weight": 1.7,
    },
    "fake_job": {
        "name": "Việc làm giả",
        "keywords": FAKE_JOB_KEYWORDS,
        "weight": 1.3,
    },
    "fake_notification": {
        "name": "Thông báo giả mạo",
        "keywords": FAKE_NOTIFICATION_KEYWORDS,
        "weight": 1.2,
    },
}

# ==============================================================================
# DOMAIN ĐÁNG TIN CẬY (Whitelist)
# Các email từ domain này sẽ tự động được coi là Normal
# ==============================================================================
TRUSTED_DOMAINS = [
    # Google services
    "google.com", "gmail.com", "classroom.google.com",
    "googlemail.com", "youtube.com", "accounts.google.com",

    # Microsoft
    "microsoft.com", "outlook.com", "hotmail.com",
    "live.com", "office365.com", "teams.microsoft.com",

    # Development / Tech
    "github.com", "gitlab.com", "bitbucket.org",
    "stackoverflow.com", "npmjs.com", "docker.com",
    "vercel.com", "netlify.com", "heroku.com",
    "aws.amazon.com", "azure.com", "cloud.google.com",

    # Social Media
    "facebook.com", "facebookmail.com", "instagram.com",
    "twitter.com", "x.com", "linkedin.com", "tiktok.com",
    "discord.com", "discordapp.com", "telegram.org",

    # Apple
    "apple.com", "icloud.com",

    # E-commerce VN (chính thức)
    "shopee.vn", "shopee.com", "tiki.vn",
    "lazada.vn", "lazada.com", "sendo.vn",

    # Ngân hàng VN (domain chính thức)
    "vietcombank.com.vn", "techcombank.com.vn", "mbbank.com.vn",
    "bidv.com.vn", "agribank.com.vn", "tpbank.com.vn",
    "vpbank.com.vn", "acb.com.vn", "sacombank.com.vn",
    "hdbank.com.vn", "vib.com.vn",

    # Dịch vụ VN
    "grab.com", "shopee.co.id", "vnpay.vn",
    "momo.vn", "zalopay.vn", "viettel.com.vn",
    "vnpt.com.vn", "fpt.com.vn", "mobifone.com.vn",

    # Education
    "edu.vn", "edu.com", "coursera.org",
    "udemy.com", "edx.org",

    # Other trusted
    "paypal.com", "stripe.com", "notion.so",
    "slack.com", "zoom.us", "canva.com",
    "figma.com", "spotify.com", "netflix.com",
]

# ==============================================================================
# PATTERN DOMAIN ĐÁNG NGỜ
# Domain sử dụng TLD rẻ tiền, hoặc pattern giả mạo
# ==============================================================================
SUSPICIOUS_DOMAIN_PATTERNS = [
    # TLD rẻ tiền thường bị lạm dụng
    r".*\.xyz$",
    r".*\.top$",
    r".*\.buzz$",
    r".*\.click$",
    r".*\.loan$",
    r".*\.work$",
    r".*\.gq$",
    r".*\.ml$",
    r".*\.cf$",
    r".*\.tk$",
    r".*\.ga$",
    r".*\.icu$",
    r".*\.cam$",
    r".*\.rest$",
    r".*\.monster$",
    r".*\.casa$",
    r".*\.surf$",

    # Pattern giả mạo domain
    r".*-verify\..*",        # fake-bank-verify.com
    r".*-secure\..*",        # secure-login-bank.com
    r".*-confirm\..*",       # confirm-account.com
    r".*-update\..*",        # update-info.com
    r".*-login\..*",         # login-verify.com
    r".*-support\..*",       # support-help.com
    r".*banking-.*",         # banking-verify.xyz
    r".*account-.*",         # account-recovery.top

    # Subdomain giả mạo ngân hàng
    r"vietcombank\..*\..*",  # vietcombank.fake.com
    r"techcombank\..*\..*",
    r"mbbank\..*\..*",
    r"bidv\..*\..*",
    r".*\.vietcombank\.(?!com\.vn).*",  # fake.vietcombank.xyz
]

# ==============================================================================
# PATTERN TÊN EMAIL ĐÁNG NGỜ
# ==============================================================================
SUSPICIOUS_SENDER_PATTERNS = [
    r"^[a-z]{18,}@",                    # Tên toàn chữ thường dài bất thường
    r"^[a-z0-9]{22,}@",                 # Mix chữ + số quá dài (random)
    r"^[0-9]{8,}@",                     # Toàn số dài
    r".*noreply.*bank(?!\.com\.vn).*",   # noreply giả mạo bank
    r".*admin.*verify.*",               # admin-verify giả
    r".*support.*confirm.*",            # support-confirm giả
    r".*security.*alert.*",             # security-alert giả
    r".*info\d{3,}@.*",                 # info12345@... (bulk sender)
    r".*promo\d+@.*",                   # promo123@...
    r".*offer\d+@.*",                   # offer456@...
    r".*winner\d*@.*",                  # winner@... / winner123@...
    r".*prize\d*@.*",                   # prize@...
    r".*lucky\d*@.*",                   # lucky@...
]

# ==============================================================================
# PATTERN EMAIL HỢP LỆ (sẽ ưu tiên Normal)
# ==============================================================================
TRUSTED_SENDER_PATTERNS = [
    r".*@.*\.google\.com$",             # noreply@classroom.google.com
    r".*noreply@github\.com$",          # GitHub notifications
    r".*notifications@github\.com$",
    r".*@.*\.microsoft\.com$",          # Microsoft services
    r".*@.*\.facebook\.com$",
    r".*@.*\.facebookmail\.com$",
    r".*@.*\.shopee\.(vn|com)$",
    r".*@.*\.tiki\.vn$",
    r".*@.*\.edu\.vn$",                 # Education VN
    r".*@.*\.gov\.vn$",                 # Government VN
]


def get_all_spam_keywords():
    """Trả về list tất cả spam keywords (flat list)."""
    all_keywords = []
    for group in SPAM_KEYWORD_GROUPS.values():
        all_keywords.extend(group["keywords"])
    return all_keywords


def get_keyword_groups_summary():
    """Trả về summary các nhóm keywords để hiển thị."""
    summary = {}
    for group_id, group in SPAM_KEYWORD_GROUPS.items():
        summary[group_id] = {
            "name": group["name"],
            "count": len(group["keywords"]),
            "weight": group["weight"],
            "sample": group["keywords"][:5],  # 5 keywords mẫu
        }
    return summary
