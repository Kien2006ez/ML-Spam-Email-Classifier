"""Đường dẫn và hằng số dùng chung cho toàn bộ dự án."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw_data"
BACKUP_MODEL_DIR = PROJECT_ROOT / "data" / "save_model"
MODEL_DIR = PROJECT_ROOT / "model" / "save_models"
REPORT_DIR = PROJECT_ROOT / "reports"
EVAL_RESULT_DIR = REPORT_DIR / "eval-result"
DEMO_RESULT_DIR = REPORT_DIR / "demo"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Bộ dữ liệu Enron-Spam (~33.700 email thật, đã gán nhãn spam/ham)
DATASET_URL = (
    "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/"
    "master/enron_spam_data.zip"
)
RAW_CSV = RAW_DIR / "enron_spam_data.csv"

# Email có ít hơn số từ/cụm từ này nằm trong từ vựng thì mô hình không đủ thông tin để kết luận
MIN_KNOWN_FEATURES = 5
# Quá tỉ lệ chữ cái ngoài bảng ASCII (có dấu tiếng Việt...) này thì coi là không phải tiếng Anh
MAX_NON_ASCII_RATIO = 0.05

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_NAMES = ["logistic_regression", "svm", "naive_bayes"]
MODEL_TITLES = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM (LinearSVC)",
    "naive_bayes": "Naive Bayes",
}

SAMPLE_EMAILS = {
    "spam": (
        "Subject: URGENT!!! You have WON $1,000,000\n\n"
        "Congratulations! You have been selected to receive a FREE cash prize. "
        "Click here http://win-money-now.biz to claim your $1,000,000 now. "
        "Limited time offer, act now!!! Cheap viagra and online pharmacy discount."
    ),
    "ham": (
        "Subject: Meeting tomorrow\n\n"
        "Hi John, just a reminder that our project meeting is scheduled for "
        "tomorrow at 10am in the conference room. Please send me the updated "
        "report before then so we can review the numbers together. Thanks, Mary"
    ),
}
