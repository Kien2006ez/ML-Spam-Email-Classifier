"""Tiền xử lý dữ liệu: đọc dataset, làm sạch văn bản, chia train/test."""

import html
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from model.config import RANDOM_STATE, RAW_CSV, TEST_SIZE

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_ADDR = re.compile(r"\S+@\S+")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Bỏ thẻ HTML, URL, địa chỉ email, dấu câu, chữ số; đưa về chữ thường.

    Stop words được loại ở bước TF-IDF (stop_words="english").
    """
    text = html.unescape(str(text)).lower()
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" urltoken ", text)
    text = _EMAIL_ADDR.sub(" emailtoken ", text)
    text = _NON_ALPHA.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def load_dataset() -> pd.DataFrame:
    """Đọc CSV gốc, trả về DataFrame 2 cột: text (subject + body), label (1 = spam)."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Không thấy dữ liệu tại {RAW_CSV}. "
            "Hãy chạy `python -m data.scripts.data_collection` để tải về."
        )

    df = pd.read_csv(RAW_CSV)
    subject = df["Subject"].fillna("")
    message = df["Message"].fillna("")

    out = pd.DataFrame(
        {
            "text": (subject + "\n" + message).str.strip(),
            "label": (df["Spam/Ham"].str.lower() == "spam").astype(int),
        }
    )
    out = out[out["text"].str.len() > 0]
    return out.drop_duplicates(subset="text").reset_index(drop=True)


def get_train_test_split():
    """Chia 80/20, giữ tỉ lệ spam/ham; random_state cố định để train và eval khớp nhau."""
    df = load_dataset()
    return train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )


if __name__ == "__main__":
    data = load_dataset()
    print(f"Số email: {len(data)}")
