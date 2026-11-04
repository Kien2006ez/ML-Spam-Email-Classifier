"""Đánh giá dữ liệu: thống kê nhanh về bộ dataset.

Chạy:  python -m data.scripts.eva
"""

import pandas as pd

from data.scripts.data_processing import load_dataset


def dataset_summary(df: pd.DataFrame) -> dict:
    """Trả về số lượng email, tỉ lệ spam và độ dài trung bình (số từ)."""
    words = df["text"].str.split().str.len()
    return {
        "total": int(len(df)),
        "spam": int(df["label"].sum()),
        "ham": int((df["label"] == 0).sum()),
        "spam_ratio": float(df["label"].mean()),
        "avg_words_spam": float(words[df["label"] == 1].mean()),
        "avg_words_ham": float(words[df["label"] == 0].mean()),
    }


if __name__ == "__main__":
    for key, value in dataset_summary(load_dataset()).items():
        print(f"{key:>15}: {value:.3f}" if isinstance(value, float) else f"{key:>15}: {value}")
