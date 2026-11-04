"""Chạy thử dữ liệu: in vài email gốc cùng bản đã làm sạch.

Chạy:  python -m data.scripts.demol
"""

from data.scripts.data_processing import clean_text, load_dataset


def main(n: int = 3) -> None:
    df = load_dataset()
    for label in (1, 0):
        name = "SPAM" if label else "HAM"
        for text in df[df["label"] == label]["text"].head(n):
            print(f"--- [{name}] gốc ---\n{text[:300]}")
            print(f"--- [{name}] sau làm sạch ---\n{clean_text(text)[:300]}\n")


if __name__ == "__main__":
    main()
