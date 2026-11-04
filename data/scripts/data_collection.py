"""Thu thập dữ liệu: tải bộ Enron-Spam về data/raw_data.

Chạy:  python -m data.scripts.data_collection
"""

import urllib.request
import zipfile

from model.config import DATASET_URL, RAW_CSV, RAW_DIR


def download_dataset(force: bool = False) -> None:
    """Tải và giải nén dataset nếu chưa có."""
    if RAW_CSV.exists() and not force:
        print(f"Đã có dữ liệu: {RAW_CSV}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "enron_spam_data.zip"

    print(f"Đang tải {DATASET_URL} ... (khoảng 15 MB)")
    urllib.request.urlretrieve(DATASET_URL, zip_path)

    print("Đang giải nén ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Giải nén xong nhưng không thấy {RAW_CSV.name} trong {RAW_DIR}."
        )
    print(f"Xong: {RAW_CSV}")


if __name__ == "__main__":
    download_dataset()
