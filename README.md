# Lab 1: Phân loại Email Spam bằng học máy

Bài thực hành phân loại email thành **spam** (email rác) hoặc **ham** (email thường) bằng ba thuật toán tự cài đặt: Logistic Regression, SVM và Naive Bayes, với đặc trưng TF-IDF và tần suất ký tự.

- **Đầu vào**: nội dung email (tiêu đề + nội dung, tiếng Anh).
- **Đầu ra**: nhãn `spam` / `ham` kèm xác suất spam từ 0 đến 1.
- **Dữ liệu**: Enron-Spam, 33.716 email có nhãn, tự tải về khi huấn luyện ([nguồn](https://github.com/MWiechmann/enron_spam_data)).

## Cách chạy

```powershell
pip install -r requirements.txt
python -m model.script.training   # tải dữ liệu, huấn luyện, đánh giá, ghi log
python -m model.script.tuning     # (tuỳ chọn) tuning, ensemble, so sánh đặc trưng
streamlit run demo/app.py         # mở giao diện demo
```

Chạy các lệnh từ thư mục gốc của dự án.

## Cấu trúc

```
data/raw_data/        dữ liệu thô
data/scripts/         thu thập, tiền xử lý, thống kê dữ liệu
data/save_model/      bản sao lưu mô hình
model/algorithm/      3 thuật toán (mỗi thuật toán một file) và đặc trưng
model/script/         chuẩn bị dữ liệu, huấn luyện, đánh giá, tuning, demo
model/save_models/    mô hình đã huấn luyện
reports/              data.html, eval-result/ (kèm training_log.md), demo/
evaluation/           bảng so sánh mô hình
demo/app.py           giao diện demo (Streamlit)
docs/                 thiết kế bài lab, giải thích thuật toán
```

## Tài liệu

- `docs/lab-design.md`: đặt vấn đề, mục tiêu, dữ liệu, quy trình, cấu trúc thư mục.
- `docs/algorithm.md`: công thức và cách hoạt động của từng thuật toán.

## Kết quả

Xem `evaluation/summary.md` (3 mô hình) và `evaluation/extra_experiments.md` (thí nghiệm mở rộng).

Lưu ý: mô hình học từ email tiếng Anh (công việc và spam thập niên 2000) nên không phù hợp với email tiếng Việt, và email thông báo hiện đại (ngân hàng, ứng dụng) có thể bị nhận nhầm là spam.