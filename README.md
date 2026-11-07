# Phân loại Email Spam (Logistic Regression · SVM · Naive Bayes)

Bài thực hành: phân loại email thành **spam** / **không spam (ham)**.



## Cách chạy
```powershell
pip install -r requirements.txt
python -m model.script.training   # tải dữ liệu, huấn luyện, đánh giá, ghi log
python -m model.script.tuning     # (tuỳ chọn) tuning, ensemble, so sánh đặc trưng
streamlit run demo/app.py         # mở giao diện demo
```

Các lệnh khác (chạy từ thư mục gốc dự án):

| Lệnh | Tác dụng |
|---|---|
| `python -m data.scripts.data_collection` | chỉ tải dữ liệu |
| `python -m data.scripts.eva` | thống kê dữ liệu |
| `python -m data.scripts.demol` | xem email gốc và bản đã làm sạch |
| `python -m model.script.eval` | đánh giá lại, ghi báo cáo |
| `python -m model.script.demol` | dự đoán email mẫu bằng 3 mô hình |

## Cấu trúc
data/raw_data/        dữ liệu thô
data/scripts/         thu thập, tiền xử lý, thống kê dữ liệu
data/save_model/      bản sao lưu mô hình
model/algorithm/      đặc trưng và bộ phân loại
model/script/         chuẩn bị dữ liệu, huấn luyện, đánh giá, tuning, demo
model/save_models/    mô hình đã huấn luyện
reports/              data.html, eval-result/, demo/
evaluation/           bảng so sánh mô hình
demo/app.py           giao diện Streamlit
docs/                 thiết kế bài lab, giải thích thuật toán
