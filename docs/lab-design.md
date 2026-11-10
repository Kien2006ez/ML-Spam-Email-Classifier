# Thiết kế bài Lab

## Đặt vấn đề

Email rác (spam) chiếm một phần lớn thư gửi đến, gây phiền toái và có thể chứa link lừa đảo. Việc tự động lọc spam bằng học máy là một ứng dụng quen thuộc của bài toán phân loại văn bản. Bài Lab 1 yêu cầu thực hiện đầy đủ quy trình này, từ dữ liệu đến sản phẩm demo.

## Mục tiêu

1. Phân loại mỗi email thành **spam** hoặc **không spam (ham)**.
2. Tự cài đặt và so sánh 3 thuật toán: Logistic Regression, SVM, Naive Bayes.
3. Đánh giá bằng Accuracy, Precision, Recall, F1 trên tập test.
4. Tìm hiểu thêm các hướng cải thiện: đặc trưng, tuning siêu tham số, ensemble.
5. Triển khai giao diện web để phân loại email mới.

## Đầu vào / Đầu ra

**Bài toán**
- **Đầu vào**: nội dung một email (tiêu đề + nội dung, văn bản tiếng Anh).
- **Đầu ra**: nhãn `spam` hoặc `ham` (không spam) kèm xác suất spam từ 0 đến 1; từ 0,5 trở lên là spam.

**Toàn bộ bài lab**
- **Đầu vào**: file `data/raw_data/enron_spam_data.csv` (cột Subject, Message, Spam/Ham) và email người dùng nhập trên giao diện.
- **Đầu ra**: mô hình `.joblib`, bảng chỉ số (`metrics.csv`), báo cáo `data.html`, kết quả dự đoán trên giao diện web.

## Dữ liệu

Enron-Spam, bản CSV của M. Wiechmann (`github.com/MWiechmann/enron_spam_data`), tổng hợp từ bộ gốc của nhóm nghiên cứu Đại học Kinh tế Athens (`www2.aueb.gr/users/ion/data/enron-spam/`). Dữ liệu tự tải về `data/raw_data/`.

- **File gốc**: 33.716 email, 5 cột: `Message ID`, `Subject`, `Message`, `Spam/Ham`, `Date`. Có 17.171 spam (50,9%) và 16.545 ham (49,1%).
- **Giá trị thiếu**: 289 `Subject` và 371 `Message`, điền chuỗi rỗng; 51 email rỗng cả hai phần bị loại.
- **Sau khi bỏ trùng**: 30.462 email (14.552 spam, 15.910 ham).
- **Chia dữ liệu**: 80/20 phân tầng, `random_state = 42`: 24.369 email train, 6.093 email test.

## Quy trình

1. Thu thập dữ liệu: `data/scripts/data_collection.py`
2. Tiền xử lý, chia train/test 80/20: `data/scripts/data_processing.py`
3. Đặc trưng và 3 thuật toán tự cài đặt (mỗi thuật toán một file): `model/algorithm/` gồm `logistic_regression.py`, `svm_feature_engineer.py`, `naive_bayes.py`
4. Huấn luyện: `model/script/training.py` (ghi log vào `reports/eval-result/training_log.md`)
5. Đánh giá: `model/script/eval.py`
6. Thí nghiệm mở rộng: `model/script/tuning.py`
7. Giao diện web: `demo/app.py`

## Cấu trúc thư mục

```
BT Lab/
├── docs/
│   ├── lab-design.md             # Tài liệu thiết kế bài Lab (file này)
│   └── algorithm.md              # Tài liệu giải thích thuật toán
├── data/
│   ├── raw_data/                 # Dữ liệu thô (enron_spam_data.csv, tải tự động)
│   ├── scripts/                  # Các script thu thập và xử lý dữ liệu
│   │   ├── data_collection.py    # Tải dữ liệu
│   │   ├── data_processing.py    # Làm sạch văn bản, chia train/test 80/20
│   │   ├── eva.py                # Thống kê dữ liệu (số spam/ham, độ dài)
│   │   └── demol.py              # In thử email gốc và bản đã làm sạch
│   └── save_model/               # Bản sao lưu mô hình
├── model/
│   ├── config.py                 # Đường dẫn và hằng số dùng chung
│   ├── algorithm/                # Các thuật toán cốt lõi (tự cài đặt)
│   │   ├── logistic_regression.py   # Logistic Regression
│   │   ├── svm_feature_engineer.py  # SVM và trích xuất đặc trưng
│   │   └── naive_bayes.py           # Naive Bayes
│   ├── script/                   # Các script điều khiển huấn luyện và chạy mô hình
│   │   ├── data_preparation.py   # Chuẩn bị dữ liệu cho mô hình
│   │   ├── training.py           # Huấn luyện 3 mô hình, ghi log
│   │   ├── eval.py               # Đánh giá, dự đoán email mới, xuất báo cáo
│   │   ├── tuning.py             # Tuning, ensemble, so sánh đặc trưng
│   │   └── demol.py              # Chạy thử dự đoán email mẫu
│   └── save_models/              # Mô hình đã huấn luyện (.joblib)
├── reports/
│   ├── data.html                 # Báo cáo dữ liệu, biểu đồ Accuracy, Confusion Matrix SVM
│   ├── eval-result/              # metrics, extra_experiments, best_params, training_log.md
│   └── demo/                     # Kết quả chạy demo (demo_results.csv)
├── evaluation/                   # summary.md, extra_experiments.md
├── demo/
│   └── app.py                    # Giao diện demo (Streamlit)
├── readme.md                     # Hướng dẫn tổng quan dự án
├── requirements.txt              # Danh sách thư viện
└── todo.md                       # File ra lệnh cho AI Agent
```

Ngoài cấu trúc ban đầu, dự án có thêm `model/config.py`, `model/script/tuning.py`, `requirements.txt` và các file `__init__.py` (để chạy được `python -m ...`).

## Chạy

```
pip install -r requirements.txt
python -m model.script.training
streamlit run demo/app.py
```

## Hạn chế

- Dữ liệu là email tiếng Anh nên không phù hợp với tiếng Việt.
- Ham chỉ nằm trong 1999–2002 còn gần như toàn bộ spam ở 2002–2005 (theo cột `Date`), nên mô hình có thể học một phần dấu hiệu thời đại.
- Email thông báo hiện đại (ngân hàng, ứng dụng) có thể bị nhận nhầm là spam.
- Độ chính xác ~99% có thể cao hơn thực tế vì bộ Enron-Spam khá dễ tách.

Chi tiết công thức và cách hoạt động của từng thuật toán: xem `docs/algorithm.md`.
