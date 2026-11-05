# Dự án: Phân loại Email Spam (Spam Email Classification)

## 🛠 Công nghệ sử dụng
- Ngôn ngữ: Python 3.10+
- Thư viện Core: `pandas`, `numpy`, `scikit-learn`, `nltk` (hoặc `spacy`)
- Phân tích dữ liệu: `matplotlib`, `seaborn`

## 📂 Cấu trúc thư mục mong muốn
```text
BT Lab/
├── docs/
│   ├── lab-design.md             # Tài liệu thiết kế bài Lab 
│   └── algorithm.md              # Tài liệu giải thích thuật toán 
├── data/
│   ├── __init__.py               # Đánh dấu thư mục là package Python 
│   ├── raw_data/                 # Dữ liệu thô
│   │   └── enron_spam_data.csv   # Bộ Enron-Spam, tải về khi huấn luyện 
│   ├── scripts/                  # Các script thu thập và xử lý dữ liệu
│   │   ├── __init__.py           # 
│   │   ├── data_collection.py    # Tải dataset về data/raw_data
│   │   ├── data_processing.py    # Làm sạch văn bản, chia train/test 80/20
│   │   ├── eva.py                # Thống kê dữ liệu (số spam/ham, độ dài)
│   │   └── demol.py              # In thử email gốc và bản đã làm sạch
│   └── save_model/               # Bản sao lưu mô hình 
├── model/
│   ├── __init__.py               # 
│   ├── config.py                 # Đường dẫn và hằng số dùng chung 
│   ├── algorithm/                # Các thuật toán cốt lõi
│   │   ├── __init__.py           # 
│   │   └── svm_feature_engineer.py # TF-IDF + đặc trưng ký tự, 3 bộ phân loại 
│   ├── script/                   # Các script điều khiển huấn luyện và chạy mô hình
│   │   ├── __init__.py           # Dùng để chạy các lệnh python -m 
│   │   ├── data_preparation.py   # Chuẩn bị dữ liệu cho mô hình
│   │   ├── training.py           # Huấn luyện 3 mô hình, lưu .joblib
│   │   ├── eval.py               # Đánh giá, predict_new_email, xuất báo cáo
│   │   ├── tuning.py             # Tuning, ensemble, so sánh đặc trưng 
│   │   └── demol.py              # Chạy thử dự đoán email mẫu
│   └── save_models/              # Mô hình đã huấn luyện (svm, logistic_regression, naive_bayes) 
├── reports/
│   ├── data.html                 # Báo cáo dữ liệu, biểu đồ Accuracy, Confusion Matrix SVM 
│   ├── eval-result/              # metrics.csv, metrics.json, extra_experiments.csv, best_params.json 
│   └── demo/                     # demo_results.csv 
├── evaluation/                   # summary.md, extra_experiments.md 
├── demo/
│   └── app.py                    # Giao diện web Streamlit 
├── readme.md                     # Hướng dẫn tổng quan dự án 
├── requirements.txt              # Danh sách thư viện 
└── todo.md                       # File ra lệnh cho AI Agent


#🎯 Nhiệm vụ chi tiết 

## Bước 1: Khởi tạo tệp cấu hình môi trường
- [x] Tạo file `requirements.txt` chứa các thư viện: `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`.

### Bước 2: Tạo cấu trúc thư mục trống
- [x] Tự động tạo toàn bộ các thư mục: `data/raw`, `data/processed`, `src`, `notebooks`, `models`.

### Bước 3: Phát triển mã nguồn Tiền xử lý dữ liệu (Workflow 1)
- [x] Tạo file `src/preprocessing.py`: Viết các hàm làm sạch văn bản (loại bỏ stop words, dấu câu, HTML tags). Tạo code mock-up giả lập một tập dữ liệu email nhỏ (gồm văn bản và nhãn spam/not spam) để chạy test.

### Bước 4: Phát triển Trích xuất đặc trưng & Chia dữ liệu (Workflow 1)
- [x] Tạo file `src/feature_engineering.py`: Sử dụng `TfidfVectorizer` từ `sklearn` để chuyển dữ liệu chữ sang dạng số. Thực hiện chia tập dữ liệu thành `train` và `test` theo tỷ lệ 80/20.

### Bước 5: Xây dựng các mô hình Phân loại (Workflow 2)
- [x] Tạo file `src/train.py`: Cài đặt khung huấn luyện cho cả 3 thuật toán được yêu cầu:
  - `Logistic Regression`
  - `Support Vector Machines (SVM)`
  - `Naive Bayes`
- [x] Lưu mô hình sau khi huấn luyện vào thư mục `models/`.

### Bước 6: Đánh giá mô hình (Workflow 3 & 4)
- [x] Tạo file `src/evaluate.py`: Tính toán và in ra các chỉ số: **Accuracy**, **Precision**, **Recall**, và **F1-score** trên tập test cho cả 3 mô hình để so sánh.
- [x] Viết một hàm `predict_new_email(text)` để giả lập việc triển khai (Deployment) phân loại một email bất kỳ được nhập vào.

## Tích hợp giao diện Web và báo cáo
- [x] Thêm Streamlit vào `requirements.txt` và tạo giao diện phân loại SVM tại `demo/app.py`.
- [x] Xuất báo cáo HTML gồm biểu đồ Accuracy và Confusion Matrix SVM vào `reports/data.html` khi chạy đánh giá.

## 📑 Quy tắc thực hiện (Rules)
1. Viết code sạch (clean code), chuẩn PEP8, có comment giải thích ngắn gọn bằng tiếng Anh hoặc tiếng Việt ở đầu các hàm.
2. Các file phải liên kết import/export chính xác với nhau (ví dụ: `train.py` gọi hàm từ `feature_engineering.py`).
3. Sau khi tạo xong mỗi file hoặc mỗi bước, hãy cập nhật dấu tích `[x]` vào file `todo.md` này để tôi theo dõi tiến độ.

AI Agent hãy đọc cấu trúc trên và tự động tạo tất cả các thư mục trống cùng các file script (.py, .md) tương ứng. Nội dung trong các file script chỉ cần viết sẵn khung hàm (boilerplate) cơ bản để tôi điền logic sau