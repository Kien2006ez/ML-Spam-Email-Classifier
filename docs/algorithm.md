# Thuật toán

## Mục tiêu

Biến nội dung email thành số rồi dùng mô hình học máy dự đoán email đó có phải spam không.

## Đầu vào / Đầu ra

- **Đầu vào**: văn bản email (tiêu đề + nội dung).
- **Đầu ra**: nhãn `spam` / `ham` và xác suất spam (0–1).

## Tiền xử lý

`data/scripts/data_processing.py`: ghép tiêu đề và nội dung, đưa về chữ thường, bỏ thẻ HTML, URL, dấu câu, chữ số. Stop words bỏ ở bước TF-IDF.

## Đặc trưng

`model/algorithm/svm_feature_engineer.py`:
- **TF-IDF** (unigram + bigram, tối đa 50.000 đặc trưng), chỉ học trên tập train.
- **Tần suất ký tự**: `!`, `$`, chữ HOA, chữ số, URL, HTML, độ dài email.

## Mô hình

Mỗi thuật toán nằm ở một file riêng trong `model/algorithm/`:

- **Logistic Regression** (`logistic_regression.py`): tính xác suất spam bằng hàm logistic.
- **SVM (LinearSVC)** (`svm_feature_engineer.py`, kèm phần đặc trưng): tìm siêu phẳng tách hai lớp với lề lớn nhất; bọc `CalibratedClassifierCV` để có xác suất.
- **Naive Bayes** (`naive_bayes.py`): dựa trên định lý Bayes, giả định các từ độc lập theo lớp.

## Đánh giá

Accuracy, Precision, Recall, F1 và ma trận nhầm lẫn trên tập test 20% (chia phân tầng).

## Thí nghiệm mở rộng

`model/script/tuning.py`: so sánh tổ hợp đặc trưng, tuning bằng GridSearchCV, ensemble (Random Forest, soft voting). Kết quả đều quanh F1 ≈ 0,991, tức là mô hình cơ bản đã đủ tốt trên bộ dữ liệu này.
