# Thuật toán

## Mục tiêu

Biến nội dung email thành số rồi dùng mô hình học máy dự đoán email đó có phải spam không.

## Đầu vào / Đầu ra

- **Đầu vào**: nội dung email (tiêu đề + nội dung, tiếng Anh).
- **Đầu ra**: nhãn `spam` / `ham` kèm xác suất spam từ 0 đến 1.
- **Dữ liệu**: Enron-Spam, 33.716 email có nhãn, tự tải về khi huấn luyện ([nguồn](https://github.com/MWiechmann/enron_spam_data)).

## Tiền xử lý

`data/scripts/data_processing.py`: ghép tiêu đề và nội dung, đưa về chữ thường, bỏ thẻ HTML, URL, dấu câu, chữ số. Stop words bỏ ở bước TF-IDF.

## Đặc trưng

`model/algorithm/svm_feature_engineer.py`:
- **TF-IDF** (unigram + bigram, tối đa 50.000 đặc trưng), chỉ học trên tập train.
- **Tần suất ký tự**: `!`, `$`, chữ HOA, chữ số, URL, HTML, độ dài email.

## Mô hình

Ba thuật toán được **tự cài đặt bằng NumPy**, mỗi thuật toán một file trong `model/algorithm/`:

- **Logistic Regression** (`logistic_regression.py`): tính điểm tuyến tính rồi đổi thành xác suất spam bằng sigmoid; học trọng số bằng gradient descent.
- **SVM (LinearSVC)** (`svm_feature_engineer.py`, kèm phần đặc trưng): tìm siêu phẳng tách hai lớp với lề lớn nhất; dùng hinge loss, học bằng Pegasos (SGD); đổi điểm sang xác suất bằng sigmoid (Platt scaling).
- **Naive Bayes** (`naive_bayes.py`): dựa trên định lý Bayes, đếm từ theo từng lớp và cộng log xác suất; giả định các từ độc lập theo lớp.

## Đánh giá

Accuracy, Precision, Recall, F1 và ma trận nhầm lẫn trên tập test 20% (chia phân tầng).

## Thí nghiệm mở rộng

`model/script/tuning.py`: so sánh tổ hợp đặc trưng, tuning bằng GridSearchCV, ensemble (Random Forest, soft voting). Kết quả đều quanh F1 ≈ 0,99 (Random Forest thấp hơn một chút, 0,983), tức là mô hình cơ bản đã đủ tốt trên bộ dữ liệu này.

## Ghi chú

Các lớp thuật toán kế thừa `BaseEstimator` của scikit-learn chỉ để ghép được vào `Pipeline`; phần tính toán là code tay. TF-IDF, chia train/test và các chỉ số đánh giá vẫn dùng scikit-learn.
