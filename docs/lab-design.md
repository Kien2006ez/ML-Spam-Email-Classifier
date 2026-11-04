# Thiết kế bài Lab

## Mục tiêu

Xây dựng workflow phân loại email thành `spam` hoặc `not spam`, từ làm sạch
văn bản đến đánh giá mô hình và dự đoán email mới.

## Quy trình

1. Chuẩn hóa văn bản: loại HTML, dấu câu và stop words.
2. Chia dữ liệu có phân tầng thành train/test theo tỷ lệ 80/20.
3. Biến đổi email thành vector TF-IDF; chỉ fit vectorizer trên tập train.
4. Huấn luyện Logistic Regression, SVM và Multinomial Naive Bayes.
5. So sánh Accuracy, Precision, Recall và F1-score trên tập test.
6. Lưu mô hình và vectorizer để sử dụng khi dự đoán.

## Chạy

Từ thư mục gốc, chạy `python -m src.train` rồi `python -m src.evaluate`.
Workflow hiện sử dụng dữ liệu mock trong `src/preprocessing.py`; thay bằng dữ
liệu thực có cột `text` và `label` khi mở rộng bài Lab.
