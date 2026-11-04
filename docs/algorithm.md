# Thuật toán

## Tiền xử lý

Nội dung email được chuyển về chữ thường, HTML và dấu câu được loại bỏ, sau đó
các stop words tiếng Anh được bỏ qua. Các bước này được dùng nhất quán trong
huấn luyện và dự đoán.

## TF-IDF

`TfidfVectorizer` biểu diễn email bằng trọng số tần suất từ đã chuẩn hóa theo
mức độ phổ biến của từ trong toàn bộ tập train. Vectorizer chỉ được fit trên
tập train để tránh rò rỉ thông tin từ tập test.

## Bộ phân loại

- Logistic Regression: mô hình tuyến tính phân loại nhị phân.
- SVM: bộ phân loại margin tối đa với kernel tuyến tính.
- Multinomial Naive Bayes: mô hình xác suất phù hợp với đặc trưng văn bản.

Các mô hình được so sánh bằng Accuracy, Precision, Recall và F1-score.
