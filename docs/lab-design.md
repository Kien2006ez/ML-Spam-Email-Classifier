# Thiết kế bài Lab

## Bối cảnh

Email rác (spam) chiếm một phần lớn thư gửi đến, gây phiền toái và có thể chứa link lừa đảo. Việc tự động lọc spam bằng học máy là một ứng dụng quen thuộc của bài toán phân loại văn bản. Bài Lab 1 yêu cầu xây dựng một hệ thống như vậy, từ dữ liệu đến sản phẩm demo.

## Mục tiêu

1. Phân loại mỗi email thành **spam** hoặc **không spam (ham)**.
2. Áp dụng và so sánh 3 mô hình: Logistic Regression, SVM, Naive Bayes.
3. Đánh giá bằng Accuracy, Precision, Recall, F1 trên tập test.
4. Tìm hiểu thêm các hướng cải thiện: đặc trưng, tuning siêu tham số, ensemble.
5. Triển khai giao diện web để phân loại email mới.

## Đầu vào / Đầu ra

**Bài toán**
- **Đầu vào**: nội dung một email (tiêu đề + nội dung, văn bản tiếng Anh).
- **Đầu ra**: nhãn `spam` hoặc `ham` (không spam) kèm xác suất spam từ 0 đến 1; từ 0,5 trở lên là spam.

**Hệ thống**
- **Đầu vào**: file `data/raw_data/enron_spam_data.csv` (cột Subject, Message, Spam/Ham) và email người dùng nhập trên giao diện.
- **Đầu ra**: mô hình `.joblib`, bảng chỉ số (`metrics.csv`), báo cáo `data.html`, kết quả dự đoán trên giao diện web.

## Dữ liệu

Enron-Spam (~33.700 email có nhãn, còn ~30.400 sau khi bỏ trùng), tự tải về `data/raw_data/`.

## Quy trình

1. Thu thập dữ liệu: `data/scripts/data_collection.py`
2. Tiền xử lý, chia train/test 80/20: `data/scripts/data_processing.py`
3. Đặc trưng và 3 thuật toán (mỗi thuật toán một file): `model/algorithm/` gồm `logistic_regression.py`, `svm_feature_engineer.py`, `naive_bayes.py`
4. Huấn luyện: `model/script/training.py` (ghi log vào `reports/eval-result/training_log.md`)
5. Đánh giá: `model/script/eval.py`
6. Thí nghiệm mở rộng: `model/script/tuning.py`
7. Giao diện web: `demo/app.py`

## Vị trí lưu kết quả

Mô hình ở `model/save_models/` (bản sao ở `data/save_model/`), kết quả ở `reports/` và `evaluation/`.

## Chạy

```
pip install -r requirements.txt
python -m model.script.training
streamlit run demo/app.py
```

## Hạn chế

Dữ liệu là email tiếng Anh nên không phù hợp với tiếng Việt. Độ chính xác ~99% có thể cao hơn thực tế vì bộ Enron dễ tách.
