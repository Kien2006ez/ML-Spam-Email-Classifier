# Log huấn luyện

- Bắt đầu: 2026-11-08 19:34:45
- Thời gian chạy: 19 giây
- Trạng thái: Thành công

## Nội dung

```
Đã có dữ liệu: C:\Users\Admin\Documents\BT Lab\data\raw_data\enron_spam_data.csv
Số email huấn luyện: 24369
Trích xuất đặc trưng (TF-IDF + ký tự) ...
Số đặc trưng: 50002

Huấn luyện logistic_regression: LogisticRegression()
  xong sau 2.3 giây
  đã lưu logistic_regression.joblib (2385 KB)

Huấn luyện svm: LinearSVM()
  xong sau 1.0 giây
  đã lưu svm.joblib (2385 KB)

Huấn luyện naive_bayes: NaiveBayes()
  xong sau 0.0 giây
  đã lưu naive_bayes.joblib (2775 KB)

Đã lưu mô hình vào C:\Users\Admin\Documents\BT Lab\model\save_models (bản sao: C:\Users\Admin\Documents\BT Lab\data\save_model)

Kết quả trên tập test (6093 email):
              model  accuracy  precision  recall     f1   TN  FP  FN   TP
logistic_regression    0.9906     0.9857  0.9948 0.9903 3140  42  15 2896
                svm    0.9918     0.9877  0.9952 0.9914 3146  36  14 2897
        naive_bayes    0.9893     0.9880  0.9897 0.9888 3147  35  30 2881

Đã ghi: C:\Users\Admin\Documents\BT Lab\reports\eval-result, C:\Users\Admin\Documents\BT Lab\evaluation\summary.md, C:\Users\Admin\Documents\BT Lab\reports\data.html
```
