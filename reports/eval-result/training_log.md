# Log huấn luyện

- Bắt đầu: 2026-11-07 14:08:27
- Thời gian chạy: 23 giây
- Trạng thái: Thành công

## Nội dung

```
Đã có dữ liệu: C:\Users\Admin\Documents\BT Lab\data\raw_data\enron_spam_data.csv
Số email huấn luyện: 24369
Trích xuất đặc trưng (TF-IDF + ký tự) ...
Số đặc trưng: 50007

Huấn luyện logistic_regression: LogisticRegression()
  xong sau 2.5 giây
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
logistic_regression    0.9910     0.9870  0.9942 0.9906 3144  38  17 2894
                svm    0.9925     0.9877  0.9966 0.9921 3146  36  10 2901
        naive_bayes    0.9892     0.9877  0.9897 0.9887 3146  36  30 2881

Đã ghi: C:\Users\Admin\Documents\BT Lab\reports\eval-result, C:\Users\Admin\Documents\BT Lab\evaluation\summary.md, C:\Users\Admin\Documents\BT Lab\reports\data.html
```
