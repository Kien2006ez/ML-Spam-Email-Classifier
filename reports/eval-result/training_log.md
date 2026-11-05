# Log huấn luyện

- Bắt đầu: 2026-11-05 23:42:55
- Thời gian chạy: 20 giây
- Trạng thái: Thành công

## Nội dung

```
Đã có dữ liệu: C:\Users\Admin\Documents\BT Lab\data\raw_data\enron_spam_data.csv
Số email huấn luyện: 24369
Trích xuất đặc trưng (TF-IDF + ký tự) ...
Số đặc trưng: 50007

Huấn luyện logistic_regression: LogisticRegression(C=10, max_iter=1000, solver='liblinear')
  xong sau 0.5 giây
  đã lưu logistic_regression.joblib (2385 KB)

Huấn luyện svm: CalibratedClassifierCV(cv=3, estimator=LinearSVC())
  xong sau 0.3 giây
  đã lưu svm.joblib (3168 KB)

Huấn luyện naive_bayes: MultinomialNB(alpha=0.1)
  xong sau 0.0 giây
  đã lưu naive_bayes.joblib (3557 KB)

Đã lưu mô hình vào C:\Users\Admin\Documents\BT Lab\model\save_models (bản sao: C:\Users\Admin\Documents\BT Lab\data\save_model)

Kết quả trên tập test (6093 email):
              model  accuracy  precision  recall     f1   TN  FP  FN   TP
logistic_regression    0.9916     0.9877  0.9948 0.9913 3146  36  15 2896
                svm    0.9916     0.9904  0.9921 0.9912 3154  28  23 2888
        naive_bayes    0.9892     0.9877  0.9897 0.9887 3146  36  30 2881

Đã ghi: C:\Users\Admin\Documents\BT Lab\reports\eval-result, C:\Users\Admin\Documents\BT Lab\evaluation\summary.md, C:\Users\Admin\Documents\BT Lab\reports\data.html
```
