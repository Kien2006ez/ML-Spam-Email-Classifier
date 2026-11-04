# Thí nghiệm mở rộng

## Feature engineering

| Cấu hình | Ghi chú | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| TF-IDF unigram | 46891 đặc trưng | 0.9918 | 0.9877 | 0.9952 | 0.9914 |
| TF-IDF unigram + bigram | 50000 đặc trưng | 0.9918 | 0.9877 | 0.9952 | 0.9914 |
| TF-IDF unigram + bigram + đặc trưng ký tự | 50007 đặc trưng | 0.9916 | 0.9877 | 0.9948 | 0.9913 |

## Hyperparameter tuning

| Cấu hình | Ghi chú | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | tốt nhất: {'C': 100} | 0.9911 | 0.9864 | 0.9952 | 0.9908 |
| SVM (LinearSVC) | tốt nhất: {'C': 1} | 0.9916 | 0.9877 | 0.9948 | 0.9913 |
| Naive Bayes | tốt nhất: {'alpha': 0.01} | 0.9885 | 0.9866 | 0.9894 | 0.9880 |

## Ensemble

| Cấu hình | Ghi chú | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Random Forest (200 cây) |  | 0.9842 | 0.9773 | 0.9900 | 0.9836 |
| Soft voting: LR + SVM + NB | trung bình xác suất | 0.9915 | 0.9904 | 0.9918 | 0.9911 |
| Soft voting: LR + SVM + NB + RF | trung bình xác suất | 0.9916 | 0.9891 | 0.9935 | 0.9913 |
