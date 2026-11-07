# Thí nghiệm mở rộng

## Feature engineering

| Cấu hình | Ghi chú | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| TF-IDF unigram | 46891 đặc trưng | 0.9923 | 0.9881 | 0.9959 | 0.9920 |
| TF-IDF unigram + bigram | 50000 đặc trưng | 0.9915 | 0.9867 | 0.9955 | 0.9911 |
| TF-IDF unigram + bigram + đặc trưng ký tự | 50007 đặc trưng | 0.9925 | 0.9877 | 0.9966 | 0.9921 |

## Hyperparameter tuning

| Cấu hình | Ghi chú | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | tốt nhất: {'C': 10} | 0.9910 | 0.9870 | 0.9942 | 0.9906 |
| SVM | tốt nhất: {'C': 1} | 0.9925 | 0.9877 | 0.9966 | 0.9921 |
| Naive Bayes | tốt nhất: {'alpha': 0.01} | 0.9885 | 0.9866 | 0.9894 | 0.9880 |

## Ensemble

| Cấu hình | Ghi chú | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Random Forest (200 cây) |  | 0.9842 | 0.9773 | 0.9900 | 0.9836 |
| Soft voting: LR + SVM + NB | trung bình xác suất | 0.9918 | 0.9901 | 0.9928 | 0.9914 |
| Soft voting: LR + SVM + NB + RF | trung bình xác suất | 0.9923 | 0.9901 | 0.9938 | 0.9919 |
