"""Thuật toán 1: Logistic Regression.

Mô hình tính điểm tuyến tính từ các đặc trưng rồi đưa qua hàm logistic (sigmoid)
để ra xác suất email là spam, nằm trong khoảng 0 đến 1.
Tham số C càng lớn thì mô hình càng ít bị phạt (regularization yếu).
"""

from sklearn.linear_model import LogisticRegression

NAME = "logistic_regression"
TITLE = "Logistic Regression"


def build_model(C: float = 10, max_iter: int = 1000) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=max_iter, solver="liblinear")
