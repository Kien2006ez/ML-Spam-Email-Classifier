"""Thuật toán 1: Logistic Regression, tự cài đặt bằng NumPy (gradient descent).

Ý tưởng:
  1. Tính điểm tuyến tính   z = w1*x1 + w2*x2 + ... + b      (x = đặc trưng của email)
  2. Đổi điểm thành xác suất spam bằng sigmoid:  p = 1 / (1 + e^(-z))
  3. Học w, b sao cho p gần nhãn thật nhất, bằng cách lặp lại:
        sai số   = p - nhãn_thật                 (nhãn: 1 = spam, 0 = ham)
        gradient = đặc_trưng * sai số / số_mẫu + lambda * w
        w = w - tốc_độ_học * gradient            (đi ngược hướng làm sai số tăng)

Tham số C càng nhỏ thì phạt w lớn càng mạnh (regularization), tránh học vẹt.

Chạy ví dụ nhỏ để theo dõi từng con số:  python -m model.algorithm.logistic_regression
"""

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin

NAME = "logistic_regression"
TITLE = "Logistic Regression"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))  # clip để exp không bị tràn số


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Thuật toán tự cài đặt. Kế thừa BaseEstimator chỉ để ghép được vào Pipeline của
    scikit-learn; toàn bộ phần tính toán bên dưới là code tay."""

    def __init__(
        self,
        C: float = 10,
        learning_rate: float = 50,
        n_epochs: int = 100,
        lr_decay: float = 0.02,
        batch_size: int = 256,
        random_state: int = 42,
    ):
        self.C = C
        self.learning_rate = learning_rate  # bước đi mỗi lần cập nhật
        self.n_epochs = n_epochs  # số lần duyệt qua toàn bộ dữ liệu
        self.lr_decay = lr_decay  # giảm dần tốc độ học để hội tụ êm
        self.batch_size = batch_size  # mỗi lần cập nhật dùng một nhóm nhỏ email (mini-batch)
        self.random_state = random_state

    def fit(self, X, y):
        X = sparse.csr_matrix(X)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.classes_ = np.array([0, 1])

        lam = 1.0 / (self.C * n_samples)  # hệ số phạt lambda
        rng = np.random.default_rng(self.random_state)
        w = np.zeros(n_features)
        b = 0.0

        for epoch in range(self.n_epochs):
            lr = self.learning_rate / (1 + self.lr_decay * epoch)
            order = rng.permutation(n_samples)  # xáo trộn thứ tự email mỗi vòng
            for start in range(0, n_samples, self.batch_size):
                idx = order[start : start + self.batch_size]
                X_batch = X[idx]

                p = sigmoid(X_batch @ w + b)  # xác suất spam hiện tại
                error = p - y[idx]  # sai số so với nhãn thật

                grad_w = X_batch.T @ error / len(idx) + lam * w
                grad_b = error.mean()
                w -= lr * grad_w  # cập nhật trọng số
                b -= lr * grad_b

        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X):
        p_spam = sigmoid(sparse.csr_matrix(X) @ self.coef_ + self.intercept_)
        return np.column_stack([1 - p_spam, p_spam])  # cột 0 = ham, cột 1 = spam

    def predict(self, X):
        return (sparse.csr_matrix(X) @ self.coef_ + self.intercept_ > 0).astype(int)


def build_model(C: float = 10) -> LogisticRegression:
    return LogisticRegression(C=C)


if __name__ == "__main__":
    # Ví dụ nhỏ: 4 email, 2 từ. Cột = [free, meeting]
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y = np.array([1, 1, 0, 0])  # 2 email spam chứa "free", 2 email ham chứa "meeting"

    # Đúng 1 bước cập nhật, không phạt (C rất lớn), tốc độ học = 1
    one_step = LogisticRegression(C=1e12, learning_rate=1, n_epochs=1, batch_size=4)
    one_step.fit(X, y)
    print("Sau 1 bước cập nhật (w ban đầu = 0):")
    print("  p ban đầu = sigmoid(0) = 0.5 cho cả 4 email")
    print("  sai số = p - nhãn = [-0.5, -0.5, 0.5, 0.5]")
    print("  gradient của w = [(-0.5-0.5)/4, (0.5+0.5)/4] = [-0.25, 0.25]")
    print(f"  w mới = w - 1 * gradient = [{one_step.coef_[0]:.2f}, {one_step.coef_[1]:.2f}]")

    # Huấn luyện đầy đủ rồi dự đoán email mới
    model = LogisticRegression(C=10, learning_rate=1, n_epochs=200, batch_size=4).fit(X, y)
    print(f"\nSau khi huấn luyện đủ: w(free) = {model.coef_[0]:.2f}, w(meeting) = {model.coef_[1]:.2f}")
    for name, email in [("free", [[1, 0]]), ("meeting", [[0, 1]])]:
        print(f"Email chỉ có '{name}': P(spam) = {model.predict_proba(np.array(email))[0, 1]:.3f}")
