"""Thuật toán 2: SVM tuyến tính tự cài đặt bằng NumPy, cùng phần trích xuất đặc trưng.

Ý tưởng SVM: tìm đường (siêu phẳng) w.x + b = 0 tách spam và ham sao cho khoảng cách
(lề) tới các email gần đường nhất là lớn nhất.
  - Nhãn được đổi thành +1 (spam) và -1 (ham).
  - Hàm mất mát hinge:  max(0, 1 - nhãn * (w.x + b))
      + Email đã nằm đúng phía và cách đường đủ xa (nhãn * điểm >= 1): không bị phạt.
      + Email nằm sai phía hoặc quá sát đường: bị phạt, và chỉ những email này kéo w thay đổi.
  - Mỗi bước, w còn bị co nhẹ (regularization) để lề rộng và không học vẹt.
  - Dùng thuật toán Pegasos (gradient descent ngẫu nhiên) để học w, b.
  - Điểm SVM không phải xác suất, nên đổi sang xác suất bằng sigmoid(a * điểm + c)
    với a, c học từ dữ liệu train (Platt scaling).

Logistic Regression và Naive Bayes nằm ở file riêng cùng thư mục này.
Chạy ví dụ nhỏ:  python -m model.algorithm.svm_feature_engineer
"""

import math
import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from data.scripts.data_processing import clean_text
from model.algorithm import logistic_regression, naive_bayes

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_HTML_TAG = re.compile(r"<[^>]+>")


class CharFeatures(BaseEstimator, TransformerMixin):
    """Tần suất ký tự đặc biệt (!, $, chữ HOA, số, URL, HTML, độ dài); đều không âm."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            text = str(text)
            n = max(len(text), 1)
            rows.append(
                [
                    text.count("!") / n,
                    text.count("$") / n,
                    sum(c.isupper() for c in text) / n,
                    sum(c.isdigit() for c in text) / n,
                    min(len(_URL.findall(text)), 10) / 10,
                    1.0 if _HTML_TAG.search(text) else 0.0,
                    math.log1p(n) / 15,
                ]
            )
        return np.asarray(rows, dtype=float)


def build_feature_extractor(
    ngram_range: tuple = (1, 2), use_char_features: bool = True
) -> FeatureUnion:
    """Nhận danh sách email thô, trả về ma trận đặc trưng.

    Hai tham số dùng cho thí nghiệm so sánh tổ hợp đặc trưng; mặc định là cấu hình
    của mô hình chính.
    """
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words="english",
        token_pattern=r"(?u)\b[a-z]{2,}\b",
        ngram_range=ngram_range,
        min_df=3,
        max_df=0.9,
        max_features=50000,
        sublinear_tf=True,
    )
    parts = [("tfidf", tfidf)]
    if use_char_features:
        chars = Pipeline([("chars", CharFeatures()), ("scale", MaxAbsScaler())])
        parts.append(("chars", chars))
    return FeatureUnion(parts)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LinearSVM(BaseEstimator, ClassifierMixin):
    """Thuật toán tự cài đặt. Kế thừa BaseEstimator chỉ để ghép được vào Pipeline của
    scikit-learn; toàn bộ phần tính toán bên dưới là code tay."""

    def __init__(
        self,
        C: float = 1.0,
        n_epochs: int = 30,
        batch_size: int = 256,
        random_state: int = 42,
    ):
        self.C = C  # C nhỏ: lề rộng, chấp nhận sai nhiều hơn; C lớn: cố phân loại đúng hết
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        X = sparse.csr_matrix(X)
        y01 = np.asarray(y, dtype=float)
        y_signed = 2 * y01 - 1  # đổi nhãn 0/1 thành -1/+1
        n_samples, n_features = X.shape
        self.classes_ = np.array([0, 1])

        lam = 1.0 / (self.C * n_samples)  # hệ số phạt lambda
        rng = np.random.default_rng(self.random_state)
        w = np.zeros(n_features)
        b = 0.0
        step = 0

        for _ in range(self.n_epochs):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                step += 1
                eta = 1.0 / (lam * (step + 100))  # tốc độ học giảm dần theo số bước (Pegasos)
                idx = order[start : start + self.batch_size]
                X_batch, y_batch = X[idx], y_signed[idx]

                # Chỉ những email vi phạm lề (nhãn * điểm < 1) mới kéo w thay đổi
                violating = y_batch * (X_batch @ w + b) < 1
                pull = X_batch[violating].T @ y_batch[violating] / len(idx)

                w = (1 - eta * lam) * w + eta * pull  # co w nhẹ, rồi kéo theo các email vi phạm
                b += eta * 0.1 * y_batch[violating].sum() / len(idx)

        self.coef_ = w
        self.intercept_ = b
        self._fit_probability(self.decision_function(X), y01)
        return self

    def decision_function(self, X):
        """Điểm SVM: dương nghĩa là spam, càng xa 0 thì càng chắc chắn."""
        return sparse.csr_matrix(X) @ self.coef_ + self.intercept_

    def _fit_probability(self, scores, y01):
        # Platt scaling: học a, c để P(spam) = sigmoid(a * điểm + c) bằng gradient descent
        a, c = 1.0, 0.0
        for _ in range(2000):
            error = _sigmoid(a * scores + c) - y01
            a -= 0.5 * (error * scores).mean()
            c -= 0.5 * error.mean()
        self.platt_ = (a, c)

    def predict_proba(self, X):
        a, c = self.platt_
        p_spam = _sigmoid(a * self.decision_function(X) + c)
        return np.column_stack([1 - p_spam, p_spam])  # cột 0 = ham, cột 1 = spam

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def build_svm(C: float = 1.0) -> LinearSVM:
    return LinearSVM(C=C)


def build_classifiers() -> dict:
    """Ba bộ phân loại theo đề bài, mỗi thuật toán nằm ở một file riêng."""
    return {
        logistic_regression.NAME: logistic_regression.build_model(),
        "svm": build_svm(),
        naive_bayes.NAME: naive_bayes.build_model(),
    }


if __name__ == "__main__":
    # Ví dụ nhỏ: 4 email, 2 từ. Cột = [free, meeting]
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y = np.array([1, 1, 0, 0])

    model = LinearSVM(C=10, n_epochs=200, batch_size=4).fit(X, y)
    print(f"w(free) = {model.coef_[0]:.2f}, w(meeting) = {model.coef_[1]:.2f}, b = {model.intercept_:.2f}")
    for name, email in [("free", [[1, 0]]), ("meeting", [[0, 1]])]:
        e = np.array(email)
        print(
            f"Email chỉ có '{name}': điểm = {model.decision_function(e)[0]:+.2f} -> "
            f"{'SPAM' if model.predict(e)[0] == 1 else 'HAM'} "
            f"(P(spam) = {model.predict_proba(e)[0, 1]:.3f})"
        )
