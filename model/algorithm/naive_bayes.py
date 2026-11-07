import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
 
NAME = "naive_bayes"
TITLE = "Naive Bayes"
 
 
class NaiveBayes(BaseEstimator, ClassifierMixin):
    """Thuật toán tự cài đặt. Kế thừa BaseEstimator chỉ để ghép được vào Pipeline của scikit-learn;
    toàn bộ phần tính toán bên dưới là code tay."""
 
    def __init__(self, alpha: float = 0.1):
        # alpha: hệ số làm mượt, để từ chưa từng gặp trong một lớp không làm xác suất bằng 0
        self.alpha = alpha
 
    def fit(self, X, y):
        X = sparse.csr_matrix(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)  # [0, 1] = [ham, spam]
        n_features = X.shape[1]
 
        log_prior = []
        log_likelihood = []
        for c in self.classes_:
            X_c = X[y == c]
 
            # Bước 1: xác suất tiên nghiệm P(lớp) = số email của lớp / tổng số email
            log_prior.append(np.log(X_c.shape[0] / X.shape[0]))
 
            # Bước 2: tổng "số lần" mỗi từ xuất hiện trong lớp này (với TF-IDF là tổng trọng số)
            word_count = np.asarray(X_c.sum(axis=0)).ravel()
 
            # Bước 3: P(từ | lớp) = (số lần từ + alpha) / (tổng số từ của lớp + alpha * số từ vựng)
            smoothed = word_count + self.alpha
            log_likelihood.append(np.log(smoothed / smoothed.sum()))
 
        self.log_prior_ = np.array(log_prior)  # kích thước: (số lớp,)
        self.log_likelihood_ = np.array(log_likelihood)  # kích thước: (số lớp, số từ vựng)
        return self
 
    def _scores(self, X):
        # Điểm của mỗi lớp = log P(lớp) + tổng (giá trị của từ * log P(từ | lớp))
        X = sparse.csr_matrix(X)
        return X @ self.log_likelihood_.T + self.log_prior_
 
    def predict_proba(self, X):
        scores = self._scores(X)
        scores = scores - scores.max(axis=1, keepdims=True)  # trừ max để exp không bị tràn số
        probs = np.exp(scores)
        return probs / probs.sum(axis=1, keepdims=True)  # chuẩn hoá để tổng hai lớp = 1
 
    def predict(self, X):
        return self.classes_[np.argmax(self._scores(X), axis=1)]
 
 
def build_model(alpha: float = 0.1) -> NaiveBayes:
    return NaiveBayes(alpha=alpha)