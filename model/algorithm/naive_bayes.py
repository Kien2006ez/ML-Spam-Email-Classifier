"""Thuật toán 3: Naive Bayes (Multinomial).

Dựa trên định lý Bayes: P(spam | email) tỉ lệ với P(email | spam) * P(spam).
"Naive" vì giả định các từ độc lập với nhau khi đã biết lớp (spam hay ham).
Tham số alpha là hệ số làm mượt (smoothing) để từ chưa gặp không làm xác suất bằng 0.
"""

from sklearn.naive_bayes import MultinomialNB

NAME = "naive_bayes"
TITLE = "Naive Bayes"


def build_model(alpha: float = 0.1) -> MultinomialNB:
    return MultinomialNB(alpha=alpha)
