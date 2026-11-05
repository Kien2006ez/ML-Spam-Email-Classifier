"""Thuật toán 2: SVM, cùng phần trích xuất đặc trưng (TF-IDF + ký tự).

SVM tìm một siêu phẳng tách email spam và ham sao cho khoảng cách (lề) tới các điểm
gần nhất là lớn nhất; rất hợp với dữ liệu văn bản nhiều chiều.
Logistic Regression và Naive Bayes nằm ở file riêng cùng thư mục này.
"""

import math
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

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


def build_svm(C: float = 1.0) -> CalibratedClassifierCV:
    """LinearSVC không có predict_proba nên bọc CalibratedClassifierCV để có xác suất."""
    return CalibratedClassifierCV(LinearSVC(C=C), cv=3)


def build_classifiers() -> dict:
    """Ba bộ phân loại theo đề bài, mỗi thuật toán nằm ở một file riêng."""
    return {
        logistic_regression.NAME: logistic_regression.build_model(),
        "svm": build_svm(),
        naive_bayes.NAME: naive_bayes.build_model(),
    }
