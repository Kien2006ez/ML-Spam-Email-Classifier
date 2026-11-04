"""Chuẩn bị dữ liệu cho mô hình: chia train/test và fit bộ trích xuất đặc trưng."""

from data.scripts.data_processing import get_train_test_split
from model.algorithm.svm_feature_engineer import build_feature_extractor


def prepare_training_data():
    """Trả về (bộ đặc trưng đã fit, ma trận train, nhãn train)."""
    X_train, _, y_train, _ = get_train_test_split()
    print(f"Số email huấn luyện: {len(X_train)}")

    print("Trích xuất đặc trưng (TF-IDF + ký tự) ...")
    features = build_feature_extractor()
    X_train_vec = features.fit_transform(X_train)
    print(f"Số đặc trưng: {X_train_vec.shape[1]}")
    return features, X_train_vec, y_train


def prepare_test_data():
    """Trả về (email test, nhãn test)."""
    _, X_test, _, y_test = get_train_test_split()
    return X_test, y_test
