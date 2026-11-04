"""Huấn luyện 3 mô hình và lưu vào model/save_models (kèm bản sao ở data/save_model).

Chạy:  python -m model.script.training
"""

import shutil

import joblib
from sklearn.pipeline import Pipeline

from data.scripts.data_collection import download_dataset
from model.algorithm.svm_feature_engineer import build_classifiers
from model.config import BACKUP_MODEL_DIR, MODEL_DIR
from model.script.data_preparation import prepare_training_data


def main() -> None:
    download_dataset()  # bỏ qua nếu đã có dữ liệu
    features, X_train_vec, y_train = prepare_training_data()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, clf in build_classifiers().items():
        print(f"Huấn luyện {name} ...")
        clf.fit(X_train_vec, y_train)

        # Ghép bộ đặc trưng (đã fit) + bộ phân loại thành 1 pipeline
        pipeline = Pipeline([("features", features), ("clf", clf)])
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(pipeline, path)
        shutil.copy(path, BACKUP_MODEL_DIR / path.name)

    print(f"Đã lưu mô hình vào {MODEL_DIR} (bản sao: {BACKUP_MODEL_DIR})")

    from model.script.eval import evaluate_all

    evaluate_all()


if __name__ == "__main__":
    main()
