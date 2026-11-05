"""Huấn luyện 3 mô hình và lưu vào model/save_models (kèm bản sao ở data/save_model).

Toàn bộ quá trình được ghi lại vào reports/eval-result/training_log.md.

Chạy:  python -m model.script.training
"""

import contextlib
import io
import shutil
import sys
import time
import traceback
from datetime import datetime

import joblib
from sklearn.pipeline import Pipeline

from data.scripts.data_collection import download_dataset
from model.algorithm.svm_feature_engineer import build_classifiers
from model.config import BACKUP_MODEL_DIR, EVAL_RESULT_DIR, MODEL_DIR
from model.script.data_preparation import prepare_training_data

LOG_PATH = EVAL_RESULT_DIR / "training_log.md"


class _Tee(io.TextIOBase):
    """Ghi đồng thời ra màn hình và vào bộ nhớ đệm để lưu log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_training() -> None:
    download_dataset()  # bỏ qua nếu đã có dữ liệu
    features, X_train_vec, y_train = prepare_training_data()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, clf in build_classifiers().items():
        print(f"\nHuấn luyện {name}: {clf}")
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        print(f"  xong sau {time.time() - t0:.1f} giây")

        # Ghép bộ đặc trưng (đã fit) + bộ phân loại thành 1 pipeline
        pipeline = Pipeline([("features", features), ("clf", clf)])
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(pipeline, path)
        shutil.copy(path, BACKUP_MODEL_DIR / path.name)
        print(f"  đã lưu {path.name} ({path.stat().st_size / 1024:.0f} KB)")

    print(f"\nĐã lưu mô hình vào {MODEL_DIR} (bản sao: {BACKUP_MODEL_DIR})")

    from model.script.eval import evaluate_all

    evaluate_all()


def main() -> None:
    buffer = io.StringIO()
    started = datetime.now()
    status = "Thành công"
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, buffer)):
            run_training()
    except Exception:
        status = "Lỗi"
        buffer.write(traceback.format_exc())
        raise
    finally:
        EVAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text(
            "# Log huấn luyện\n\n"
            f"- Bắt đầu: {started:%Y-%m-%d %H:%M:%S}\n"
            f"- Thời gian chạy: {(datetime.now() - started).total_seconds():.0f} giây\n"
            f"- Trạng thái: {status}\n\n"
            "## Nội dung\n\n```\n" + buffer.getvalue().strip() + "\n```\n",
            encoding="utf-8",
        )
        print(f"Đã ghi log: {LOG_PATH}")


if __name__ == "__main__":
    main()
