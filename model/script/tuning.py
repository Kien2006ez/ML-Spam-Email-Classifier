"""Thí nghiệm mở rộng theo mục "Additional Considerations" của đề bài.

1. Feature Engineering: so sánh các tổ hợp đặc trưng.
2. Hyperparameter Tuning: GridSearchCV cho Logistic Regression, SVM, Naive Bayes.
3. Ensemble: Random Forest và trung bình xác suất (soft voting) của nhiều mô hình.

Chạy:  python -m model.script.tuning     (mất vài phút)
Kết quả: reports/eval-result/extra_experiments.csv, best_params.json
         evaluation/extra_experiments.md
Tham số được chọn bằng cross-validation trên tập train; tập test chỉ dùng để báo cáo.
"""

import json
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from data.scripts.data_collection import download_dataset
from data.scripts.data_processing import get_train_test_split
from model.algorithm import logistic_regression, naive_bayes
from model.algorithm.svm_feature_engineer import (
    build_classifiers,
    build_feature_extractor,
    build_svm,
)
from model.config import EVAL_RESULT_DIR, EVALUATION_DIR, RANDOM_STATE


def _row(group: str, name: str, details: str, y_true, y_pred) -> dict:
    return {
        "group": group,
        "name": name,
        "details": details,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def feature_experiments(X_train, X_test, y_train, y_test) -> list:
    """So sánh 3 tổ hợp đặc trưng, cùng dùng SVM tự cài đặt để công bằng."""
    configs = [
        ("TF-IDF unigram", dict(ngram_range=(1, 1), use_char_features=False)),
        ("TF-IDF unigram + bigram", dict(ngram_range=(1, 2), use_char_features=False)),
        (
            "TF-IDF unigram + bigram + đặc trưng ký tự",
            dict(ngram_range=(1, 2), use_char_features=True),
        ),
    ]
    rows = []
    for name, kwargs in configs:
        print(f"[Đặc trưng] {name} ...")
        extractor = build_feature_extractor(**kwargs)
        Xtr = extractor.fit_transform(X_train)
        clf = build_svm().fit(Xtr, y_train)
        rows.append(
            _row("Feature engineering", name, f"{Xtr.shape[1]} đặc trưng",
                 y_test, clf.predict(extractor.transform(X_test)))
        )
    return rows


def tuning_experiments(Xtr, Xte, y_train, y_test):
    """GridSearchCV (3-fold, tối ưu F1) cho từng mô hình."""
    searches = {
        # Dùng các thuật toán tự cài đặt trong model/algorithm/
        "Logistic Regression": (
            logistic_regression.build_model(),
            {"C": [0.1, 1, 10, 100]},
        ),
        "SVM": (build_svm(), {"C": [0.1, 1, 10]}),
        "Naive Bayes": (naive_bayes.build_model(), {"alpha": [0.01, 0.1, 0.5, 1.0]}),
    }
    rows, best_params = [], {}
    for name, (estimator, grid) in searches.items():
        print(f"[Tuning] {name} ...")
        search = GridSearchCV(estimator, grid, scoring="f1", cv=3, n_jobs=-1)
        search.fit(Xtr, y_train)
        best_params[name] = {
            "params": search.best_params_,
            "cv_f1": round(float(search.best_score_), 4),
        }
        rows.append(
            _row("Hyperparameter tuning", name, f"tốt nhất: {search.best_params_}",
                 y_test, search.predict(Xte))
        )
    return rows, best_params


def ensemble_experiments(Xtr, Xte, y_train, y_test) -> list:
    """Random Forest và soft voting (trung bình xác suất spam)."""
    models = {name: clf.fit(Xtr, y_train) for name, clf in build_classifiers().items()}

    print("[Ensemble] Random Forest ...")
    forest = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
    ).fit(Xtr, y_train)

    probs = {name: m.predict_proba(Xte)[:, 1] for name, m in models.items()}
    probs["random_forest"] = forest.predict_proba(Xte)[:, 1]

    def vote(names):
        avg = sum(probs[n] for n in names) / len(names)
        return (avg >= 0.5).astype(int)

    return [
        _row("Ensemble", "Random Forest (200 cây)", "", y_test, forest.predict(Xte)),
        _row("Ensemble", "Soft voting: LR + SVM + NB", "trung bình xác suất",
             y_test, vote(["logistic_regression", "svm", "naive_bayes"])),
        _row("Ensemble", "Soft voting: LR + SVM + NB + RF", "trung bình xác suất",
             y_test, vote(list(probs))),
    ]


def _to_markdown(df: pd.DataFrame) -> str:
    lines = ["# Thí nghiệm mở rộng", ""]
    for group, part in df.groupby("group", sort=False):
        lines += [f"## {group}", "",
                  "| Cấu hình | Ghi chú | Accuracy | Precision | Recall | F1 |",
                  "|---|---|---|---|---|---|"]
        for _, r in part.iterrows():
            lines.append(
                f"| {r['name']} | {r['details']} | {r['accuracy']:.4f} | "
                f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    start = time.time()
    download_dataset()
    X_train, X_test, y_train, y_test = get_train_test_split()

    rows = feature_experiments(X_train, X_test, y_train, y_test)

    print("Trích xuất đặc trưng chính ...")
    extractor = build_feature_extractor()
    Xtr = extractor.fit_transform(X_train)
    Xte = extractor.transform(X_test)

    tuning_rows, best_params = tuning_experiments(Xtr, Xte, y_train, y_test)
    rows += tuning_rows
    rows += ensemble_experiments(Xtr, Xte, y_train, y_test)

    df = pd.DataFrame(rows)
    EVAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EVAL_RESULT_DIR / "extra_experiments.csv", index=False)
    (EVAL_RESULT_DIR / "best_params.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (EVALUATION_DIR / "extra_experiments.md").write_text(
        _to_markdown(df), encoding="utf-8"
    )

    print("\n" + df.drop(columns="details").round(4).to_string(index=False))
    print(f"\nHoàn tất sau {time.time() - start:.0f} giây.")


if __name__ == "__main__":
    main()
