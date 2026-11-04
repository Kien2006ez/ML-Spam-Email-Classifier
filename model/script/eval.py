"""Đánh giá mô hình, xuất báo cáo và dự đoán email mới.

Chạy:  python -m model.script.eval
"""

import json
from functools import lru_cache

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data.scripts.data_processing import load_dataset
from data.scripts.eva import dataset_summary
from model.config import (
    EVAL_RESULT_DIR,
    EVALUATION_DIR,
    MODEL_DIR,
    MODEL_NAMES,
    MODEL_TITLES,
    REPORT_DIR,
)
from model.script.data_preparation import prepare_test_data


@lru_cache(maxsize=8)
def _load_cached(path: str, mtime: float):
    return joblib.load(path)


def load_model(model_name: str):
    path = MODEL_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có mô hình '{model_name}' tại {path}.")
    return _load_cached(str(path), path.stat().st_mtime)


def predict_new_email(email_text: str, model_name: str = "svm") -> dict:
    """Phân loại 1 email. Trả về dict: model, label, is_spam, spam_probability."""
    model = load_model(model_name)
    proba = float(model.predict_proba([email_text])[0][1])
    is_spam = proba >= 0.5
    return {
        "model": model_name,
        "label": "spam" if is_spam else "ham",
        "is_spam": is_spam,
        "spam_probability": proba,
    }


def evaluate_all() -> pd.DataFrame:
    """Tính Accuracy, Precision, Recall, F1 trên tập test cho cả 3 mô hình."""
    X_test, y_test = prepare_test_data()

    rows = []
    for name in MODEL_NAMES:
        y_pred = load_model(name).predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )
    result = pd.DataFrame(rows)

    EVAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(EVAL_RESULT_DIR / "metrics.csv", index=False)
    (EVAL_RESULT_DIR / "metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    _write_summary(result, len(y_test))
    generate_html_report(result, len(y_test))

    print(f"\nKết quả trên tập test ({len(y_test)} email):")
    print(result.round(4).to_string(index=False))
    print(f"\nĐã ghi: {EVAL_RESULT_DIR}, {EVALUATION_DIR / 'summary.md'}, {REPORT_DIR / 'data.html'}")
    return result


def _write_summary(result: pd.DataFrame, n_test: int) -> None:
    """Ghi bảng so sánh dạng Markdown vào evaluation/summary.md."""
    best = result.loc[result["f1"].idxmax()]
    lines = [
        "# Tóm tắt đánh giá mô hình",
        "",
        f"Tập test: {n_test} email.",
        "",
        "| Mô hình | Accuracy | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for _, r in result.iterrows():
        lines.append(
            f"| {MODEL_TITLES[r['model']]} | {r['accuracy']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |"
        )
    lines += ["", f"Mô hình có F1 cao nhất: **{MODEL_TITLES[best['model']]}** ({best['f1']:.4f})."]
    (EVALUATION_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_html_report(result: pd.DataFrame, n_test: int) -> None:
    """Xuất reports/data.html: thống kê dữ liệu, biểu đồ Accuracy và Confusion Matrix của SVM."""
    stats = dataset_summary(load_dataset())

    bars = "".join(
        f'<div class="row"><span class="name">{MODEL_TITLES[r["model"]]}</span>'
        f'<div class="track"><div class="bar" style="width:{r["accuracy"] * 100:.2f}%"></div></div>'
        f'<span class="val">{r["accuracy"] * 100:.2f}%</span></div>'
        for _, r in result.iterrows()
    )
    table_rows = "".join(
        f"<tr><td>{MODEL_TITLES[r['model']]}</td><td>{r['accuracy']:.4f}</td>"
        f"<td>{r['precision']:.4f}</td><td>{r['recall']:.4f}</td><td>{r['f1']:.4f}</td></tr>"
        for _, r in result.iterrows()
    )
    svm = result[result["model"] == "svm"].iloc[0]

    page = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Báo cáo phân loại email spam</title>
<style>
body{{font-family:Arial,sans-serif;max-width:820px;margin:30px auto;padding:0 16px;color:#222}}
h1{{font-size:24px}} h2{{margin-top:32px;font-size:18px}}
table{{border-collapse:collapse;margin-top:8px}}
td,th{{border:1px solid #bbb;padding:8px 14px;text-align:center}}
th{{background:#f0f0f0}}
.row{{display:flex;align-items:center;gap:10px;margin:8px 0}}
.name{{width:170px}} .val{{width:70px}}
.track{{flex:1;background:#eee;height:20px;border-radius:4px}}
.bar{{background:#3b82f6;height:20px;border-radius:4px}}
.tn,.tp{{background:#d1fae5}} .fp,.fn{{background:#fee2e2}}
</style></head><body>
<h1>Báo cáo phân loại email spam</h1>

<h2>1. Dữ liệu (Enron-Spam)</h2>
<p>Tổng số email: <b>{stats['total']}</b> &nbsp;|&nbsp; Spam: <b>{stats['spam']}</b>
&nbsp;|&nbsp; Ham: <b>{stats['ham']}</b> &nbsp;|&nbsp; Tỉ lệ spam: <b>{stats['spam_ratio'] * 100:.1f}%</b></p>
<p>Độ dài trung bình: spam {stats['avg_words_spam']:.0f} từ, ham {stats['avg_words_ham']:.0f} từ.
Tập test: {n_test} email (20%).</p>

<h2>2. Accuracy của các mô hình</h2>
{bars}

<h2>3. Chỉ số chi tiết</h2>
<table><tr><th>Mô hình</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
{table_rows}</table>

<h2>4. Confusion Matrix - SVM</h2>
<table>
<tr><th></th><th>Dự đoán: Ham</th><th>Dự đoán: Spam</th></tr>
<tr><th>Thực tế: Ham</th><td class="tn">{svm['TN']}</td><td class="fp">{svm['FP']}</td></tr>
<tr><th>Thực tế: Spam</th><td class="fn">{svm['FN']}</td><td class="tp">{svm['TP']}</td></tr>
</table>
<p>FP: email thường bị đánh dấu nhầm là spam. FN: spam lọt qua bộ lọc.</p>
</body></html>
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "data.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    evaluate_all()
