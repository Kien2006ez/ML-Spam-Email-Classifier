"""Chạy thử demo: dự đoán email mẫu bằng cả 3 mô hình, lưu vào reports/demo.

Chạy:  python -m model.script.demol
"""

import pandas as pd

from model.config import DEMO_RESULT_DIR, MODEL_NAMES, SAMPLE_EMAILS
from model.script.eval import predict_new_email


def main() -> None:
    rows = []
    for expected, text in SAMPLE_EMAILS.items():
        for name in MODEL_NAMES:
            r = predict_new_email(text, model_name=name)
            rows.append(
                {
                    "expected": expected,
                    "model": name,
                    "predicted": r["label"],
                    "spam_probability": round(r["spam_probability"], 4),
                }
            )

    df = pd.DataFrame(rows)
    DEMO_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEMO_RESULT_DIR / "demo_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
