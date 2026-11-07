"""Giao diện web Streamlit cho  thực hành demo phân loại email spam.

Chạy:  streamlit run demo/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import (  # noqa: E402
    EVAL_RESULT_DIR,
    MODEL_NAMES,
    MODEL_TITLES,
    SAMPLE_EMAILS,
)
from model.script.eval import predict_new_email  # noqa: E402

EXAMPLES = {"Ví dụ spam": SAMPLE_EMAILS["spam"], "Ví dụ email thường": SAMPLE_EMAILS["ham"]}


def _set_example(text: str) -> None:
    st.session_state["email_text"] = text


def render_classifier_tab() -> None:
    st.write("Dán nội dung email bên dưới để phân loại bằng mô hình đã huấn luyện.")
    st.caption(
        "Lưu ý: mô hình được huấn luyện trên bộ dữ liệu Enron-Spam (email tiếng Anh), "
        "nên hãy nhập email bằng tiếng Anh để có kết quả đáng tin cậy."
    )

    model_name = st.selectbox(
        "Chọn mô hình",
        MODEL_NAMES,
        index=MODEL_NAMES.index("svm"),
        format_func=lambda n: MODEL_TITLES[n],
    )

    col1, col2 = st.columns(2)
    for col, (label, text) in zip((col1, col2), EXAMPLES.items()):
        col.button(label, on_click=_set_example, args=(text,))

    email_text = st.text_area(
        "Nội dung Email",
        key="email_text",
        height=260,
        placeholder="Dán nội dung email cần phân tích tại đây...",
    )
    compare = st.checkbox("So sánh cả 3 mô hình")

    if not st.button("Phân tích Email", type="primary"):
        return
    if not email_text.strip():
        st.warning("Vui lòng nhập nội dung email trước khi phân tích.")
        return

    try:
        result = predict_new_email(email_text, model_name=model_name)
    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "Hãy chạy `python -m model.script.training` từ thư mục gốc dự án, "
            "sau đó tải lại ứng dụng."
        )
        return

    prob = result["spam_probability"]
    if result["is_spam"]:
        st.error(f"🚨 SPAM — xác suất spam {prob:.1%}")
    else:
        st.success(f"✅ KHÔNG PHẢI SPAM (ham) — xác suất spam chỉ {prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0), text="Xác suất là spam")

    if compare:
        rows = []
        for name in MODEL_NAMES:
            r = predict_new_email(email_text, model_name=name)
            rows.append(
                {
                    "Mô hình": MODEL_TITLES[name],
                    "Kết quả": "SPAM" if r["is_spam"] else "HAM",
                    "Xác suất spam": f"{r['spam_probability']:.1%}",
                }
            )
        st.table(pd.DataFrame(rows))


def render_metrics_tab() -> None:
    csv_path = EVAL_RESULT_DIR / "metrics.csv"
    if not csv_path.exists():
        st.info("Chưa có kết quả đánh giá. Hãy chạy `python -m model.script.training` trước.")
        return

    df = pd.read_csv(csv_path)
    df["model"] = df["model"].map(MODEL_TITLES)
    st.subheader("Kết quả trên tập test")
    st.dataframe(df.set_index("model"))
    st.bar_chart(
    df.set_index("model")[["accuracy", "precision", "recall", "f1"]], stack=False
)
    st.caption(
        "TN/FP/FN/TP là các ô của ma trận nhầm lẫn (spam = lớp dương). "
        "FP là email thường bị đánh dấu nhầm là spam, FN là spam lọt qua bộ lọc."
    )


def main() -> None:
    st.set_page_config(page_title="Phân loại Email Spam", page_icon="📧", layout="centered")
    st.title("Demo Phân loại Email Spam")
    st.write("Logistic Regression · SVM · Naive Bayes với đặc trưng TF-IDF")

    tab_classify, tab_metrics = st.tabs(["Phân loại email", "Đánh giá mô hình"])
    with tab_classify:
        render_classifier_tab()
    with tab_metrics:
        render_metrics_tab()


if __name__ == "__main__":
    main()
