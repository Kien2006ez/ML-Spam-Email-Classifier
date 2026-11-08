# PHÂN LOẠI EMAIL SPAM / HAM (ENRON DATASET)
**Môn học**: Machine Learning / Xử lý ngôn ngữ tự nhiên (NLP)  

**Phương pháp tiếp cận**: Tự lập trình thuật toán tính tay từ đầu (**From Scratch**) bằng `NumPy`.

---

## MỤC LỤC
1. [Giới thiệu đề tài](#1-giới-thiệu-đề-tài)
2. [Chi tiết bộ dữ liệu Enron Spam](#2-chi-tiết-bộ-dữ-liệu-enron-spam)
3. [Quy trình tiền xử lý dữ liệu (Data Preprocessing)](#3-quy-trình-tiền-xử-lý-dữ-liệu-data-preprocessing)
4. [Trích xuất đặc trưng với TF-IDF](#4-trích-xuất-đặc-trưng-với-tf-idf)
5. [Cơ sở toán học 3 thuật toán tính tay (From Scratch)](#5-cơ-sở-toán-học-3-thuật-toán-tính-tay-from-scratch)
   - [5.1. Logistic Regression](#51-logistic-regression)
   - [5.2. Multinomial Naive Bayes](#52-multinomial-naive-bayes)
   - [5.3. Linear Support Vector Machine (SVM)](#53-linear-support-vector-machine-svm)
6. [Hệ thống đánh giá và so sánh mô hình](#6-hệ-thống-đánh-giá-và-so-sánh-mô-hình)
7. [Kiến trúc thư mục dự án](#7-kiến-trúc-thư-mục-dự-án)
8. [Hướng dẫn cài đặt và thực thi (Quickstart)](#8-hướng-dẫn-cài-đặt-và-thực-thi-quickstart)

---

## 1. GIỚI THIỆU ĐỀ TÀI

### 1.1. Bối cảnh
Thư điện tử (Email) là phương tiện giao tiếp cốt lõi trong công việc và đời sống số. Tuy nhiên, sự gia tăng của thư rác (**Spam Email**) — bao gồm thư quảng cáo không mong muốn, thư giả mạo lừa đảo (Phishing), và email đính kèm mã độc — gây lãng phí tài nguyên mạng, giảm năng suất làm việc và tiềm ẩn nguy cơ bảo mật nghiêm trọng.

### 1.2. Mục tiêu bài làm
- Xây dựng hệ thống học máy phân loại tự động email thành 2 nhãn:
  - **Ham (Nhãn 0)**: Email hợp lệ, email công việc thông thường.
  - **Spam (Nhãn 1)**: Email rác, lừa đảo hoặc quảng cáo.
- **Yêu cầu kỹ thuật cốt lõi**: Không sử dụng các hàm huấn luyện đóng gói sẵn (`fit`) của Scikit-Learn. Toàn bộ logic toán học của các thuật toán được **tự lập trình tính tay từ đầu (from scratch)** bằng `NumPy` để làm chủ bản chất mô hình.

---

## 2. CHI TIẾT BỘ DỮ LIỆU ENRON SPAM

### 2.1. Xuất xứ và Lịch sử bộ dữ liệu (Data Provenance)
- **Tên tập dữ liệu**: Enron-Spam Dataset.
- **Quốc gia xuất xứ**: Hợp chúng quốc Hoa Kỳ (United States).
- **Cơ quan phát hành gốc**: Ủy ban Điều tiết Năng lượng Liên bang Hoa Kỳ (FERC - Federal Energy Regulatory Commission) phối hợp cùng nhóm nghiên cứu xử lý ngôn ngữ tự nhiên tại Đại học Carnegie Mellon (CMU) và MIT.
- **Nguồn gốc nội dung**:
  - **Email hợp lệ (Ham)**: Được trích xuất từ dữ liệu hòm thư điện tử thực tế của các nhân viên, kỹ sư và ban quản trị thuộc Tập đoàn Năng lượng Enron (Houston, Texas, Mỹ) bị chính phủ Hoa Kỳ tịch thu và công khai sau cuộc điều tra gian lận tài chính năm 2001.
  - **Email rác (Spam)**: Được thu thập từ các dự án bẫy thư rác nổi tiếng thế giới như SpamAssassin, SpamBayes và kho lưu trữ của Bruce Guenter.
- **Khoảng thời gian thu thập**: Từ **tháng 12 năm 1999** đến **tháng 01 năm 2002** (chủ yếu tập trung dày đặc trong 2 năm 2000 và 2001).
- **Nguồn trích xuất file**: Dự án chuẩn hóa dữ liệu Enron-Spam bởi Marc Wiechmann (`https://github.com/MWiechmann/enron_spam_data`).

### 2.2. Cấu trúc file dữ liệu gốc (`enron_spam_data.csv`)
- **Dung lượng file gốc**: ~51.7 MB.
- **Tổng số dòng dữ liệu**: **33.716 dòng**.
- **Các trường thông tin (5 cột)**:
  1. `Message ID`: Số định danh duy nhất của mỗi email (kiểu `int64`).
  2. `Subject`: Tiêu đề của email (chứa nhiều từ khóa nhạy cảm: "Urgent", "Meeting", "Bonus",...).
  3. `Message`: Nội dung chi tiết của bức thư.
  4. `Spam/Ham`: Nhãn gốc dạng chuỗi (`spam` hoặc `ham`).
  5. `Date`: Ngày giờ gửi thư (kiểu chuỗi `YYYY-MM-DD`).

### 2.3. Phân phối nhãn dữ liệu (Class Distribution)
| Nhãn | Số lượng email | Tỉ lệ (%) | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Spam** | 17.171 | **50.93%** | Tương đối cân bằng |
| **Ham** | 16.545 | **49.07%** | Tương đối cân bằng |
| **Tổng cộng** | **33.716** | **100%** | Dữ liệu lý tưởng để huấn luyện |

### 2.4. Phân tích giá trị bị khuyết (Missing Values)
- Cột `Subject`: Thiếu **289** giá trị (nhiều email không có tiêu đề).
- Cột `Message`: Thiếu **371** giá trị (email chỉ có tiêu đề, thân thư rỗng).
- Cột `Spam/Ham`, `Message ID`, `Date`: **0** giá trị thiếu.

---

## 3. QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)

Do văn bản người gửi nhập vào có nhiều nhiễu, link web, dấu câu và các định dạng không đồng nhất, quy trình tiền xử lý được thiết kế nghiêm ngặt:

## 3.1. QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)

Quy trình tiền xử lý đóng vai trò quyết định đến chất lượng phân loại của mô hình túi từ:

### Chi tiết các bước kỹ thuật:
1. **Xử lý Dữ liệu khuyết thiếu (Missing Data Handling)**:
   * *Thực trạng*: Cột `Subject` có 289 dòng bị thiếu (`NaN`), cột `Message` có 371 dòng bị thiếu (`NaN`).
   * *Giải pháp*: Điền chuỗi rỗng `""` thay vì xóa dòng ngay lập tức, vì email không tiêu đề vẫn có thể phân loại qua nội dung và ngược lại.
   * *Gộp dữ liệu*: Ghép hai trường thành cột duy nhất: `text = Subject + " " + Message`. Các dòng rỗng cả hai phần mới bị loại bỏ.
2. **Chuẩn hóa chữ thường (Lowercasing)**:
   * Áp dụng `text.lower()` để đưa các từ như `"FREE"`, `"Free"`, `"free"` về cùng một hình vị, giảm số chiều không gian đặc trưng và tránh phân mảnh từ vựng.
3. **Loại bỏ Link URLs & Chữ số bằng Biểu thức chính quy (Regex)**:
   * `re.sub(r'http\S+', '', text)`: Loại bỏ hoàn toàn các liên kết web (`http://`, `https://`, `www...`).
   * `re.sub(r'\d+', '', text)`: Loại bỏ các con số ngẫu nhiên, ngày tháng, số tiền.
4. **Loại bỏ dấu câu (Punctuation Stripping)**:
   * Sử dụng bảng ánh xạ `str.maketrans('', '', string.punctuation)` để loại bỏ toàn bộ 32 ký tự dấu câu (`!`, `@`, `#`, `$`, `%`,...), giúp tách các từ dính dấu câu như `"winner!"` thành `"winner"`.
5. **Lọc trùng lặp đa tầng (Multi-tier Duplicate Removal)**:
   * Xóa trùng lặp ở tầng dữ liệu gốc bằng `df.drop_duplicates()`.
   * Xóa trùng lặp ở tầng văn bản sạch bằng `df.drop_duplicates(subset=['text_clean'])` để loại bỏ các email spam được gửi hàng loạt cùng một nội dung, chống hiện tượng mô hình học vẹt (Overfitting).
6. **Lưu trữ dữ liệu sạch**:
   * Lưu kết quả ra file `data/raw/clean_spam_data.csv` với 2 cột trọng tâm: `text_clean` và `label` ($0$ hoặc $1$).

---

## 4. TRÍCH XUẤT ĐẶC TRƯNG VỚI TF-IDF (FEATURE EXTRACTION)

Phương pháp **TF-IDF (Term Frequency - Inverse Document Frequency)** được sử dụng để ánh xạ các chuỗi văn bản thành ma trận số học dựa trên tầm quan trọng thống kê của từng từ.

### 4.1. Công thức toán học
1. **Tần suất từ trong văn bản (Term Frequency - TF)**:
   $$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
   *(với $f_{t, d}$ là số lần từ $t$ xuất hiện trong văn bản $d$).*

2. **Nghịch đảo tần suất văn bản (Inverse Document Frequency - IDF)**:
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
   *(Các từ xuất hiện phổ biến ở mọi email như "the", "and" có IDF xấp xỉ 0; các từ hiếm và mang đặc trưng Spam cao như "lottery", "prize", "viagra" có IDF rất lớn).*

3. **Trọng số TF-IDF tổng hợp và chuẩn hóa $L_2$**:
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
   $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum v_i^2}}$$
   *(Chuẩn hóa vector giúp triệt tiêu hoàn toàn sự chênh lệch độ dài ngắn giữa các email).*

### 4.2. Cấu hình thực thi
* **Chia tập dữ liệu**: 80% Huấn luyện (Train) - 20% Kiểm thử (Test), áp dụng phân tầng `stratify=y`.
* **Giới hạn số lượng từ (`max_features = 5000`)**: Giữ lại 5.000 từ có trọng số TF-IDF cao nhất toàn bộ kho dữ liệu.
* **Lưu trữ**: File trích xuất được lưu tại `data/save_model/tfidf.pkl`.

---

## 5. CƠ SỞ TOÁN HỌC 3 THUẬT TOÁN TÍNH TAY (FROM SCRATCH)

Tất cả các thuật toán đều được lập trình thủ công bằng `numpy`, không dùng `sklearn.fit`:

### 5.1. Logistic Regression (Hồi quy Logistic)
* **Bản chất**: Phân loại tuyến tính nhị phân kết hợp hàm kích hoạt phi tuyến.
* **Công thức tuyến tính**:
  $$z = X \cdot w + b$$
* **Hàm kích hoạt Sigmoid**:
  $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
  *(Code sử dụng `np.clip(z, -500, 500)` để chống tràn số lũy thừa `OverflowError`).*
* **Hàm mất mát Cross-Entropy (Log Loss)**:
  $$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big]$$
* **Hạ độ dốc (Gradient Descent)**:
  $$dw = \frac{1}{m} X^T (\hat{y} - y), \quad db = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$
  $$w \leftarrow w - \alpha \cdot dw, \quad b \leftarrow b - \alpha \cdot db$$

---

### 5.2. Multinomial Naive Bayes (Bayes ngây thơ đa thức)
* **Bản chất**: Dựa trên định lý Bayes với giả định các từ khóa xuất hiện độc lập có điều kiện:
  $$P(y \mid X) \propto P(y) \prod_{j=1}^{d} P(x_j \mid y)$$
* **Làm mịn Laplace (Laplace Smoothing với $\alpha = 1.0$)**:
  Tránh xác suất bằng 0 khi gặp từ vựng mới chưa có trong tập train:
  $$P(x_j \mid y) = \frac{\sum_{i \in y} x_{ij} + \alpha}{\sum_{k=1}^{d} \sum_{i \in y} x_{ik} + \alpha \cdot d}$$
* **Log-Likelihood (Tránh tràn số dưới underflow)**:
  $$\log P(y \mid X) = \log P(y) + \sum_{j=1}^{d} x_j \cdot \log P(x_j \mid y)$$
  Quy tắc quyết định:
  $$\hat{y} = \arg\max_{y} \Big[ \log P(y) + X \cdot (\log P(x \mid y))^T \Big]$$

---

### 5.3. Linear Support Vector Machine (Linear SVM)
* **Bản chất**: Tìm kiếm siêu phẳng phân cách tối ưu $w \cdot x - b = 0$ sao cho khoảng cách lề (margin) lớn nhất.
* **Quy ước nhãn toán học**: $y \in \{-1, +1\}$ (với $0 \to -1$, $1 \to +1$).
* **Hàm mục tiêu Soft-Margin kết hợp Hinge Loss và Regularization $L_2$**:
  $$J(w, b) = \lambda \|w\|^2 + \frac{1}{m} \sum_{i=1}^{m} \max\Big(0, 1 - y_i(w \cdot x_i - b)\Big)$$
* **Cập nhật bằng Stochastic Gradient Descent (SGD)**:
  Tại mỗi mẫu huấn luyện $(x_i, y_i)$, kiểm tra điều kiện lề $y_i(w \cdot x_i - b) \ge 1$:
  * **Nếu đúng điều kiện** (nằm ngoài lề an toàn):
    $$w \leftarrow w - \alpha \cdot (2\lambda w)$$
  * **Nếu vi phạm điều kiện** (lọt vào lề hoặc đoán sai nhãn):
    $$w \leftarrow w - \alpha \cdot (2\lambda w - y_i x_i)$$
    $$b \leftarrow b - \alpha \cdot y_i$$

---

## 6. HỆ THỐNG ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH

Hệ thống đánh giá trên tập kiểm thử (Test Set) thông qua 4 thước đo chuẩn:
1. **Accuracy**: Đo độ chính xác trên toàn bộ email.
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision**: Đo tỉ lệ email hệ thống đoán là Spam thì thực sự là Spam (giảm thiểu chặn nhầm thư quan trọng).
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall**: Tỉ lệ hệ thống bắt được bao nhiêu % tổng số thư rác thực tế.
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score**: Trung bình điều hòa giữa Precision và Recall.
   $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Kết quả đánh giá được tự động lưu vào file `evaluation/comparison.csv` và trực quan hóa qua biểu đồ cột `evaluation/comparison_chart.png`.

---

## 7. KIẾN TRÚC THƯ MỤC DỰ ÁN

```text
Spam_Email_Classificationlab1/
│
├── data/
│   ├── raw_data/                  # Dữ liệu thô ban đầu
│   │   └── enron_spam_data.csv
│   │
│   ├── scripts/                   # Thu thập, tiền xử lý và thống kê dữ liệu
│   │   ├── load_data.py           # Đọc và tải dữ liệu
│   │   ├── data_cleaning.py       # Làm sạch dữ liệu
│   │   ├── preprocessing.py       # Tiền xử lý văn bản
│   │   ├── data_analysis.py       # Thống kê và phân tích dữ liệu
│   │   └── feature_extraction.py  # Trích xuất đặc trưng TF-IDF
│   │
│   └── save_model/                # Lưu trữ các bản sao lưu mô hình
│
├── model/
│   ├── algorithm/                 # Đặc trưng và bộ phân loại
│   │   ├── logistic_regression.py  # Thuật toán Logistic Regression
│   │   ├── naive_bayes.py          # Thuật toán Naive Bayes
│   │   └── svm.py                  # Thuật toán SVM
│   │
│   ├── script/                    # Chuẩn bị dữ liệu và xây dựng mô hình
│   │   ├── prepare_data.py        # Chuẩn bị và chia tập dữ liệu
│   │   ├── train.py               # Huấn luyện mô hình
│   │   ├── evaluate.py            # Đánh giá mô hình
│   │   ├── hyperparameter_tuning.py # Tối ưu siêu tham số
│   │   └── predict.py             # Dự đoán email mới
│   │
│   └── save_models/               # Các mô hình đã huấn luyện
│       ├── logistic_regression.pkl
│       ├── naive_bayes.pkl
│       ├── svm.pkl
│       └── tfidf.pkl              # Vectorizer TF-IDF đã fit
│
├── reports/                       # Báo cáo và kết quả trực quan
│   ├── data.html                  # Báo cáo phân tích dữ liệu
│   ├── eval-result/               # Kết quả đánh giá mô hình
│   └── demo/                      # Hình ảnh/kết quả demo
│
├── evaluation/                    # So sánh và tổng hợp mô hình
│   ├── comparison.py              # So sánh các mô hình
│   └── comparison.csv             # Bảng so sánh kết quả
│
├── demo/                          # Giao diện ứng dụng
│   └── app.py                     # Giao diện Streamlit
│
├── docs/                          # Tài liệu dự án và bài lab
│   ├── project_design.md           # Thiết kế tổng thể dự án
│   ├── logistic_regression.md      # Giải thích Logistic Regression
│   ├── naive_bayes.md              # Giải thích Naive Bayes
│   ├── svm.md                      # Giải thích SVM
│   └── lab_notes.md                # Nội dung và ghi chú bài lab
│
├── requirements.txt               # Các thư viện cần thiết
└── README.md                       # Giới thiệu và hướng dẫn sử dụng
