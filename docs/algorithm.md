# Giải thích thuật toán

## 1. Tổng quan

- **Đầu vào**: nội dung một email (tiêu đề + nội dung, tiếng Anh).
- **Đầu ra**: nhãn `spam` hoặc `ham` kèm xác suất spam từ 0 đến 1; từ 0,5 trở lên là spam.
- **Tự cài đặt bằng NumPy**: 3 thuật toán Logistic Regression, SVM, Naive Bayes (`model/algorithm/`).
- **Dùng scikit-learn**: TF-IDF, chia train/test, các chỉ số đánh giá, `GridSearchCV` và Random Forest (thí nghiệm mở rộng). Các lớp thuật toán kế thừa `BaseEstimator` chỉ để ghép được vào `Pipeline`; phần tính toán là code tay.

## 2. Tiền xử lý (`data/scripts/data_processing.py`)

1. **Điền giá trị thiếu**: `Subject` thiếu 289 dòng, `Message` thiếu 371 dòng; điền chuỗi rỗng thay vì xoá dòng.
2. **Ghép văn bản**: `text = Subject + "\n" + Message`. Các email rỗng cả hai phần (51 email) bị loại.
3. **Bỏ trùng**: `drop_duplicates` theo `text`.
4. **Làm sạch** (hàm `clean_text`): giải mã ký tự HTML, đưa về chữ thường, bỏ thẻ HTML, thay URL và địa chỉ email bằng token, bỏ dấu câu và chữ số (chỉ giữ chữ cái a–z).
5. **Bỏ stop words** tiếng Anh ở bước TF-IDF.
6. **Chia dữ liệu**: 80% train, 20% test, phân tầng theo nhãn, `random_state = 42`.

Kết quả: 33.716 dòng gốc còn 30.462 email (14.552 spam, 15.910 ham), chia thành 24.369 email train và 6.093 email test.

## 3. Trích xuất đặc trưng (`model/algorithm/svm_feature_engineer.py`)

**TF-IDF** biến mỗi email thành một vector, mỗi từ (hoặc cặp từ liền nhau) có một trọng số. Cấu hình: unigram + bigram, tối đa 50.000 đặc trưng, bỏ từ xuất hiện dưới 3 email hoặc trên 90% email.

$$\text{tf}(t,d) = 1 + \ln f_{t,d}, \qquad \text{idf}(t) = \ln\frac{1+n}{1+\text{df}(t)} + 1$$

$$\text{tfidf}(t,d) = \text{tf}(t,d)\cdot\text{idf}(t), \qquad \mathbf{v} \leftarrow \frac{\mathbf{v}}{\lVert \mathbf{v}\rVert_2}$$

Trong đó $f_{t,d}$ là số lần từ $t$ xuất hiện trong email $d$, $n$ là số email, $\text{df}(t)$ là số email chứa từ $t$. Từ phổ biến như "the" có idf thấp, từ hiếm và đặc trưng như "viagra" có idf cao. Chuẩn hoá $L_2$ để độ dài email không ảnh hưởng.

**Đặc trưng ký tự**: tần suất dấu `!` và `$` (số lần chia độ dài email), co về cùng thang bằng `MaxAbsScaler`. Không dùng tỉ lệ chữ HOA vì bộ dữ liệu viết toàn chữ thường, đặc trưng đó làm email bình thường ngoài đời bị nhận nhầm là spam.

Tổng cộng mỗi email là một vector 50.002 số.

## 4. Ba thuật toán

Nhãn: ham = 0, spam = 1. $x$ là vector đặc trưng, $n$ là số email train, $m$ là kích thước mini-batch (256).

### 4.1. Logistic Regression (`logistic_regression.py`)

Tính điểm tuyến tính rồi đổi thành xác suất bằng sigmoid:

$$z = w\cdot x + b, \qquad p = \sigma(z) = \frac{1}{1+e^{-z}}$$

Hàm mất mát là cross-entropy cộng phạt $L_2$ với $\lambda = \dfrac{1}{C\,n}$ ($C = 10$):

$$J = -\frac{1}{n}\sum_i \big[y_i\ln p_i + (1-y_i)\ln(1-p_i)\big] + \frac{\lambda}{2}\lVert w\rVert^2$$

Học bằng gradient descent trên từng mini-batch:

$$g_w = \frac{1}{m}X_B^{\top}(p-y) + \lambda w, \qquad g_b = \frac{1}{m}\sum (p_i - y_i)$$

$$w \leftarrow w - \eta\, g_w, \qquad b \leftarrow b - \eta\, g_b, \qquad \eta_{\text{epoch}} = \frac{50}{1 + 0{,}02\cdot\text{epoch}}$$

Chạy 100 epoch. Mỗi bước, trọng số của từ đẩy về hướng giảm sai số $p - y$: từ hay xuất hiện trong spam có $w$ dương, từ hay xuất hiện trong ham có $w$ âm.

### 4.2. SVM tuyến tính (`svm_feature_engineer.py`)

Đổi nhãn thành $y \in \{-1,+1\}$ và tìm đường $w\cdot x + b = 0$ có lề lớn nhất. Hàm mục tiêu:

$$J = \frac{\lambda}{2}\lVert w\rVert^2 + \frac{1}{n}\sum_i \max\big(0,\ 1 - y_i (w\cdot x_i + b)\big), \qquad \lambda = \frac{1}{C\,n}\ (C=1)$$

Học bằng thuật toán Pegasos (SGD theo mini-batch). Chỉ những email vi phạm lề $V = \{\, i : y_i(w\cdot x_i + b) < 1 \,\}$ mới kéo $w$ thay đổi:

$$w \leftarrow (1-\eta\lambda)\,w + \frac{\eta}{m}\sum_{i\in V} y_i x_i, \qquad \eta_t = \frac{1}{\lambda\,(t+100)}$$

Chạy 30 epoch. Bias $b$ cập nhật theo $\sum_{i\in V} y_i$ với bước nhỏ hơn 10 lần.

Điểm SVM không phải xác suất, nên đổi sang xác suất bằng sigmoid học từ dữ liệu train (Platt scaling): $P(\text{spam}) = \sigma(a\cdot f(x) + c)$.

### 4.3. Naive Bayes (`naive_bayes.py`)

Dựa trên định lý Bayes với giả định các từ độc lập khi đã biết lớp:

$$P(y\mid x) \propto P(y)\prod_j P(x_j\mid y)$$

Ước lượng từ dữ liệu train, có làm mượt với $\alpha = 0{,}1$ để từ chưa gặp không làm xác suất bằng 0:

$$P(y) = \frac{N_y}{N}, \qquad P(j\mid y) = \frac{\sum_{i\in y} x_{ij} + \alpha}{\sum_{k}\big(\sum_{i\in y} x_{ik} + \alpha\big)}$$

Để tránh nhân nhiều số rất nhỏ, tính trong không gian log, và lớp nào có điểm cao hơn thì thắng:

$$\text{score}(y) = \ln P(y) + \sum_j x_j \ln P(j\mid y)$$

Xác suất cuối cùng lấy bằng softmax của hai điểm. Thuật toán không cần lặp: chỉ đếm một lần. Đầu vào là trọng số TF-IDF thay cho số lần đếm, công thức vẫn áp dụng được.

## 5. Đánh giá

Trên tập test (spam là lớp dương; TP, TN, FP, FN là bốn ô của ma trận nhầm lẫn):

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}, \qquad \text{Precision} = \frac{TP}{TP+FP}$$

$$\text{Recall} = \frac{TP}{TP+FN}, \qquad F_1 = \frac{2\cdot\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$

Precision cao nghĩa là ít email thường bị chặn nhầm, recall cao nghĩa là ít spam lọt qua. Kết quả nằm ở `evaluation/summary.md` và `reports/data.html`.

## 6. Thí nghiệm mở rộng (`model/script/tuning.py`)

- **Feature engineering**: so sánh TF-IDF unigram, unigram + bigram, và thêm đặc trưng ký tự (cùng dùng SVM).
- **Tuning**: `GridSearchCV` (3-fold, tối ưu F1) cho $C$ của Logistic Regression và SVM, $\alpha$ của Naive Bayes. Tham số được chọn bằng cross-validation trên tập train.
- **Ensemble**: Random Forest (200 cây) và soft voting (trung bình xác suất spam của các mô hình).

Các cấu hình đều có F1 quanh 0,99, riêng Random Forest thấp hơn một chút. Mô hình cơ bản đã gần mức tối đa trên bộ dữ liệu này nên tuning và ensemble không cải thiện thêm. Kết quả chi tiết: `evaluation/extra_experiments.md`.

## 7. Giới hạn

- Dữ liệu là email tiếng Anh nên không phù hợp với tiếng Việt; giao diện demo cảnh báo thay vì kết luận khi email chủ yếu không phải tiếng Anh hoặc quá ngắn.
- Ham chỉ nằm trong khoảng 1999–2002, còn gần như toàn bộ spam ở 2002–2005 (theo cột `Date`), nên mô hình có thể học một phần dấu hiệu thời đại.
- Email thông báo hiện đại (ngân hàng, ứng dụng) có thể bị nhận nhầm là spam, vì ham trong dữ liệu chỉ là thư công việc.
- Độ chính xác khoảng 99% có thể cao hơn thực tế vì bộ Enron-Spam khá dễ tách.
