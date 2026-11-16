# NHẬT KÝ HỌC TẬP - SECTION 6: ADABOOST (ADAPTIVE BOOSTING) & HOÀN TẤT KHÓA HỌC
* **Thời gian học:** 16/11/2026 - 17/11/2026
* **Chủ đề bài học:** Thuật toán AdaBoost (Adaptive Boosting), Phân loại với Decision Stumps, Cơ chế điều chỉnh trọng số mẫu (Sample Re-weighting), Bản chất tối ưu hóa Exponential Loss, So sánh toàn diện Gradient Boosting vs AdaBoost, và Bộ ba thư viện công nghiệp (XGBoost, LightGBM, CatBoost)

---

### 1. Kiến thức cốt lõi tiếp thu được

#### 1.1. Triết lý & Hai tư tưởng chính của AdaBoost
- **Khái niệm:** AdaBoost (Adaptive Boosting) do Yoav Freund và Robert Schapire phát minh (1996), là biến thể Boosting chuyên biệt cho bài toán Phân loại (Classification).
- **Hai tư tưởng cốt lõi:**
  1. *Kết hợp tuần tự các bộ học yếu (Weak Learners):* Sử dụng các gốc cây quyết định (**Decision Stumps**) chỉ gồm 1 nút rẽ nhánh và 2 lá (`max_depth = 1`).
  2. *Học từ sai lầm bằng cách thích ứng trọng số mẫu (Adaptive Re-weighting):* Cây sau sẽ tập trung vào những mẫu mà cây trước phân loại sai bằng cách gán trọng số lớn hơn cho các mẫu đó.

#### 1.2. Kỹ thuật quy ước nhãn $\{-1, +1\}$
- Sử dụng nhãn $y \in \{-1, +1\}$ mang lại ý nghĩa đại số học tuyệt đẹp:
  - Nếu đoán **ĐÚNG**: $y_n \cdot S(x_n) = +1 > 0 \rightarrow$ Trọng số nhân với $e^{-\lambda} < 1$ (thu nhỏ trọng số).
  - Nếu đoán **SAI**: $y_n \cdot S(x_n) = -1 < 0 \rightarrow$ Trọng số nhân với $e^{+\lambda} > 1$ (phóng to trọng số).
- Điều này tự động hướng sự chú ý của gốc cây tiếp theo vào các điểm dữ liệu khó phân loại nhất.

#### 1.3. Quy trình tính toán & Công thức quyền biểu quyết $\lambda$
1. Khởi tạo trọng số ban đầu: $w_n^{(0)} = \frac{1}{N}$.
2. Với mỗi vòng lặp $i = 0, 1, \dots, B$:
   - Huấn luyện Decision Stump $S^{(i)}$ bằng chỉ số *Weighted Gini Index*.
   - Tính tổng sai số có trọng số:
     $$\epsilon^{(i)} = \sum_{n=1}^N w_n^{(i)} \mathbb{I}(y_n \neq S^{(i)}(x_n))$$
   - Tính hệ số tin cậy (quyền biểu quyết) của cây con:
     $$\lambda^{(i)} = \frac{1}{2} \ln \left( \frac{1 - \epsilon^{(i)}}{\epsilon^{(i)}} \right)$$
     *(Nếu $\epsilon \approx 0 \rightarrow \lambda$ rất lớn; nếu $\epsilon = 0.5 \rightarrow \lambda = 0$).*
   - Cập nhật và chuẩn hóa trọng số mẫu cho vòng lặp sau:
     $$w_n^{(i+1)} = \frac{w_n^{(i)} \exp(-\lambda^{(i)} y_n S^{(i)}(x_n))}{Z_i}$$
3. **Mô hình phán quyết cuối cùng:** Lấy hàm dấu (`sign`) của tổng các cây có trọng số:
   $$T(x) = \text{sign} \left[ \sum_{i=0}^B \lambda^{(i)} S^{(i)}(x) \right]$$

#### 1.4. Nền tảng toán học: AdaBoost là Gradient Descent trên Exponential Loss
- Khác với Gradient Boosting tối thiểu hóa MSE, AdaBoost tối thiểu hóa **Exponential Loss**:
  $$\text{Exp Loss} = \frac{1}{N} \sum_{n=1}^N \exp(-y_n \hat{y}_n)$$
- Đạo hàm riêng của Exponential Loss theo $\hat{y}_n$ chính là:
  $$\nabla_{\hat{y}} \text{Exp Loss} = [-y_1 w_1, -y_2 w_2, \dots, -y_N w_N]$$
- Do đó, việc cập nhật trọng số mẫu $w_n$ trong AdaBoost thực chất chính là **bước đi theo hướng âm gradient** của hàm Exponential Loss trong không gian hàm!

#### 1.5. Bộ ba thư viện Boosting công nghiệp hàng đầu
- **XGBoost:** Tối ưu hóa bộ nhớ đệm phần cứng, hỗ trợ tính toán phân tán song song, tích hợp sẵn Regularization chống Overfit.
- **LightGBM (Microsoft):** Phân nhánh theo lá (*Leaf-wise split*) và thuật toán Histogram, tốc độ huấn luyện nhanh nhất trên dữ liệu lớn.
- **CatBoost (Yandex):** Tự động xử lý tối ưu các biến phân loại (*Categorical features*), giảm thiểu rò rỉ dữ liệu mục tiêu.

---

### 2. Điểm kỹ thuật & So sánh đối chiếu toàn diện
- **Gradient Boosting vs AdaBoost:**
  - *Gradient Boosting:* Tối ưu MSE cho hồi quy, cây con khớp trực tiếp trên phần dư liên tục ($r_n$).
  - *AdaBoost:* Tối ưu Exponential Loss cho phân loại, cây con khớp trên dữ liệu được gán trọng số mẫu rời rạc ($w_n$).
- **Hiện tượng Overfitting:** Cả Gradient Boosting và AdaBoost đều có thể bị Overfitting nếu số lượng cây $B$ (`n_estimators`) tăng quá lớn. Bắt buộc phải sử dụng Cross-Validation hoặc Early Stopping để kiểm soát.

---

### 3. Mối liên hệ với môn Cơ sở AI (HCMUS)
- Thuật toán AdaBoost kết hợp nhuần nhuyễn giữa **Lý thuyết Học PAC (Probably Approximately Correct Learning)** của Leslie Valiant với các phương pháp tối ưu giải tích.
- Minh chứng sinh động cho việc biến các bộ phân loại tuyến tính đơn giản thành một bộ phân loại phi tuyến phức tạp với đường biên quyết định mềm mại, hoàn thiện bức tranh toàn cảnh về Họ thuật toán Cây quyết định và Học kết hợp (Ensemble Learning) trong Trí tuệ Nhân tạo.
