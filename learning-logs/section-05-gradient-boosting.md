# NHẬT KÝ HỌC TẬP - SECTION 5: GRADIENT BOOSTING & BẢN CHẤT TỐI ƯU HÓA
* **Thời gian học:** 16/11/2026
* **Chủ đề bài học:** Thuật toán Boosting, Đánh đổi Bias-Variance (Bagging vs Boosting), Cơ chế học trên phần dư (Residuals), Bản chất tối ưu hóa Gradient Descent trong không gian hàm (Function Space), Tốc độ học $\lambda$ (Shrinkage) & Điều kiện dừng (Termination)

---

### 1. Kiến thức cốt lõi tiếp thu được

#### 1.1. Triết lý của Boosting: Biến mô hình yếu thành mô hình mạnh
- **Câu hỏi của GS. Leslie Valiant:** Liệu một tập hợp các mô hình học yếu (*weak learners*) có thể kết hợp lại để tạo nên một mô hình học mạnh (*strong learner*) hay không?
- **Sự khác biệt cốt lõi giữa Bagging và Boosting:**
  - *Bagging / Random Forest:* Đi từ các cây **rất sâu** (Low Bias, High Variance $\rightarrow$ Overfitting), sử dụng lấy mẫu Bootstrap và lấy trung bình để **giảm Variance**.
  - *Boosting:* Đi từ các cây **rất nông / đơn giản** (High Bias, Low Variance $\rightarrow$ Underfitting), huấn luyện tuần tự để **giảm Bias** bằng cách liên tục học từ các sai lầm của mô hình đi trước.
- **Tổ hợp cộng tính tuyến tính (Additive Model):**
  $$T(x) = \sum_{k=1}^B \lambda_k T_k(x)$$
  Trong đó các mô hình con $T_k$ được kết hợp theo tổng có trọng số tin cậy.

#### 1.2. Cơ chế vận hành của Gradient Boosting (Bài toán Hồi quy)
1. **Khởi tạo:** Huấn luyện cây đơn giản ban đầu $T^{(0)}$ (chỉ 1 nhát cắt - *Decision Stump*) trên dữ liệu gốc $(x_i, y_i)$.
2. **Tính phần dư ban đầu (Residuals):** $r_i = y_i - T^{(0)}(x_i)$.
3. **Học trên phần dư:** Huấn luyện cây con tiếp theo $T^{(1)}$ bằng cách gán đặc trưng đầu vào là $x$ và nhãn mục tiêu chính là **phần dư $r$** (thay vì $y$).
4. **Cập nhật mô hình tổng hợp:** 
   $$T \leftarrow T + \lambda T^{(1)}$$
   (với $\lambda$ là tốc độ học / Shrinkage).
5. **Cập nhật nhanh phần dư:**
   $$r_n \leftarrow r_n - \lambda T^{(i)}(x_n)$$
   Lặp lại quy trình cho đến khi thỏa mãn điều kiện dừng. Máy tính không xây dựng cây sâu phức tạp mà chỉ đơn giản cộng dồn dự đoán của các cây con.

#### 1.3. Bản chất toán học: Gradient Descent trong Không gian Hàm (Function Space)
- **Bí mật của phần dư:** Với hàm mất mát bình phương $L = \frac{1}{2}(y - \hat{y})^2$, đạo hàm theo giá trị dự đoán là:
  $$-\frac{\partial L}{\partial \hat{y}} = y - \hat{y} = \mathbf{r}$$
  $\rightarrow$ **Phần dư chính là Âm Gradient (Negative Gradient)** — hướng dốc xuống nhanh nhất của hàm mất mát!
- **Nghịch lý Không gian Dự đoán & Giải pháp Không gian Hàm:**
  - Nếu tối ưu trực tiếp trên không gian dự đoán $[\hat{y}_1, \dots, \hat{y}_N]$, thuật toán sẽ rơi vào nghiệm tầm thường $\hat{y}_i = y_i$ (học vẹt, không có khả năng dự đoán điểm mới).
  - Vì cây quyết định là mô hình **phi tham số (Non-parametric)**, ta không thể tối ưu hóa vector trọng số $w$.
  - **Đột phá của Gradient Boosting:** Chiếu vector âm gradient lý tưởng $t^*$ lên họ các cây đơn giản $\mathbb{H}$ bằng cách huấn luyện cây con $t \in \mathbb{H}$ bám sát nhất có thể với vector phần dư.

#### 1.4. Tốc độ học $\lambda$ (Shrinkage) & Điều kiện dừng
- $\lambda$ tương đương với **Learning Rate** trong Gradient Descent:
  - $\lambda$ quá nhỏ: Học rất chậm, tốn nhiều cây tính toán.
  - $\lambda$ quá lớn: Bước nhảy vọt qua đáy, mô hình dao động mạnh và có thể phân kỳ (*diverge*).
  - Quy tắc kinh nghiệm (*rule of thumb*): $\lambda \in [10^{-3}, 10^{-2}]$.
- **Điều kiện dừng:** Giới hạn số lượng cây $B$ hoặc dừng khi mức cải thiện sai số rơi xuống dưới ngưỡng $\epsilon$. Khác với Random Forest, Boosting **hoàn toàn có thể bị Overfitting** nếu số lượng cây $B$ tăng quá lớn!

---

### 2. Điểm kỹ thuật & Đúc kết thực chiến
- Gradient Boosting cơ bản là thuật toán **tất định (Deterministic)**: Chạy nhiều lần với cùng siêu tham số trên cùng tập dữ liệu sẽ luôn cho kết quả giống hệt nhau (khác với Random Forest có yếu tố ngẫu nhiên Bootstrap).
- Với hàm mất mát lồi (Convex), Gradient Descent đảm bảo hội tụ về cực tiểu toàn cục (Global Minimum). Với hàm không lồi (Non-convex), cần áp dụng các kỹ thuật như khởi tạo lại ngẫu nhiên (Random Restarts) hoặc Stochastic Gradient Descent (SGD) để tránh bẫy cực tiểu địa phương (Local Minima).

---

### 3. Mối liên hệ với môn Cơ sở AI (HCMUS)
- Mở rộng tư duy của các thuật toán Tìm kiếm cục bộ (**Local Search**) như Leo đồi (Hill Climbing) và Hạ độ dốc (Gradient Descent) từ không gian biến số thông thường sang **không gian hàm giả thuyết (Function Space)**.
- Thấy rõ sự tiến hóa của các giải thuật Học máy: Từ cây đơn lẻ $\rightarrow$ Tổ hợp song song giảm phương sai (Bagging / Random Forest) $\rightarrow$ Tổ hợp tuần tự giảm độ chệch (Gradient Boosting) thống trị các bài toán dữ liệu bảng (Tabular data) hiện đại.
