# NHẬT KÝ HỌC TẬP - SECTION 2: REGRESSION TREES, ENCODING & PRUNING
* **Thời gian học:** 11/11/2026
* **Chủ đề bài học:** Cây quyết định cho bài toán Hồi quy (Regression Trees), Xử lý biến phân loại (Categorical Encoding) & Kỹ thuật cắt tỉa cây (Pruning)

---

### 1. Kiến thức cốt lõi tiếp thu được
- **Cây quyết định cho Hồi quy (Regression Trees):**
  - Khi đầu ra là số thực liên tục, thay vì bầu chọn đa số (majority vote), nút lá sẽ dự đoán bằng **giá trị trung bình ($\bar{y}_l$)** của các mẫu rơi vào vùng đó (tạo thành đường dự đoán dạng bậc thang).
  - Tiêu chí phân nhánh: Thay Gini/Entropy bằng **Weighted MSE** (chọn nhát cắt làm tổng sai số bình phương trung bình có trọng số nhỏ nhất).
  - Điều kiện dừng tương tự cây phân loại (`max_depth`, `min_samples_leaf`, `max_leaf_nodes`), nhưng dùng độ giảm phương sai (**Reduction in Variance / MSE Reduction**) thay cho độ lợi thuần khiết.
- **Xử lý dữ liệu hỗn hợp (Số và Chữ / Categorical Features):**
  - Thuật toán CART trong `scikit-learn` chưa hỗ trợ trực tiếp biến chữ, bắt buộc phải mã hóa sang số trước khi huấn luyện.
  - *Ordinal Encoding* (gán 0, 1, 2): Dễ tạo ra thứ tự giả tạo cho các biến không có thứ tự (như màu sắc), khiến các nhát cắt đơn không thể tách riêng giá trị nằm ở giữa.
  - *One-Hot Encoding & Dummy Encoding*: Tách thành các cột nhị phân 0/1 riêng biệt. Với *Dummy Encoding*, ta có thể lược bỏ 1 cột dư thừa (ví dụ có $C_1, C_2$ thì không cần $C_3$) để tiết kiệm chiều dữ liệu.
- **Kỹ thuật cắt tỉa cây (Cost Complexity Pruning):**
  - Khắc phục nhược điểm dừng quá sớm (Underfit) hoặc quá muộn (Overfit) của Stopping Conditions bằng cách cho cây mọc tối đa ($T_0$) rồi tỉa ngược từ dưới lên.
  - Hàm chi phí đánh đổi: $C(T) = \text{Error}(T) + \alpha |T|$, trong đó $|T|$ là số lượng nút lá và $\alpha$ là hệ số phạt độ phức tạp (`ccp_alpha` trong `sklearn`).
  - Ở mỗi bước tỉa, thuật toán tìm "mắt xích yếu nhất" — nhánh nào cắt đi làm sai số tăng lên ít nhất trên mỗi lá giảm được.

---

### 2. Điểm kỹ thuật & Lưu ý thực tế
- **Hạn chế của Regression Tree:** Vì dự đoán dựa trên trung bình mẫu có sẵn trong tập huấn luyện, Cây hồi quy **không có khả năng ngoại suy (cannot extrapolate)** ra ngoài khoảng giá trị Min–Max của tập train.
- **Bẫy khi chọn siêu tham số $\alpha$:** Dùng **Cross-Validation** để chọn $\alpha$ tối ưu, nhưng phải dựa trên **sai số của tập Validation (Validation Error)**, tuyệt đối không chọn theo điểm số $C(T)$ trên tập train.

---

### 3. Mối liên hệ với môn Cơ sở AI (HCMUS)
- Kỹ thuật Cost Complexity Pruning thực chất là áp dụng **Điều chuẩn (Regularization)** cho cây quyết định — tương đồng với cách kiểm soát độ phức tạp mô hình để cân bằng Bias–Variance trong chương Học máy.
- Chuỗi tỉa cây từng bước ($T_0 \rightarrow T_1 \rightarrow \dots \rightarrow T_L$) thể hiện chiến lược tối ưu tham lam ngược (Bottom-up Greedy Pruning) nhằm thu nhỏ không gian giả thuyết mà vẫn giữ hiệu năng tổng quát hóa tốt nhất.
