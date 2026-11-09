# NHẬT KÝ HỌC TẬP - SECTION 1: DECISION TREES (PHẦN 1)
* **Thời gian học:** 09/11/2026
* **Chủ đề bài học:** Giới thiệu Cây quyết định (Tree Anatomy) & Tiêu chí phân nhánh (Splitting Criteria: Classification Error, Gini Index, Entropy)

---

### 1. Kiến thức cốt lõi tiếp thu được
- **Cấu trúc cây quyết định:**
  - Nút gốc (Root node) và các nút nội (Internal nodes): Đại diện cho các điều kiện kiểm tra ngưỡng của từng thuộc tính.
  - Nút lá (Leaf nodes): Đưa ra nhãn dự đoán cuối cùng (lấy theo lớp chiếm đa số - majority vote).
  - Ý nghĩa hình học: Mỗi lần phân nhánh tương đương với việc kẻ một nhát cắt vuông góc với trục toạ độ để chia nhỏ không gian đặc trưng.
- **Động lực sử dụng Cây quyết định (So với Logistic Regression):**
  - Hồi quy Logistic chỉ hiệu quả khi ranh giới phân chia là tuyến tính (đường thẳng/mặt phẳng). Khi dùng đa thức phi tuyến bậc cao, mô hình trở nên rất khó giải thích.
  - Cây quyết định vừa xử lý được ranh giới phi tuyến phức tạp, vừa cực kỳ trực quan, dễ giải thích tương tự lưu đồ tư duy của con người (Engineering Flowchart).
- **3 Tiêu chí phân nhánh đánh giá độ vẩn đục (Impurity):**
  - *Classification Error:* $1 - \max(p_k)$. Đo tỷ lệ đoán sai của lớp thiểu số, nhưng nhược điểm là không khả vi và ít nhạy.
  - *Gini Index:* $1 - \sum p_k^2$. Bình phương tỷ lệ giúp lớp đa số áp đảo tổng, làm hàm số khả vi và trơn tru.
  - *Entropy (Information Theory):* $- \sum p_k \log_2(p_k)$. Đo mức độ hỗn loạn của phân phối nhãn trong vùng. Vùng hoàn toàn thuần khiết có Entropy = 0.

---

### 2. Điểm kỹ thuật / Bản chất toán học sâu sắc
- **Tại sao lại dùng thuật toán Tham lam (Greedy Algorithm)?**
  - Việc tìm ra cây quyết định tối ưu toàn cục cho một tập dữ liệu là bài toán **NP-Complete** (không thể giải trong thời gian đa thức).
  - Vì vậy, thuật toán phải áp dụng chiến lược tham lam: Tại mỗi bước, chọn thuộc tính và ngưỡng cắt làm giảm độ vẩn đục (hoặc tăng lượng thông tin thu được) nhiều nhất tại thời điểm đó.
- **Mức độ trừng phạt độ lẫn lộn:**
  - Khi dữ liệu hoàn toàn thuần khiết ($p=0$ hoặc $p=1$), cả 3 chỉ số đều bằng 0.
  - Nhưng khi dữ liệu bị pha tạp, Entropy trừng phạt mạnh nhất, kế đến là Gini Index, và nhẹ nhất là Classification Error.

---

### 3. Mối liên hệ với môn Cơ sở AI (HCMUS)
- Rất sát với chương **Học máy (Machine Learning)** và giải thuật **ID3** trong chương trình Cơ sở AI của trường HCMUS.
- Quá trình phân nhánh tại mỗi nút thực chất là một dạng **Tìm kiếm cục bộ / Tham lam (Greedy Search)**.
- Khái niệm giảm Entropy chính là lượng thông tin thu được (**Information Gain** - độ lợi thông tin) được dùng làm hàm heuristic để chọn thuộc tính phân chia tốt nhất.
