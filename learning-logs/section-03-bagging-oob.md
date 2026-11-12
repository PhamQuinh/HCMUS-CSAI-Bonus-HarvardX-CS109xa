# NHẬT KÝ HỌC TẬP - SECTION 3: BAGGING & OUT-OF-BAG ERROR
* **Thời gian học:** 12/11/2026
* **Chủ đề bài học:** Học kết hợp (Ensemble Learning), Lấy mẫu Bootstrap, Thuật toán Bagging & Đánh giá Out-of-Bag Error (OOB)

---

### 1. Kiến thức cốt lõi tiếp thu được
- **Học kết hợp (Ensemble Learning):**
  - Thay vì phụ thuộc vào một cây quyết định đơn lẻ (dễ bị Overfit khi sâu hoặc Underfit khi nông), ta huấn luyện nhiều mô hình độc lập rồi tổng hợp ý kiến:
    - *Phân loại (Classification):* Lấy nhãn theo số đông (**Majority Vote**).
    - *Hồi quy (Regression):* Lấy giá trị trung bình (**Average**) của tất cả các cây.
- **Kỹ thuật Bootstrapping & Bagging (Bootstrap Aggregating):**
  - *Bootstrapping:* Tạo ra nhiều tập dữ liệu mới bằng cách **lấy mẫu ngẫu nhiên có hoàn lại (sampling with replacement)** từ tập gốc. Mỗi tập Bootstrap có kích thước $N$ bằng tập gốc nhưng có các điểm bị lặp lại và một số điểm bị bỏ sót.
  - *Bagging:* Huấn luyện các cây quyết định sâu độc lập trên từng tập Bootstrap rồi tổng hợp kết quả $\rightarrow$ Giúp mô hình giữ được sức biểu diễn biên phức tạp (**High Expressiveness**) đồng thời giảm mạnh phương sai (**Low Variance**), làm đường dự đoán mượt mà hơn hẳn.
- **Đánh giá Out-of-Bag Error (OOB):**
  - Tận dụng chính những điểm dữ liệu bị bỏ sót ngoài túi Bootstrap (*Out-of-bag*) để kiểm thử ngược lại những cái cây chưa từng thấy chúng trong lúc huấn luyện.
  - Với mỗi điểm $x_i$, chỉ dùng các cây không chứa $x_i$ để dự đoán $\hat{y}_{i,pw}$, tính sai số điểm (*pointwise error*) rồi lấy trung bình toàn bộ tập huấn luyện.

---

### 2. Điểm kỹ thuật & Hạn chế cần lưu ý
- **Ưu điểm của OOB so với Cross-Validation (CV):** Tiết kiệm chi phí tính toán (chỉ train 1 lần), tận dụng 100% tập dữ liệu (rất hữu ích khi dữ liệu ít) và tránh hiện tượng rò rỉ dữ liệu trong quá trình đánh giá.
- **3 Hạn chế còn tồn đọng của Bagging:**
  1. *Mất tính diễn giải (Interpretability):* Không thể nhìn vào 1 cái cây để đọc luật quyết định rõ ràng như cây đơn lẻ.
  2. *Cây con quá nông vẫn Underfit:* Kết hợp nhiều cây yếu (`depth` thấp) vẫn không giảm được Bias.
  3. *Cây sâu vẫn có thể Overfit (High Correlation):* Nếu tập dữ liệu có một thuộc tính $x_j$ quá mạnh, mọi cây trong Bagging đều có xu hướng chọn $x_j$ làm nút gốc $\rightarrow$ Các cây mọc ra giống hệt nhau (tương quan cao), làm giảm hiệu quả lấy trung bình. (Đây là lý do cần nâng cấp lên **Random Forest**).

---

### 3. Mối liên hệ với môn Cơ sở AI (HCMUS)
- Kỹ thuật Bagging giải quyết bài toán cân bằng **Bias–Variance** trong Học máy bằng cách kết hợp nhiều giả thuyết phức tạp nhằm triệt tiêu nhiễu ngẫu nhiên (giảm Variance).
- Hạn chế "tương quan cao giữa các cây" (Correlated Trees) giải thích bản chất tại sao việc đa dạng hóa không gian tìm kiếm (Feature Subspace Sampling trong Random Forest ở bài sau) lại quan trọng trong các thuật toán học máy hiện đại.
