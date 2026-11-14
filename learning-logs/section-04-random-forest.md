# NHẬT KÝ HỌC TẬP - SECTION 4: RANDOM FOREST, MISSING DATA & IMBALANCED DATASETS
* **Thời gian học:** 14/11/2026 - 15/11/2026
* **Chủ đề bài học:** Rừng ngẫu nhiên (Random Forest), Khử tương quan cây quyết định (Decorrelation), Đánh giá tầm quan trọng đặc trưng (Feature Importance), Xử lý dữ liệu khuyết (Surrogate Splits) & Kỹ thuật xử lý dữ liệu mất cân bằng (Imbalanced Classes)

---

### 1. Kiến thức cốt lõi tiếp thu được

#### 1.1. Bản chất của Random Forest & Cơ chế khử tương quan (Decorrelating Trees)
- **Động lực:** Ở Section 3 (Bagging), nếu tập dữ liệu có một đặc trưng quá mạnh ($x_j$), hầu hết các cây đều chọn $x_j$ làm nút gốc $\rightarrow$ Các cây bị tương quan cao (**Correlated Trees**), làm giảm hiệu quả giảm phương sai khi lấy trung bình.
- **Giải pháp của Random Forest:** Tại mỗi bước phân nhánh của từng cây, thuật toán chỉ chọn ngẫu nhiên một tập con gồm $m$ thuộc tính trong tổng số $p$ thuộc tính có sẵn (**Feature Subspace Sampling**):
  - *Bài toán Phân loại (Classification):* $m \approx \sqrt{p}$
  - *Bài toán Hồi quy (Regression):* $m \approx p / 3$
- **Đặc tính mô hình:** Tăng số lượng cây $B$ (`n_estimators`) giúp đường sai số Out-of-Bag (OOB Error) hội tụ ổn định mà **không làm mô hình bị Overfitting**.

#### 1.2. Đánh giá tầm quan trọng của đặc trưng (Feature Importance)
- **MDI (Mean Decrease in Impurity / Gini Importance):**
  - Tính tổng độ giảm phương sai (Regression) hoặc Gini impurity (Classification) mà thuộc tính đó mang lại trên toàn bộ các cây.
  - *Hạn chế chí mạng:* Rất dễ bị thiên vị đối với các biến liên tục (*numerical*) hoặc các biến phân loại có quá nhiều mức giá trị phân biệt (*high cardinality*).
- **Permutation Feature Importance (PFI):**
  - Xáo trộn ngẫu nhiên thứ tự giá trị của một thuộc tính trên tập OOB (hoặc tập Validation) và đo lường mức độ tụt giảm hiệu năng của mô hình.
  - Phản ánh trung thực, khách quan hơn nhiều so với MDI và áp dụng được cho bất kỳ mô hình hộp đen nào.

#### 1.3. Xử lý dữ liệu bị khuyết bằng Surrogate Splits (Nhát cắt đóng thế)
- Thay vì xóa bỏ dòng dữ liệu khuyết hoặc điền giá trị trung bình thô sơ (*Mean Imputation*), thuật toán CART tìm các thuộc tính có phân phối rẽ nhánh tương đồng nhất với thuộc tính chính để làm **người đóng thế (Surrogate)**.
- Khi gặp mẫu dữ liệu mới bị khuyết giá trị ở thuộc tính chính, cây lập tức gọi thuộc tính đóng thế ra thay thế để quyết định rẽ nhánh.
- **Điều kiện phát huy tối đa:** Khi dữ liệu có tính đa cộng tuyến cao (**Multicollinearity**) — các biến có liên kết tự nhiên mật thiết với nhau (ví dụ: Tắc động mạch đi kèm tức ngực, hoặc Chiều cao đi kèm Cân nặng).

#### 1.4. Xử lý dữ liệu mất cân bằng (Imbalanced Classes)
- **Cái bẫy Accuracy Paradox:** Trong bài toán phát hiện gian lận hoặc chẩn đoán ca bệnh hiếm (như xét nghiệm COVID), một mô hình đoán "mọi người đều âm tính" vẫn có thể đạt Accuracy $99.1\%$ nhưng hoàn toàn vô giá trị trong thực tế vì bỏ lọt toàn bộ bệnh nhân.
- **Thước đo thay thế chuẩn xác:**
  - *Recall (True Positive Rate - TPR):* $\frac{TP}{TP + FN}$ — Đánh giá khả năng bắt trọn vẹn các ca dương tính thực sự.
  - *Precision:* $\frac{TP}{TP + FP}$ — Đánh giá độ tin cậy khi mô hình đã phát cảnh báo dương tính.
  - *F1-Score:* Trung bình điều hòa giữa Precision và Recall ($2 \cdot \frac{P \cdot R}{P + R}$), ngăn chặn việc một chỉ số cao che giấu chỉ số thấp.
  - *ROC Curve & AUC:* Thể hiện sự đánh đổi giữa TPR và FPR qua mọi ngưỡng phân loại. Mô hình phân loại ngẫu nhiên có $AUC = 0.5$, mô hình hoàn hảo đạt $AUC = 1.0$.
- **3 Chiến lược giải quyết mất cân bằng:**
  1. *Undersampling (NearMiss):* Chỉ giữ lại các mẫu đa số nằm sát ranh giới phân định nhất thay vì xóa ngẫu nhiên.
  2. *Oversampling (SMOTE):* Nội suy tuyến tính trên đoạn thẳng nối các điểm thiểu số lân cận để tạo ra các điểm dữ liệu mới, làm đầy không gian biểu diễn.
  3. *Class Weighting:* Gán trọng số phạt tỉ lệ nghịch với tần suất lớp: $W_k = \frac{N}{K \cdot N_k}$ (Scikit-Learn: `class_weight='balanced'`).

---

### 2. Điểm kỹ thuật & Đúc kết thực chiến
- Việc khử tương quan cây trong Random Forest giúp đường biên phân loại mềm mại hơn rất nhiều so với Bagging thông thường.
- Tuyệt đối không dựa vào Accuracy khi tỷ lệ mẫu giữa các lớp chênh lệch lớn; bắt buộc phải vẽ ma trận nhầm lẫn (Confusion Matrix), theo dõi F1-Score và đường cong ROC.
- Phương pháp gán trọng số `class_weight='balanced'` là giải pháp nhanh gọn và tiết kiệm bộ nhớ nhất vì không làm tăng kích thước tập dữ liệu.

---

### 3. Mối liên hệ với môn Cơ sở AI (HCMUS)
- Cơ chế Feature Subspace Sampling trong Random Forest thể hiện tư duy **ngẫu nhiên hóa không gian tìm kiếm (Randomized Search)** nhằm thoát khỏi các cực tiểu địa phương và tránh thiên vị vào các thuộc tính cục bộ quá nổi bật.
- Các khái niệm Precision, Recall, F1 và xử lý phân phối dữ liệu lệch là nền tảng cốt lõi trong học máy có giám sát (Supervised Learning), bổ trợ trực tiếp cho phần Học máy và Tìm kiếm thông minh trong chương trình Cơ sở Trí tuệ Nhân tạo.
