# BÁO CÁO KẾT QUẢ HOÀN THÀNH KHÓA HỌC MOOC
## ĐIỂM CỘNG (10% BONUS) MÔN CƠ SỞ TRÍ TUỆ NHÂN TẠO - TRƯỜNG ĐH KHOA HỌC TỰ NHIÊN (HCMUS)

---

### PHẦN 1: THÔNG TIN CHUNG

* **Họ và tên sinh viên:** Phạm Trường Quỳnh
* **Mã số sinh viên (MSSV):** 24120426
* **Lớp:** CQ2024/23
* **Email sinh viên:** 24120426@student.hcmus.edu.vn
* **Môn học:** Cơ sở Trí tuệ Nhân tạo
* **Giảng viên phụ trách:** Thầy Bùi Duy Đăng
* **Tên khóa học MOOC:** [HarvardX CS109xa: Machine Learning and AI with Python](https://learning.edx.org/course/course-v1:HarvardX+CS109xa+2T2024/home)
* **Nền tảng đào tạo:** edX (Đại học Harvard - Harvard University)
* **Thời gian học:** Từ ngày **06/10/2026** đến ngày **17/11/2026**
* **Hình thức tham gia:** Audit Track (Học tập tự do, hoàn thành 100% bài giảng lý thuyết, đọc tài liệu, kiểm tra quiz và tự nghiên cứu)
* **Kết quả đạt được:** **78% Course Completion** (Mức hoàn thành tối đa 100% của chế độ Audit) kèm xác nhận chính thức từ edX: **"Congratulations! You have completed your course."**
* **Kho lưu trữ mã nguồn & Nhật ký học tập (GitHub):**  
  [https://github.com/PhamQuinh/HCMUS-CSAI-Bonus-HarvardX-CS109xa.git](https://github.com/PhamQuinh/HCMUS-CSAI-Bonus-HarvardX-CS109xa.git)

---

### PHẦN 2: MINH CHỨNG GHI DANH (ENROLLMENT EVIDENCE)

Để chứng minh quá trình học tập được bắt đầu đúng hạn từ đầu học kỳ (không phải học dồn vào cuối kỳ), dưới đây là các minh chứng ghi danh từ edX:

![Email xác nhận đăng ký khóa học](../evidence/enrollment/email_confirmation.png)  
*Hình 1: Email xác nhận ghi danh khóa học CS109xa từ edX gửi tới địa chỉ email sinh viên.*

![Minh chứng mốc thời gian Dates](../evidence/enrollment/dates_tab.png)  
*Hình 2: Giao diện tab "Dates" trên edX ghi nhận mốc thời gian bắt đầu chính thức: **Tue, Oct 6, 2026 - Enrollment Date**.*

---

### PHẦN 3: MINH CHỨNG TIẾN ĐỘ HỌC TẬP & HOÀN THÀNH KHÓA HỌC

Trong suốt quá trình học tập, tiến độ được hệ thống edX tự động cập nhật liên tục qua từng chương học.

![Tiến độ hoàn thành tối đa trên tab Progress](../evidence/progress/progress_final_completed_78pct.png)  
*Hình 3: Giao diện tab "Progress" ghi nhận tiến độ hoàn tất **78% Course completion** (mức hoàn thành tối đa 100% toàn bộ nội dung bài học trong chế độ Audit Track, hiển thị rõ URL edX, thời gian hệ thống và menu tài khoản chính chủ Phạm Trường Quỳnh - 24120426@student.hcmus.edu.vn).*

![Màn hình chúc mừng hoàn tất khóa học edX](../evidence/progress/course_completion_congratulations.png)  
*Hình 4: Màn hình vinh danh chính thức từ edX: **"Congratulations! You have completed your course."** (minh chứng cao nhất xác nhận đã hoàn tất toàn bộ chương trình học từ Introduction, Section 1 đến Section 6 và Wrap-up).*

#### Bảng tổng kết các mốc tiến độ thực tế:
| Mốc thời gian | Nội dung hoàn thành | Tiến độ ghi nhận |
| :---: | :--- | :---: |
| **06/10/2026** | Đăng ký khóa học trên edX (Enrollment Date) | **0%** |
| **19/10/2026** | Hoàn thành phần mở đầu (Introduction to ML & Python) | **5%** |
| **09/11/2026** | Hoàn thành Section 1: Decision Trees I (Splitting criteria) | **22%** |
| **11/11/2026** | Hoàn thành Section 2: Decision Trees II (Regression Trees & Pruning) | **30%** |
| **12/11/2026** | Hoàn thành Section 3: Bagging & Out-of-Bag Error (OOB) | **39%** |
| **15/11/2026** | Hoàn thành Section 4: Random Forest, Missing Data & Imbalanced Classes | **56%** |
| **16/11/2026** | Hoàn thành Section 5: Gradient Boosting & Optimization in Function Space | **70%** |
| **16/11/2026** | Hoàn thành Section 6: AdaBoost (Adaptive Boosting for Classification) | **76%** |
| **17/11/2026** | Hoàn thành Wrap-up, Survey & Tốt nghiệp toàn bộ khóa học | **78% (Max)** |

---

### PHẦN 4: TÓM TẮT QUÁ TRÌNH TỰ HỌC & BÀI HỌC KINH NGHIỆM

Toàn bộ 6 module của khóa học tập trung chuyên sâu vào **Họ mô hình Cây quyết định (Decision Trees) và Học kết hợp (Ensemble Learning)**. Dưới đây là những đúc kết học thuật cốt lõi do em tự ghi nhận:

#### 1. Section 1: Decision Trees I (Cây quyết định phân loại & Tiêu chuẩn rẽ nhánh)
* Nắm vững cơ chế phân nhánh nhị phân đệ quy (**Recursive Binary Splitting**) theo giải thuật tham lam (**Greedy Search**).
* Hiểu rõ và tự tính toán 3 độ đo độ tinh khiết:
  * **Classification Error Rate:** $E = 1 - \max_k(p_{mk})$ (dễ hiểu nhưng kém nhạy bén với sự thay đổi xác suất ở các nút lá).
  * **Gini Impurity:** $G = 1 - \sum p_{mk}^2$ (thước đo độ vẩn đục, tối ưu tốc độ tính toán cho thuật toán CART).
  * **Entropy & Information Gain:** $H = -\sum p_{mk} \log_2(p_{mk})$ (đo lường lượng thông tin theo lý thuyết Claude Shannon).
* Điểm lưu ý: Gini và Entropy nhạy cảm với xác suất hơn hẳn Error Rate, giúp cây phân tách thành các nút thuần khiết nhanh hơn.

#### 2. Section 2: Decision Trees II (Hồi quy, Mã hóa & Kỹ thuật Tỉa cây)
* **Cây hồi quy (Regression Trees):** Dự đoán giá trị liên tục bằng trung bình cộng các điểm trong lá ($\hat{y}_{R_m} = \frac{1}{|R_m|} \sum y_i$), tối thiểu hóa hàm mất mát Residual Sum of Squares (RSS).
* **Mã hóa biến định danh:** Phân biệt rõ One-Hot Encoding (tránh gán thứ tự giả tạo nhưng làm phình to số chiều) và Label Encoding.
* **Tỉa cây (Cost-Complexity Pruning):** Cắt tỉa các nhánh phụ bằng hàm mục tiêu phạt độ phức tạp $\text{Cost} = RSS + \alpha |T|$, giúp mô hình cân bằng tối ưu giữa Bias và Variance.

#### 3. Section 3: Bagging & Đánh giá Out-of-Bag (OOB)
* **Triết lý Ensemble Learning:** Kết hợp nhiều cây độc lập để giảm Variance (ổn định hóa mô hình).
* **Bootstrapping (Lấy mẫu ngẫu nhiên có hoàn lại):** Tạo ra các tập dữ liệu có kích thước $N$ từ tập gốc, trong đó xấp xỉ $63.2\%$ mẫu được chọn và khoảng $36.8\%$ mẫu bị bỏ sót.
* **Out-of-Bag Error (OOB):** Tận dụng chính các mẫu bị bỏ sót ngoài túi để kiểm thử chéo miễn phí (*free cross-validation*), tiết kiệm chi phí tính toán mà vẫn đánh giá khách quan độ tổng quát hóa.
* Hạn chế của Bagging: Nếu có một biến dự báo quá mạnh, các cây đều bị tương quan cao (**Correlated Trees**), làm giảm hiệu quả lấy trung bình.

#### 4. Section 4: Random Forest, Dữ liệu Khuyết & Mất cân bằng
* **Khử tương quan cây (Decorrelating Trees):** Random Forest giải quyết triệt để nhược điểm của Bagging bằng kỹ thuật **Feature Subspace Sampling** — tại mỗi nút chia chỉ chọn ngẫu nhiên $m$ thuộc tính ($m \approx \sqrt{p}$ cho phân loại, $m \approx p/3$ cho hồi quy).
* **Đánh giá đặc trưng:** So sánh MDI (Gini Importance, dễ bị thiên vị vào biến liên tục) và Permutation Feature Importance (xáo trộn ngẫu nhiên trên tập OOB, đo lường khách quan).
* **Surrogate Splits (Nhát cắt đóng thế):** Cơ chế xử lý dữ liệu khuyết thông minh của CART dựa trên tính đa cộng tuyến.
* **Dữ liệu mất cân bằng:** Tránh bẫy **Accuracy Paradox** bằng các chỉ số F1-Score, ROC-AUC; áp dụng 3 kỹ thuật: Undersampling (NearMiss), Oversampling (SMOTE - nội suy tuyến tính) và gán trọng số phạt `class_weight='balanced'`.

#### 5. Section 5: Gradient Boosting & Tối ưu hóa trong Không gian Hàm
* **Đột phá triết lý:** Khác với Bagging/Random Forest (dùng cây sâu để giảm Variance), Boosting đi từ các cây rất nông (**Decision Stumps** `max_depth = 1`) và huấn luyện tuần tự để **giảm Bias**.
* **Bản chất toán học:** Chứng minh phần dư chính là Âm Gradient ($-\frac{\partial L}{\partial \hat{y}} = y - \hat{y} = r$). Gradient Boosting thực chất là **thuật toán Gradient Descent được thực hiện trong Không gian Hàm (Function Space)**, trong đó mỗi cây con đóng vai trò là một vector chỉ đường xấp xỉ Gradient để dắt mô hình đi dần về đáy của hàm mất mát.
* Tinh chỉnh tốc độ học $\lambda$ (Shrinkage) và kiểm soát số cây $B$ để chống Overfitting.

#### 6. Section 6: AdaBoost (Adaptive Boosting for Classification)
* Biến thể Boosting chuyên biệt cho bài toán Phân loại nhị phân sử dụng nhãn $\{-1, +1\}$.
* **Cơ chế thích ứng trọng số mẫu (Adaptive Re-weighting):** Mẫu đoán sai bị nhân với $e^{+\lambda}$ (phóng to trọng số), mẫu đoán đúng nhân với $e^{-\lambda}$ (thu nhỏ trọng số).
* Công thức quyền biểu quyết $\lambda = \frac{1}{2}\ln(\frac{1-\epsilon}{\epsilon})$ được chứng minh là nghiệm giải tích cực tiểu hóa hàm mất mát **Exponential Loss**.
* Nắm vững các thư viện Boosting công nghiệp hàng đầu: **XGBoost, LightGBM, CatBoost**.

---

### PHẦN 5: ĐỐI CHIẾU KIẾN THỨC VỚI MÔN CƠ SỞ AI (HCMUS)

Quá trình học khóa học HarvardX CS109xa giúp em củng cố và mở rộng sâu sắc các kiến thức nền tảng trong môn **Cơ sở Trí tuệ Nhân tạo** tại trường:

1. **Tìm kiếm Không gian Trạng thái (State Space Search) & Heuristic:**
   * Quá trình xây dựng Cây quyết định từ nút gốc đến các lá thực chất là một bài toán **Tìm kiếm theo đồ thị/cây**. Thuật toán rẽ nhánh CART áp dụng chiến lược **Tìm kiếm tham lam (Greedy Search)**, sử dụng độ đo Information Gain / Gini Gain làm **Hàm Heuristic** để lựa chọn nhánh đi giảm độ bất định nhiều nhất tại mỗi bước.
2. **Thuật toán Tối ưu hóa Cục bộ (Local Search & Optimization):**
   * Các khái niệm Hạ độ dốc (Gradient Descent), Tốc độ học (Learning Rate), Cực tiểu địa phương (Local Minima) và Cực tiểu toàn cục (Global Minima) trong Section 5 liên hệ trực tiếp với các giải thuật **Leo đồi (Hill Climbing)** và **Mô phỏng luyện kim (Simulated Annealing)** trong chương trình AI trên lớp. Điểm đột phá là khóa học đã nâng tầm từ tối ưu hóa biến số lên tối ưu hóa trong **Không gian Hàm giả thuyết**.
3. **Cân bằng Bias – Variance & Khả năng Tổng quát hóa:**
   * Môn Cơ sở AI nhấn mạnh vào tính tổng quát hóa của hệ tri thức. Hai trường phái **Bagging (giảm Variance)** và **Boosting (giảm Bias)** giúp em hiểu rõ bản chất cách dung hòa giữa việc mô hình học đủ phức tạp để nắm bắt quy luật thế giới thực nhưng không bị rơi vào bẫy học vẹt (Overfitting).
4. **Học máy trong Trí tuệ Nhân tạo hiện đại:**
   * Khóa học bổ sung mảnh ghép thực chiến hoàn hảo cho học phần Học có giám sát (Supervised Learning), từ cách tiền xử lý dữ liệu khuyết, xử lý mất cân bằng lớp đến việc triển khai các mô hình Ensemble hiện đại nhất thế giới.

---

### PHẦN 6: KẾT LUẬN & LIÊM CHÍNH HỌC THUẬT

1. **Cam kết liêm chính học thuật (Academic Integrity):**
   * Toàn bộ quá trình học tập được em thực hiện nghiêm túc từ ngày 06/10/2026 đến ngày 17/11/2026.
   * Minh chứng tiến độ và các mốc thời gian commit trên GitHub phản ánh trung thực quá trình nghiên cứu thực tế.
   * Các nội dung ghi chép tại các file `learning-logs/` và báo cáo này đều do chính em tự tổng hợp, đúc kết theo cách hiểu của bản thân, không sao chép hay dùng AI tạo nội dung văn bản suông.
2. **Nguyện vọng:**
   * Kính mong Thầy Bùi Duy Đăng xem xét hồ sơ minh chứng và công nhận kết quả hoàn thành khóa học để cộng **10% điểm bonus** môn Cơ sở Trí tuệ Nhân tạo.

**Sinh viên thực hiện:**  
*Phạm Trường Quỳnh (MSSV: 24120426)*
