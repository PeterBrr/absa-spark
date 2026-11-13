# Thiết kế: Bài tập ABSA tiếng Việt trên Spark (nhóm 2 người)

Ngày: 2026-11-13 · Thời hạn: 1 tuần (7 ngày) · Môi trường: Kaggle Notebook

## 1. Mục đích

Bài tập môn học (10 điểm, 5 notebook × 2 điểm). Nhóm 2 người (gọi là **A** và **B**) cần xây dựng 2 mô hình phân loại bình luận theo khía cạnh (Aspect-Based Sentiment Analysis), huấn luyện và lưu mô hình, đóng gói mô hình thành UDF của Spark (bằng `pandas_udf` và `predict_batch_udf`), áp dụng lên Spark DataFrame và đánh giá bằng evaluator của Spark. Notebook thứ 5 thử nghiệm cách lấy vector đặc trưng và hàm loss để tìm mô hình tốt nhất.

Mã nguồn quản lý trên GitHub để hai người phối hợp, theo dõi tiến độ, và có một nơi lưu bản cuối.

## 2. Mục tiêu / Không làm / Thất bại

**Mục tiêu**
- 5 notebook đúng yêu cầu đề, mỗi notebook **chạy độc lập từ trên xuống trên Kaggle** (bật GPU + Internet) mà không phụ thuộc notebook khác.
- Code rõ ràng, có chú thích tiếng Việt, có các tiêu đề Markdown giải thích từng bước.
- Đánh giá bằng `MulticlassClassificationEvaluator` của Spark cho **từng khía cạnh** và trung bình.
- Notebook 5 có bảng so sánh kết quả các thí nghiệm và kết luận mô hình tốt nhất.
- Hoàn thành trong 7 ngày, chia việc đều giữa A và B.

**Không làm**
- Không triển khai (deploy), không làm API/web.
- Không chạy Spark cluster thật — dùng Spark local mode trên Kaggle.
- Không đuổi theo kết quả SOTA; ưu tiên đúng yêu cầu và trình bày tốt.

**Thất bại dù code "chạy được"**
- Notebook nộp mà thầy chạy bị lỗi vì thiếu file/thư viện/mô hình từ notebook khác.
- Thiếu bước nào trong chuỗi bắt buộc: train → lưu mô hình → đóng gói UDF → áp lên Spark DataFrame → Spark evaluator.
- Chỉ báo một con số accuracy gộp, không tách theo khía cạnh.
- Sai quy tắc đặt tên file nộp.

## 3. Ràng buộc và dữ kiện đã biết

**Dữ liệu** (`cthng123/absa-vietnamese` trên Kaggle, CC BY 4.0, ~1,66 MB):
- ~11.000 bình luận về sản phẩm giày trên Shopee.
- 8 khía cạnh: `Price`, `Shipping`, `Outlook`, `Quality`, `Size`, `Shop_Service`, `General`, `Others`.
- 4 nhãn: `None = -1`, `Negative = 0`, `Positive = 1`, `Neutral = 2`. `None` được hiểu là khía cạnh không được nhắc đến (suy luận từ mô tả, cần xác nhận khi xem dữ liệu ngày 1).
- **Chưa biết**: tên file, tên cột, có sẵn tập train/val/test hay không. Việc đầu tiên của ngày 1 là xem dữ liệu và điền vào mục 6.1.

**Môi trường Kaggle**
- Image Kaggle dựng trên image Colab, có sẵn **PySpark 4.0.5** và **OpenJDK 21** (suy ra từ cấu hình image công khai; xác nhận bằng ô kiểm tra ở mục 6.2 ngày 1). `predict_batch_udf` có từ Spark 3.4 → dùng được.
- GPU miễn phí (T4 ×2 hoặc P100), ~30 giờ/tuần/tài khoản. Hai tài khoản → ~60 giờ, dư cho bài này.
- Cần bật **Internet** trong notebook (để tải PhoBERT từ Hugging Face và `pip install` thư viện tách từ). Bật Internet trên Kaggle cần tài khoản đã xác minh số điện thoại — cả A và B làm việc này ngày 1.
- Mỗi tiến trình Python worker của Spark sẽ nạp mô hình riêng; với BERT trên GPU phải giới hạn số worker (vd `local[2]`) để tránh hết VRAM.

**Cách chấm bài** (đã xác nhận với nhóm)
- Thầy **chạy lại notebook**, có thể trên Kaggle hoặc **bất kỳ công cụ chạy notebook nào có Spark** (Colab, Jupyter cài PySpark trên máy, Databricks...). Nhóm **phát triển trên Kaggle**, nhưng notebook nộp phải **không phụ thuộc Kaggle**. Cụ thể:
  - **Dữ liệu — tìm theo thứ tự**: (1) thư mục `./data/` cạnh notebook; (2) `/kaggle/input/**` (tìm bằng `glob`); (3) tải từ bản sao dữ liệu trong repo GitHub của nhóm (URL `raw.githubusercontent.com`). Dataset chỉ 1,66 MB và có giấy phép CC BY 4.0 (cho phép chia sẻ lại, ghi nguồn) nên đặt một bản sao trong `data/` của repo, ghi nguồn trong README. **Repo phải để Public** từ lúc nộp. *(Quyết định của người thiết kế.)*
  - **Thư viện**: ô đầu cài các gói còn thiếu bằng `%pip install -q` (có ghi phiên bản: `transformers`, `pyvi`/`underthesea`...). `torch` và `pyspark` chỉ kiểm tra, không cài đè, để không phá môi trường Spark sẵn có của thầy (ví dụ Databricks).
  - **Spark**: dùng `SparkSession.builder.getOrCreate()` để dùng lại session có sẵn (Databricks), nếu chưa có thì tạo local. Kiểm tra phiên bản Spark ≥ 3.4 (cần cho `predict_batch_udf`) và báo lỗi rõ ràng nếu thấp hơn.
  - **Đường dẫn ghi file**: lưu mô hình vào `./models/` (thư mục tương đối), không viết cứng `/kaggle/working`.
  - **Mô hình trong UDF phải chạy được cả trên cluster nhiều máy**: thư mục mô hình chỉ có trên máy driver, nên phân phối tới executor bằng `spark.sparkContext.addFile(model_dir, recursive=True)` và trong UDF lấy đường dẫn bằng `SparkFiles.get(...)`. Cách này chạy cả ở local mode lẫn cluster. *(Quyết định của người thiết kế.)*
  - **Không có GPU**: tự chọn thiết bị (`cuda` nếu có, ngược lại `cpu`). Khi chạy trên CPU, notebook BERT **tự chuyển sang cấu hình rút gọn** (lấy một phần tập train, 1 epoch, `max_len` ngắn) và in cảnh báo, để chạy xong trong khoảng 30–60 phút thay vì nhiều giờ. Kết quả đầy đủ từ lần chạy GPU trên Kaggle vẫn nằm trong output đã lưu của file nộp. Có biến `FAST_MODE` ở đầu notebook để bật/tắt thủ công. *(Quyết định của người thiết kế — xem mục 9.)*
  - Ô đầu tiên của mỗi notebook là ô Markdown **"Hướng dẫn chạy"**: yêu cầu môi trường (Python, PySpark ≥ 3.4, Java 17+, nên có GPU), cách cung cấp dữ liệu (tự động như trên), thời gian chạy ước tính có GPU và không GPU, và link notebook Public trên Kaggle (cách chạy dễ nhất: "Copy & Edit").
  - Ô kiểm tra môi trường dừng sớm với thông báo dễ hiểu nếu thiếu Java, Spark quá cũ hoặc không tải được dữ liệu, để thầy không phải chờ chạy hết mới thấy lỗi.
  - Thời gian chạy mục tiêu (có GPU): NB1/NB2 dưới 15 phút, NB3/NB4 dưới 30 phút, NB5 dưới 2 giờ.
  - **Ngày 6** có thêm bước chạy thử ít nhất một notebook RNN và một notebook BERT trên **Google Colab** (môi trường khác Kaggle) để chắc chắn notebook không phụ thuộc Kaggle.

**Đề bài**
- Mỗi notebook ghi rõ "Train mô hình + đóng gói ...", nên **mỗi notebook đều có bước huấn luyện** (không nạp lại mô hình từ notebook khác).
- Đặt tên file: `NhomX_danhsachTen_danhsachMSSV_ABSASpark`. Mẫu này cho cả 5 file cùng một tên, nên thêm hậu tố `_NB1` … `_NB5` và nộp 5 file `.ipynb` riêng, ví dụ `Nhom3_NguyenVanA-TranThiB_2100001-2100002_ABSASpark_NB1.ipynb`. Hỏi lại thầy nếu có quy định khác. *(Quyết định của người thiết kế; số nhóm, tên, MSSV điền ở ngày 7.)*

## 4. Các phương án đã cân nhắc

| Phương án | Ưu | Nhược | Quyết định |
|---|---|---|---|
| **A. Notebook tự chứa toàn bộ code**; code dùng chung viết 1 lần ở `common/common_cells.py` rồi dán vào đầu mỗi notebook | File nộp chạy được ngay; không có bước "ghép" trước khi nộp; code hiển thị và chú thích ngay trong notebook như đề yêu cầu | Sửa code chung phải dán lại vào 5 notebook | **Chọn** (nhóm đã chọn) |
| B. Code chung thành package `src/absa/`, notebook import (qua Kaggle Utility Script/Dataset) | Gọn, sửa một chỗ | File nộp chỉ là `.ipynb` → thầy chạy thiếu thư viện; phải chép ngược vào notebook trước khi nộp, dễ lỗi phút cuối | Loại |
| C. Train 1 lần ở NB1/NB3, NB2/NB4 chỉ nạp mô hình đã lưu | Tiết kiệm GPU | Đề ghi "Train" ở mọi notebook → rủi ro mất điểm; notebook không còn độc lập | Loại |

## 5. Tổ chức repo GitHub và cách làm việc

### 5.1 Cấu trúc repo

```
absa-spark/
├── README.md                 # thành viên, phân công, cách chạy, bảng kết quả cuối
├── .gitignore                # bỏ qua *.pt, *.bin, models/, checkpoints
├── data/                     # bản sao dataset (CC BY 4.0, ghi nguồn) — notebook tải từ đây khi chạy ngoài Kaggle
├── common/
│   └── common_cells.py       # khối code dùng chung — nguồn gốc duy nhất, dán vào notebook
├── notebooks/
│   ├── 01_rnn_pandas_udf.ipynb
│   ├── 02_rnn_predict_batch_udf.ipynb
│   ├── 03_bert_pandas_udf.ipynb
│   ├── 04_bert_predict_batch_udf.ipynb
│   └── 05_bert_experiments.ipynb
├── submission/               # 5 file đã đổi tên theo quy tắc nộp (ngày 7)
└── docs/
    └── superpowers/specs/    # tài liệu thiết kế này
```

### 5.2 Quy tắc làm việc
- **Mỗi notebook có đúng một người sở hữu** (mục 7) — chỉ người đó sửa → gần như không có xung đột merge file `.ipynb`.
- `common/common_cells.py` do **A sở hữu**. Khi sửa: A commit, báo B, cả hai dán lại khối mới vào notebook của mình. Đầu khối có dòng `# COMMON v<số>` để kiểm tra notebook nào đang dùng bản cũ.
- **Giữ output** trong notebook khi commit (thầy cần thấy kết quả; vì mỗi notebook một chủ nên output không gây xung đột).
- Đồng bộ Kaggle → GitHub: dùng *File → Link to GitHub* / "Save version to GitHub" của Kaggle, hoặc tải `.ipynb` về và commit thủ công. Commit thẳng lên `main` là đủ cho nhóm 2 người; dùng nhánh/PR khi muốn người kia xem lại (khuyến khích ở ngày 6).
- Theo dõi việc bằng GitHub Issues (mỗi notebook một issue) hoặc GitHub Projects dạng bảng.
- Không commit mô hình lên GitHub (dung lượng). Dữ liệu thì có commit (bản sao nhỏ trong `data/`). Repo để **Public** trước khi nộp để notebook tải được dữ liệu.

## 6. Thiết kế kỹ thuật

### 6.1 Bài toán
- Đầu vào: 1 bình luận. Đầu ra: 8 nhãn, mỗi khía cạnh 1 trong 4 lớp.
- **Mã hóa nhãn nội bộ** (dùng thống nhất ở mọi notebook, kể cả cho Spark evaluator — evaluator cần nhãn không âm dạng double): `0 = None`, `1 = Negative`, `2 = Positive`, `3 = Neutral` (tức `nhãn_gốc + 1`, với Neutral 2→3).
- Mô hình chung dạng: encoder → vector đặc trưng → **8 đầu phân loại** (mỗi đầu Linear → 4 lớp). Loss mặc định: tổng (hoặc trung bình) CrossEntropy của 8 đầu.
- **Chia dữ liệu**: nếu dataset có sẵn split thì dùng; nếu không, chia 80/10/10 (train/val/test) với `seed = 42` ngay trong code chung → mọi notebook có cùng tập test, kết quả so sánh được.

### 6.2 Code dùng chung (`common_cells.py`, A viết ngày 1)
1. Ô kiểm tra môi trường: in phiên bản PySpark, `java -version`, GPU (`torch.cuda`), tạo SparkSession và chạy thử.
2. Hằng số: danh sách 8 khía cạnh, ánh xạ nhãn, `SEED`, `FAST_MODE`, `DEVICE`; hàm `find_or_download_data()` theo thứ tự `./data/` → `/kaggle/input/**` → tải từ GitHub raw.
3. `set_seed()`.
4. Đọc dữ liệu + chia train/val/test.
5. Tiền xử lý tiếng Việt cơ bản: chuẩn hóa Unicode (NFC), chữ thường, bỏ URL, rút gọn ký tự lặp ("đẹppppp" → "đẹp"), chuẩn hóa khoảng trắng. (Tách từ cho PhoBERT nằm ở notebook BERT, không ở đây.)
6. `create_spark()` — `getOrCreate()` (dùng lại session có sẵn hoặc tạo local), bật Arrow, kiểm tra Spark ≥ 3.4.
6b. `ship_model_to_spark(model_dir)` — `addFile` thư mục mô hình; trong UDF dùng `SparkFiles.get`.
7. `evaluate_with_spark(pred_df)` — với mỗi khía cạnh dùng `MulticlassClassificationEvaluator` tính `accuracy`, `f1` (weighted) và macro-F1 (tính từ `fMeasureByLabel` hoặc ghi chú rõ cách tính); trả về bảng pandas 8 dòng + dòng trung bình.

Khung huấn luyện dùng **PyTorch** cho cả hai họ mô hình (thống nhất vòng lặp train/eval, dễ đóng gói UDF giống nhau). *(Quyết định của người thiết kế.)*

### 6.3 Mô hình 1 — RNN/LSTM/CNN (NB1, NB2; A làm)
- Tokenize mức âm tiết/từ đơn giản, xây vocab từ tập train, padding/truncate theo độ dài phân vị 95%.
- Kiến trúc đề xuất: Embedding → BiLSTM (có thể thêm nhánh CNN 1D) → pooling → 8 đầu. Chọn kiến trúc cụ thể và siêu tham số là việc của A.
- Lưu: `state_dict` + vocab + config (JSON) vào `./models/rnn/`.
- NB1 và NB2 dùng cùng mô hình và cùng code huấn luyện; chỉ khác phần đóng gói UDF.

### 6.4 Mô hình 2 — Transformer/BERT (NB3, NB4; B làm)
- `vinai/phobert-base` (hoặc `phobert-base-v2`). PhoBERT cần văn bản **đã tách từ** → dùng `pyvi` (cài qua pip) hoặc `underthesea`; B chọn.
- Vector đặc trưng mặc định: vector `[CLS]` → dropout → 8 đầu.
- Fine-tune vài epoch (lr ~2e-5, max_len 128–256, batch 16–32); B chỉnh theo thực tế.
- Lưu: `state_dict` (hoặc `save_pretrained`) + tokenizer vào `./models/phobert/`.

### 6.5 Đóng gói UDF và áp lên Spark DataFrame
Chuỗi bước giống nhau ở cả 4 notebook NB1–NB4:
1. Tạo Spark DataFrame từ tập test (`spark.createDataFrame(pandas_df)`), gồm `id`, `text`, 8 cột nhãn.
2. Định nghĩa UDF dự đoán, trả về 8 nhãn dự đoán.
3. `df.withColumn("pred", udf("text"))` → tách thành 8 cột `pred_<aspect>` kiểu double.
4. Gọi `evaluate_with_spark` → in bảng kết quả.

Trước khi định nghĩa UDF: gọi `ship_model_to_spark("./models/...")`; bên trong UDF nạp mô hình từ `SparkFiles.get(...)`, không dùng đường dẫn cục bộ của driver.

**`pandas_udf` (NB1, NB3)** — dạng `Iterator[pd.Series] -> Iterator[pd.DataFrame]` (trả về struct 8 trường): nạp mô hình **một lần** ở đầu iterator rồi dự đoán từng batch. Có chú thích giải thích vì sao dùng dạng iterator (tránh nạp lại mô hình mỗi batch).

**`predict_batch_udf` (NB2, NB4)** — `pyspark.ml.functions.predict_batch_udf(make_predict_fn, return_type=..., batch_size=...)`: `make_predict_fn` nạp mô hình và trả về hàm `predict(inputs: np.ndarray) -> ...`. Với đầu vào là text, cần xử lý mảng chuỗi (tokenize bên trong hàm predict). Trả về 8 cột (struct) hoặc mảng 8 phần tử.

Thêm (không bắt buộc, nếu kịp): đo thời gian dự đoán của mỗi cách để ghi vào phần kết luận.

### 6.6 Notebook 5 — Thử nghiệm (B dựng khung, A + B chạy)
Nền: mô hình PhoBERT của NB3/4. Để tiết kiệm GPU trong 1 tuần, **thử theo 2 giai đoạn** thay vì toàn bộ tổ hợp:

**Giai đoạn 1 — Cách lấy vector đặc trưng** (giữ loss = CrossEntropy), B chạy:
- `[CLS]` (baseline)
- Mean pooling (trung bình hidden states, bỏ padding)
- **Attention pooling**: học trọng số attention trên toàn bộ token hidden states (mỗi khía cạnh một vector query riêng, hoặc một query chung — B chọn, ghi rõ)
- (tùy chọn) Max pooling / ghép CLS + mean

**Giai đoạn 2 — Hàm loss** (dùng cách pooling tốt nhất ở giai đoạn 1), A chạy:
- CrossEntropy (baseline)
- **Gated loss**: tách mỗi khía cạnh thành 2 phần — đầu "có nhắc đến không" (None vs có) và đầu cảm xúc (Neg/Pos/Neu); loss cảm xúc của khía cạnh chỉ tính khi khía cạnh đó được nhắc đến (mask). Khi dự đoán: nếu đầu "nhắc đến" ra None thì nhãn = None, ngược lại lấy nhãn của đầu cảm xúc.
- **Weighted loss**: gán trọng số thấp hơn cho các nhãn có độ tin cậy thấp. Ngày 1 kiểm tra xem dataset có cột độ tin cậy không. **Nếu có** → dùng làm trọng số từng mẫu. **Nếu không** (khả năng cao) → trọng số theo từng lớp, đặt thấp hơn cho các lớp dễ nhiễu/mơ hồ (Neutral, khía cạnh Others); ghi rõ cách chọn trọng số và lý do trong notebook. *(Quyết định của người thiết kế — xem mục 9.)*
- **Focal loss** (γ = 2): tăng trọng số mẫu khó, giảm ảnh hưởng mẫu dễ.

**Kết quả**: chọn mô hình theo **macro-F1 trung bình 8 khía cạnh trên tập validation**, báo cáo cuối trên tập test. Bảng tổng hợp mọi thí nghiệm (pooling, loss, accuracy, macro-F1). Mô hình tốt nhất được lưu và chạy qua Spark (`pandas_udf` + `MulticlassClassificationEvaluator`) để giữ đúng tinh thần bài.

Để chia việc song song: code mô hình nhận tham số `pooling=` và `loss=`; trong giai đoạn phát triển, hai người chạy thí nghiệm của mình trên tài khoản Kaggle riêng (A chạy phần loss). **Bản nộp cuối của NB5 phải chạy lại toàn bộ thí nghiệm trong một lần "Run All"** (không dán kết quả từ nơi khác), nên mỗi thí nghiệm dùng cấu hình rút gọn: 2–3 epoch, `max_len = 128`. Ước tính ~7 lần fine-tune × ~8–10 phút ≈ 1–1,5 giờ GPU, nằm trong giới hạn 12 giờ/phiên của Kaggle. *(Quyết định của người thiết kế.)*

**Phạm vi NB5**: chỉ PhoBERT. Đề lấy ví dụ "thay vì lấy CLS vector" (khái niệm của BERT) nên hiểu là thử trên họ Transformer. Gated loss định nghĩa đúng theo đề: aspect không được nhắc đến thì không tính loss sentiment của aspect đó.

### 6.7 Khung trình bày chung cho mỗi notebook
1. Tiêu đề, thành viên, mục tiêu notebook
2. Cài đặt & import
3. Code dùng chung (`# COMMON vX`)
4. Đọc & khám phá dữ liệu (EDA ngắn: phân bố nhãn theo khía cạnh, độ dài câu)
5. Tiền xử lý
6. Xây dựng mô hình
7. Huấn luyện (có biểu đồ loss/metric theo epoch)
8. Lưu mô hình
9. Đóng gói UDF
10. Áp dụng UDF lên Spark DataFrame
11. Đánh giá bằng Spark evaluator
12. Kết luận

## 7. Phân công và lịch 1 tuần

| Notebook | Chủ sở hữu |
|---|---|
| `common_cells.py`, NB1, NB2 | A |
| NB3, NB4, NB5 (file) | B |
| Thí nghiệm NB5 | B: pooling · A: loss |

| Ngày | A | B |
|---|---|---|
| 1 | Tạo repo, README, `.gitignore`; xem dữ liệu; viết `common_cells.py` | Xác minh tài khoản Kaggle, chạy ô kiểm tra môi trường; thử tải PhoBERT + tách từ |
| 2 | NB1: train RNN + `pandas_udf` + đánh giá | NB3: fine-tune PhoBERT + `pandas_udf` + đánh giá |
| 3 | NB2: `predict_batch_udf` (tái dùng code NB1) | NB4: `predict_batch_udf` (tái dùng code NB3); dựng khung NB5 |
| 4 | NB5 giai đoạn 2 chuẩn bị code loss (Gated/Weighted/Focal) | NB5 giai đoạn 1: chạy các cách pooling |
| 5 | NB5 giai đoạn 2: chạy các loss | Gộp kết quả, bảng so sánh, kết luận NB5 |
| 6 | Rà chéo NB3–5 của B; chạy thử NB1 trên Colab | Rà chéo NB1–2 của A; chạy thử NB3 trên Colab; cả hai "Run All" lại toàn bộ trên Kaggle |
| 7 | Đổi tên file vào `submission/`, cập nhật README, nộp | Dự phòng / sửa lỗi |

Mốc kiểm tra: cuối ngày 3 phải có 4 notebook NB1–NB4 chạy hết từ trên xuống. Nếu trễ, cắt bớt thí nghiệm tùy chọn ở NB5.

## 8. Để lại cho người làm quyết định
- Kiến trúc cụ thể của mô hình 1 (BiLSTM, CNN, hay kết hợp) và mọi siêu tham số.
- `phobert-base` hay `phobert-base-v2`; `pyvi` hay `underthesea`.
- Dạng trả về của UDF (struct hay array), `batch_size` của `predict_batch_udf`.
- Attention pooling dùng query chung hay riêng theo khía cạnh.
- Cách bố trí biểu đồ, cách viết chú thích (miễn rõ ràng và bằng tiếng Việt).

## 9. Các quyết định của người thiết kế (cần nhóm kiểm tra lại)
- Dùng PyTorch cho cả hai họ mô hình.
- Mã hóa nhãn nội bộ `0=None, 1=Neg, 2=Pos, 3=Neu`.
- Chia 80/10/10, `seed = 42` nếu dataset không có sẵn split.
- Chỉ số chọn mô hình: macro-F1 trung bình 8 khía cạnh.
- NB5 thử 2 giai đoạn (pooling trước, loss sau) thay vì toàn bộ tổ hợp; chỉ trên PhoBERT.
- Bản nộp NB5 chạy lại toàn bộ thí nghiệm trong một lần, cấu hình rút gọn (2–3 epoch, `max_len = 128`).
- Cách hiểu "Weighted loss" là trọng số theo lớp, thấp hơn cho lớp mơ hồ.
- Giữ output notebook khi commit; commit thẳng `main`.
- Để notebook Kaggle ở chế độ Public và ghi link cho thầy.
- Notebook không phụ thuộc Kaggle: dữ liệu tìm `./data/` → `/kaggle/input` → GitHub raw (bản sao dataset trong repo Public); mô hình lưu `./models/`; phân phối mô hình tới executor bằng `addFile`/`SparkFiles`.
- Không có GPU → tự chuyển sang cấu hình rút gọn (`FAST_MODE`), kết quả đầy đủ nằm trong output đã lưu.
- Ngày 6 chạy thử trên Colab để kiểm tra tính độc lập với Kaggle.
- Tên file nộp thêm hậu tố `_NB1` … `_NB5`, nộp 5 file `.ipynb` riêng.
