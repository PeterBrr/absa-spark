# ABSA tiếng Việt trên Spark

Bài tập nhóm: phân loại bình luận theo khía cạnh (Aspect-Based Sentiment Analysis) trên dữ liệu
bình luận giày Shopee, đóng gói mô hình thành UDF của Spark (`pandas_udf`, `predict_batch_udf`)
và đánh giá bằng `MulticlassClassificationEvaluator`.

## Thành viên và phân công

| Thành viên | Phần việc |
|---|---|
| A | Code chung (`common/common_cells.py`, UDF, loss), NB1, NB2, thí nghiệm loss ở NB5 |
| B | `common/bert_cells.py`, NB3, NB4, NB5 (khung + thí nghiệm pooling) |

## Notebook

| # | File | Nội dung |
|---|---|---|
| 1 | `notebooks/01_rnn_pandas_udf.ipynb` | BiLSTM + `pandas_udf` |
| 2 | `notebooks/02_rnn_predict_batch_udf.ipynb` | BiLSTM + `predict_batch_udf` |
| 3 | `notebooks/03_bert_pandas_udf.ipynb` | PhoBERT + `pandas_udf` |
| 4 | `notebooks/04_bert_predict_batch_udf.ipynb` | PhoBERT + `predict_batch_udf` |
| 5 | `notebooks/05_bert_experiments.ipynb` | Thử nghiệm pooling và hàm loss |

## Cách chạy notebook

Mỗi notebook chạy độc lập từ trên xuống ("Run All"):

- **Kaggle (dễ nhất):** mở link Public của notebook → *Copy & Edit* → bật GPU và Internet → Run All.
- **Môi trường khác (Colab, Jupyter, Databricks):** cần Python 3.10+, PySpark ≥ 3.4, Java 17+;
  nên có GPU. Dữ liệu được tự tìm trong `./data/`, `/kaggle/input/`, hoặc tự tải từ thư mục
  `data/` của repo này.

### Môi trường Kaggle đã kiểm tra

Kết quả ô `check_environment()` trên Kaggle (tháng 11/2026), GPU T4 + Internet:

| Thành phần | Kết quả |
|---|---|
| PySpark | 4.0.4, có sẵn (thiết kế dự đoán 4.0.5) |
| Java | OpenJDK 21.0.12 |
| GPU | Có CUDA. Không bật GPU thì notebook tự chuyển sang `FAST_MODE` |
| `pyvi` | Cài được bằng `pip install pyvi`, cần bật Internet |
| `vinai/phobert-base-v2` | Tải được từ Hugging Face, cần bật Internet. Mô hình đã lưu được nạp lại mà không cần Internet |

Tài khoản Kaggle phải xác minh số điện thoại thì mới bật được Internet.

## Dữ liệu

Xem [data/SOURCE.md](data/SOURCE.md). Dataset có sẵn 3 tập train / val / test (8.424 / 936 / 2.340 dòng),
cột văn bản `Review`, không có cột độ tin cậy.

## Chạy test cho code chung

```bash
pip install -r requirements-dev.txt
pytest -q
```

Cần Java 17+. Trên máy không cài Java, có thể đặt một JDK portable vào `.tools/jdk-*`; `tests/conftest.py`
sẽ tự dùng nó.

Kiểm tra notebook dùng đúng phiên bản code chung:

```bash
python tools/check_notebooks.py notebooks/*.ipynb
```

## Kết quả

Đánh giá trên tập test (2.340 bình luận) bằng `MulticlassClassificationEvaluator`, trung bình 8 khía cạnh.
Chạy trên Kaggle (GPU T4, PySpark 4.0.4).

| Notebook | Mô hình | Đóng gói | Accuracy | F1 (weighted) | Macro-F1 | Kaggle |
|---|---|---|---|---|---|---|
| NB1 | BiLSTM | `pandas_udf` | 0.9038 | 0.8975 | 0.6435 | [link](https://www.kaggle.com/code/quangnhat6/absa-nb1-rnn-pandas-udf) |
| NB2 | BiLSTM | `predict_batch_udf` | 0.9038 | 0.8975 | 0.6435 | [link](https://www.kaggle.com/code/quangnhat6/absa-nb2-rnn-predict-batch-udf) |
| NB3 | PhoBERT | `pandas_udf` | 0.9194 | 0.9122 | 0.6558 | [link](https://www.kaggle.com/code/nha4von/absa-nb3-phobert-pandas-udf) |
| NB4 | PhoBERT | `predict_batch_udf` | 0.9194 | 0.9122 | 0.6558 | [link](https://www.kaggle.com/code/nha4von/absa-nb4-phobert-predict-batch-udf) |
| NB5 | PhoBERT tốt nhất (max + focal) | `pandas_udf` | 0.9196 | 0.9183 | 0.7094 | [link](https://www.kaggle.com/code/nha4von/absa-nb5-phobert-experiments) |

Mốc so sánh: luôn đoán None cho mọi khía cạnh được macro-F1 = 0.2426.
NB1 và NB2 dùng cùng mô hình nên cho cùng kết quả; khía cạnh tốt nhất là Others (0.876), kém nhất là General (0.488).
NB3 và NB4 cũng vậy (PhoBERT, pooling CLS, 5 epoch, ~13 phút trên T4); tốt nhất Others (0.910), kém nhất General (0.541).
Với 3 epoch như kế hoạch ban đầu, PhoBERT chỉ đạt macro-F1 0.6262 (val vẫn còn tăng) nên tăng lên 5 epoch.
NB5 (thí nghiệm 3 epoch, chọn theo val macro-F1): pooling tốt nhất `max` (0.640, so với `cls` 0.629, `mean` 0.597, `attention` 0.597);
với `max`, loss tốt nhất `focal` (0.681, so với `gated` 0.663, `ce` 0.640, `weighted` 0.596). Mô hình `max + focal` huấn luyện lại 5 epoch
đạt macro-F1 test 0.7094 qua Spark; tốt nhất Others (0.879), kém nhất General (0.556). Chạy toàn bộ NB5 ~1 giờ trên T4.
Link Kaggle hiện đang ở chế độ Private, sẽ chuyển sang Public khi nộp.

## Tài liệu

- Thiết kế: [docs/superpowers/specs/2026-11-13-absa-spark-design.md](docs/superpowers/specs/2026-11-13-absa-spark-design.md)
- Kế hoạch: [docs/superpowers/plans/2026-11-13-absa-spark.md](docs/superpowers/plans/2026-11-13-absa-spark.md)
