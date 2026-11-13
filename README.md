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
| NB3 | PhoBERT | `pandas_udf` | | | | |
| NB4 | PhoBERT | `predict_batch_udf` | | | | |
| NB5 | PhoBERT tốt nhất | `pandas_udf` | | | | |

Mốc so sánh: luôn đoán None cho mọi khía cạnh được macro-F1 = 0.2426.
NB1 và NB2 dùng cùng mô hình nên cho cùng kết quả; khía cạnh tốt nhất là Others (0.876), kém nhất là General (0.488).
Link Kaggle hiện đang ở chế độ Private, sẽ chuyển sang Public khi nộp.

## Tài liệu

- Thiết kế: [docs/superpowers/specs/2026-11-13-absa-spark-design.md](docs/superpowers/specs/2026-11-13-absa-spark-design.md)
- Kế hoạch: [docs/superpowers/plans/2026-11-13-absa-spark.md](docs/superpowers/plans/2026-11-13-absa-spark.md)
