# === COMMON udf_pandas.py v1 ===
# Đóng gói mô hình bằng pandas_udf — dán NGUYÊN VĂN file common/udf_pandas.py vào ô này.
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from pyspark.sql.functions import pandas_udf

try:  # khi chạy pytest
    from common.common_cells import ASPECTS, PRED_SCHEMA, shipped_model_path
except ImportError:  # trong notebook: đã có sẵn từ ô code chung phía trên
    pass


def make_pandas_udf(load_predictor: Callable[[str], Callable], shipped_name: str):
    """Tạo pandas_udf dự đoán 8 nhãn cho cột văn bản.

    - load_predictor(model_dir) -> predictor(list[str]) -> np.ndarray (N, 8), nhãn 0..3.
    - shipped_name: tên trả về từ ship_model_to_spark(); mô hình được gửi tới từng worker,
      vì thư mục mô hình gốc chỉ có trên máy driver — cách này chạy được cả trên cluster nhiều máy
      (shipped_model_path lấy đường dẫn qua SparkFiles.get).

    Dùng dạng Iterator[pd.Series] -> Iterator[pd.DataFrame]: mô hình được nạp MỘT LẦN cho mỗi
    partition (trước vòng lặp), rồi dùng lại cho mọi batch — tránh nạp lại mô hình ở từng batch.
    Kết quả là một cột struct (PRED_SCHEMA) gồm 8 trường; tách bằng explode_predictions().
    """

    @pandas_udf(PRED_SCHEMA)
    def predict_udf(batches: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        predictor = load_predictor(shipped_model_path(shipped_name))  # nạp 1 lần / partition
        for texts in batches:
            # Văn bản rỗng/None vẫn được dự đoán (thường ra None cho cả 8 khía cạnh)
            preds = np.asarray(predictor(texts.fillna("").astype(str).tolist()), dtype=np.int32)
            yield pd.DataFrame(preds, columns=ASPECTS)

    return predict_udf
