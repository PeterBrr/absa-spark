# === COMMON udf_batch.py v1 ===
# Đóng gói mô hình bằng predict_batch_udf — dán NGUYÊN VĂN file common/udf_batch.py vào ô này.
from typing import Callable

import numpy as np
from pyspark.ml.functions import predict_batch_udf

try:  # khi chạy pytest
    from common.common_cells import ASPECTS, PRED_SCHEMA, shipped_model_path
except ImportError:  # trong notebook: đã có sẵn từ ô code chung phía trên
    pass


def make_batch_udf(load_predictor: Callable[[str], Callable], shipped_name: str, batch_size: int = 64):
    """Tạo UDF dự đoán 8 nhãn bằng pyspark.ml.functions.predict_batch_udf (Spark >= 3.4).

    - make_predict_fn() được Spark gọi một lần trên mỗi worker Python để nạp mô hình
      (shipped_model_path → SparkFiles.get, chạy được cả trên cluster) và trả về hàm predict.
    - Spark tự gom các dòng thành batch đúng `batch_size` phần tử rồi gọi predict(batch):
      batch nhỏ tốn ít bộ nhớ (GPU) hơn, batch lớn chạy nhanh hơn.
    - predict trả về dict {tên khía cạnh: mảng nhãn} ứng với kiểu trả về struct PRED_SCHEMA.
    """

    def make_predict_fn():
        predictor = load_predictor(shipped_model_path(shipped_name))

        def predict(texts: np.ndarray) -> dict:
            # Ô null có thể đến dưới dạng None hoặc NaN (float) → coi là chuỗi rỗng
            batch = [t if isinstance(t, str) else "" for t in texts]
            preds = np.asarray(predictor(batch), dtype=np.int32)
            return {a: preds[:, i] for i, a in enumerate(ASPECTS)}

        return predict

    return predict_batch_udf(make_predict_fn, return_type=PRED_SCHEMA, batch_size=batch_size)
