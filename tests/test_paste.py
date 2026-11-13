"""Mô phỏng việc dán các khối common/*.py vào CÙNG một notebook (chung namespace __main__)."""
import os

import pyspark.sql.functions
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["common_cells.py", "rnn_cells.py", "losses.py", "udf_pandas.py", "udf_batch.py"]


def _paste_all():
    ns = {"__name__": "__main__"}
    for name in ORDER:
        with open(os.path.join(ROOT, "common", name), encoding="utf-8") as fh:
            exec(compile(fh.read(), name, "exec"), ns)
    return ns


def test_losses_survive_spark_functions_alias():
    ns = _paste_all()
    # Thói quen phổ biến trong notebook Spark: ghi đè tên F bằng pyspark.sql.functions
    ns["F"] = pyspark.sql.functions
    logits, labels = torch.randn(2, 8, 4), torch.randint(0, 4, (2, 8))
    out = {"mention": torch.randn(2, 8, 2), "sentiment": torch.randn(2, 8, 3)}
    assert ns["ce_loss"](logits, labels).ndim == 0
    assert ns["focal_loss"](logits, labels).ndim == 0
    assert ns["weighted_ce_loss"](logits, labels, torch.ones(8, 4)).ndim == 0
    assert ns["gated_loss"](out, labels).ndim == 0
