import os

import numpy as np
import pandas as pd
import pytest

from common.common_cells import ASPECTS, explode_predictions, ship_model_to_spark
from common.udf_batch import make_batch_udf
from common.udf_pandas import make_pandas_udf


def fake_loader(model_dir):
    """Predictor giả: nhãn = min(độ dài văn bản, 3) cho cả 8 khía cạnh."""
    k = int(open(os.path.join(model_dir, "k.txt")).read())
    return lambda texts: np.array([[min(len(t), 3) if k else 0] * 8 for t in texts])


@pytest.fixture
def shipped(spark, tmp_path):
    d = tmp_path / "fake"
    d.mkdir()
    (d / "k.txt").write_text("1")
    return ship_model_to_spark(spark, str(d))


TEXTS = ["", None, "ab", "abcdef"]
EXPECTED = [0, 0, 2, 3]


@pytest.mark.parametrize("factory", ["pandas", "batch"])
def test_udf_predicts_8_labels_and_handles_empty(spark, shipped, factory):
    if factory == "pandas":
        udf = make_pandas_udf(fake_loader, shipped)
    else:
        udf = make_batch_udf(fake_loader, shipped, batch_size=2)
    sdf = spark.createDataFrame(pd.DataFrame({"text": TEXTS})).repartition(2)
    out = explode_predictions(sdf.withColumn("pred", udf("text"))).toPandas()
    assert len(out) == len(TEXTS)
    for t, e in zip(TEXTS, EXPECTED):
        row = out[out["text"].isna()] if t is None else out[out["text"] == t]
        assert [row.iloc[0][f"pred_{a}"] for a in ASPECTS] == [float(e)] * 8
