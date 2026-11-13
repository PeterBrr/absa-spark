import os

import pandas as pd
from pyspark import SparkFiles
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType

from common.common_cells import (
    ASPECTS, PRED_SCHEMA, create_spark, evaluate_with_spark, explode_predictions,
    ship_model_to_spark, shipped_model_path, to_spark_df,
)


def test_create_spark_reuses_session(spark):
    assert create_spark() is spark or create_spark().sparkContext is spark.sparkContext


def test_to_spark_df_schema(spark):
    pdf = pd.DataFrame({"text": ["a", "b"], **{a: [0, 3] for a in ASPECTS}})
    sdf = to_spark_df(spark, pdf)
    types = dict(sdf.dtypes)
    assert types["id"] == "bigint" and types["text"] == "string"
    assert all(types[a] == "double" for a in ASPECTS)
    assert sdf.count() == 2


def test_evaluate_perfect(spark):
    pdf = pd.DataFrame({"text": list("abcd"), **{a: [0, 1, 2, 0] for a in ASPECTS}})
    sdf = to_spark_df(spark, pdf)
    for a in ASPECTS:
        sdf = sdf.withColumn(f"pred_{a}", F.col(a))
    res = evaluate_with_spark(sdf)
    assert list(res.index) == ASPECTS + ["Mean"]
    assert list(res.columns) == ["accuracy", "f1_weighted", "f1_macro"]
    assert (res["accuracy"] == 1.0).all() and (res["f1_macro"] == 1.0).all()


def test_evaluate_missing_class_no_nan(spark):  # nhãn 3 (Neutral) không xuất hiện
    pdf = pd.DataFrame({"text": list("abcd"), **{a: [0, 0, 1, 1] for a in ASPECTS}})
    sdf = to_spark_df(spark, pdf)
    for a in ASPECTS:
        sdf = sdf.withColumn(f"pred_{a}", F.lit(0.0))
    res = evaluate_with_spark(sdf)
    assert not res.isna().any().any()
    assert res.loc["Mean", "accuracy"] == 0.5
    # macro-F1 trên 2 nhãn có mặt {0, 1}: F1(0)=2/3, F1(1)=0 → 1/3
    assert abs(res.loc["Price", "f1_macro"] - 1 / 3) < 1e-9


def test_explode_predictions(spark):
    sdf = spark.createDataFrame([(dict(zip(ASPECTS, range(8))),)],
                                StructType([StructField("pred", PRED_SCHEMA)]))
    out = explode_predictions(sdf)
    assert [out.first()[f"pred_{a}"] for a in ASPECTS] == [float(i) for i in range(8)]


def test_ship_model_twice_is_safe(spark, tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "w.txt").write_text("v1")
    n1 = ship_model_to_spark(spark, str(d))
    (d / "w.txt").write_text("v2")
    n2 = ship_model_to_spark(spark, str(d))
    assert n1 != n2
    got = spark.sparkContext.parallelize([0], 1).map(
        lambda _: open(os.path.join(shipped_model_path(n2), "w.txt")).read()).collect()
    assert got == ["v2"]


def test_ship_local_mode_uses_absolute_copy(spark, tmp_path):
    d = tmp_path / "m2"
    d.mkdir()
    (d / "w.txt").write_text("ok")
    name = ship_model_to_spark(spark, str(d))
    assert os.path.isabs(name) and os.path.isfile(os.path.join(name, "w.txt"))
    assert os.path.normpath(name) != os.path.normpath(str(d))  # là bản chép, không phải thư mục gốc


def test_ship_on_cluster_uses_addfile(spark, tmp_path, monkeypatch):
    d = tmp_path / "m3"
    d.mkdir()
    (d / "w.txt").write_text("ok")
    calls = []
    sc = spark.sparkContext
    monkeypatch.setattr(sc, "master", "spark://cluster:7077")
    monkeypatch.setattr(sc, "addFile", lambda path, recursive=False: calls.append((path, recursive)))
    name = ship_model_to_spark(spark, str(d))
    assert not os.path.isabs(name) and name.startswith("m3_")
    assert len(calls) == 1 and calls[0][1] is True and calls[0][0].endswith(name)


def test_shipped_model_path_uses_sparkfiles_for_names(spark):
    assert shipped_model_path("ten_khong_ton_tai") == SparkFiles.get("ten_khong_ton_tai")
