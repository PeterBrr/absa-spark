import os

import pandas as pd
import pytest

import common.common_cells as cc
from common.common_cells import (
    ASPECTS, DATA_FILES, check_environment, clean_text, find_or_download_data,
    get_splits, load_dataset, resolve_train_config,
)

FIXTURE = "tests/fixtures/mini.csv"


def test_clean_text():
    assert clean_text(None) == ""
    assert clean_text("  Giày ĐẸPPPPP lắm   https://shopee.vn/x ") == "giày đẹp lắm"
    assert isinstance(clean_text("😍😍😍"), str)  # chỉ cần không lỗi
    assert clean_text(float("nan")) == ""  # ô trống trong CSV đọc ra NaN


def test_load_dataset_maps_labels():
    df = load_dataset(FIXTURE)
    assert list(df.columns[:9]) == ["text", *ASPECTS]
    assert set(df[ASPECTS].stack().unique()) <= {0, 1, 2, 3}
    assert len(df) == 20
    raw = pd.read_csv(FIXTURE)
    # -1 → 0 (None), 0 → 1 (Negative), 1 → 2 (Positive), 2 → 3 (Neutral)
    assert (df["Price"].values == raw["Price"].map({-1: 0, 0: 1, 1: 2, 2: 3}).values).all()


def test_load_dataset_from_split_dir(tmp_path):
    raw = pd.read_csv(FIXTURE)
    for name, part in zip(DATA_FILES, (raw[:12], raw[12:15], raw[15:])):
        part.to_csv(tmp_path / name, index=False)
    df = load_dataset(str(tmp_path))
    assert df["split"].value_counts().to_dict() == {"train": 12, "test": 5, "val": 3}
    tr, va, te = get_splits(df)
    assert (len(tr), len(va), len(te)) == (12, 3, 5)
    assert "split" not in tr.columns


def test_splits_are_deterministic_and_disjoint():
    df = pd.concat([load_dataset(FIXTURE)] * 10, ignore_index=True)
    a, b = get_splits(df), get_splits(df)
    assert all(x.index.equals(y.index) for x, y in zip(a, b))
    tr, va, te = a
    assert len(tr) + len(va) + len(te) == len(df)
    assert not set(tr.index) & set(te.index)
    assert abs(len(tr) / len(df) - 0.8) < 0.05


def _make_split_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        (path / name).write_text("x")


def test_find_data_prefers_local(tmp_path):
    _make_split_dir(tmp_path)
    assert find_or_download_data(local_dir=str(tmp_path), kaggle_root="/khong-co") == str(tmp_path)


def test_find_data_from_notebooks_subfolder(tmp_path, monkeypatch):
    _make_split_dir(tmp_path / "data")
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path / "notebooks")  # notebook chạy từ thư mục notebooks/ của repo
    found = find_or_download_data(kaggle_root="/khong-co", url="http://127.0.0.1:9/khong-co")
    assert os.path.samefile(found, tmp_path / "data")


def test_find_data_searches_kaggle_tree(tmp_path):
    p = tmp_path / "ten-bat-ky" / "sub"
    _make_split_dir(p)
    found = find_or_download_data(local_dir=str(tmp_path / "trong"), kaggle_root=str(tmp_path))
    assert os.path.samefile(found, p)


def test_find_data_clear_error_when_unreachable(tmp_path):
    with pytest.raises(FileNotFoundError, match="cthng123/absa-vietnamese"):
        find_or_download_data(local_dir=str(tmp_path / "trong"), kaggle_root="/khong-co",
                              url="http://127.0.0.1:9/khong-co")


def test_fast_mode_shrinks_config():
    base = {"epochs": 3, "max_len": 128, "train_frac": 1.0, "batch_size": 32}
    assert resolve_train_config(base, fast=False) == base
    fast = resolve_train_config(base, fast=True)
    assert fast["epochs"] == 1 and fast["max_len"] <= 64 and fast["train_frac"] <= 0.2
    assert base["epochs"] == 3  # không sửa dict gốc


def test_check_environment_reports(spark):
    info = check_environment()
    assert set(info) == {"python", "pyspark", "java", "cuda", "fast_mode"}


def test_check_environment_rejects_old_spark(spark):
    with pytest.raises(RuntimeError):
        check_environment(min_spark="99.0")


def test_java_major_parsing():
    assert cc._java_major('openjdk version "21.0.13" 2026-10-20 LTS') == 21
    assert cc._java_major('java version "1.8.0_392"') == 8
    assert cc._java_major('openjdk version "11.0.2" 2019-01-15') == 11
    assert cc._java_major("") == 0


def test_check_environment_rejects_old_java_for_spark4(spark, monkeypatch):
    monkeypatch.setattr(cc, "_java_version", lambda: 'openjdk version "11.0.2" 2019-01-15')
    if cc._version_tuple(__import__("pyspark").__version__) >= (4,):
        with pytest.raises(RuntimeError, match="Java 17"):
            check_environment()


def test_check_environment_stops_when_spark_cannot_start(spark, monkeypatch):
    def broken(*args, **kwargs):
        raise Exception("JAVA_GATEWAY_EXITED")

    monkeypatch.setattr(cc, "create_spark", broken)
    with pytest.raises(RuntimeError, match="Spark"):
        check_environment()


def test_java_version_timeout_is_handled(monkeypatch):
    import subprocess

    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="java", timeout=30)

    monkeypatch.setattr(cc.subprocess, "run", slow)
    assert cc._java_version() == ""
