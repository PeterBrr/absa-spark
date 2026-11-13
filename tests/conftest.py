"""Cấu hình chung cho pytest.

- Thêm thư mục gốc repo vào sys.path để `from common.xxx import ...` chạy được.
- Nếu máy chưa đặt JAVA_HOME mà repo có JDK portable trong `.tools/`, dùng JDK đó.
- Fixture `spark`: một SparkSession local[2] dùng chung cho cả phiên test.
"""
import glob
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if not os.environ.get("JAVA_HOME"):
    jdks = sorted(glob.glob(os.path.join(ROOT, ".tools", "jdk-*")))
    if jdks:
        os.environ["JAVA_HOME"] = jdks[-1]
        os.environ["PATH"] = os.path.join(jdks[-1], "bin") + os.pathsep + os.environ["PATH"]

# Worker Python của Spark phải dùng cùng trình thông dịch với pytest (quan trọng trên Windows/venv)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("absa-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    yield session
    session.stop()
