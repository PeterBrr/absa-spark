# === COMMON common_cells.py v1 ===
# Khối code dùng chung cho cả 5 notebook — dán NGUYÊN VĂN file common/common_cells.py vào ô này.
# Gồm: hằng số, kiểm tra môi trường, đọc/chia dữ liệu, vòng lặp huấn luyện, tiện ích Spark.
import copy
import glob
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.request
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

COMMON_VERSION = 1

# ---------------------------------------------------------------------------
# 1. Hằng số
# ---------------------------------------------------------------------------
ASPECTS = ["Price", "Shipping", "Outlook", "Quality", "Size", "Shop_Service", "General", "Others"]
LABEL_NAMES = ["None", "Negative", "Positive", "Neutral"]  # nhãn nội bộ 0..3
ORIG_TO_IDX = {-1: 0, 0: 1, 1: 2, 2: 3}                    # nhãn gốc của dataset → nhãn nội bộ
SEED = 42

DATASET_REF = "cthng123/absa-vietnamese"
DATA_FILES = ("train_data.csv", "val_data.csv", "test_data.csv")  # dataset có sẵn 3 tập
SPLIT_NAMES = ("train", "val", "test")
RAW_TEXT_COL = "Review"
RAW_ASPECT_COLS = {a: a for a in ASPECTS}  # tên cột gốc → tên trong ASPECTS (dataset dùng đúng tên)
# Bản sao dữ liệu trong repo GitHub (Public) — dùng khi chạy ngoài Kaggle
DATA_URL = "https://raw.githubusercontent.com/PeterBrr/absa-spark/main/data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Không có GPU → tự rút gọn cấu hình huấn luyện. Có thể gán lại thủ công: FAST_MODE = True/False
FAST_MODE = DEVICE == "cpu"


def set_seed(seed: int = SEED) -> None:
    """Cố định seed cho random, numpy, torch để kết quả lặp lại được."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 2. Kiểm tra môi trường
# ---------------------------------------------------------------------------
def _version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def _java_version() -> str:
    java = "java"
    if os.environ.get("JAVA_HOME"):
        java = os.path.join(os.environ["JAVA_HOME"], "bin", "java")
    try:
        out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, OSError):
        return ""
    lines = (out.stderr or out.stdout).strip().splitlines()
    return lines[0] if lines else ""


def check_environment(min_spark: str = "3.4", require_gpu: bool = False) -> dict:
    """Kiểm tra Python, PySpark, Java, GPU. Dừng sớm (RuntimeError) nếu môi trường không chạy được."""
    import pyspark

    info = {
        "python": sys.version.split()[0],
        "pyspark": pyspark.__version__,
        "java": _java_version(),
        "cuda": torch.cuda.is_available(),
        "fast_mode": FAST_MODE,
    }
    if _version_tuple(info["pyspark"]) < _version_tuple(min_spark):
        raise RuntimeError(
            f"PySpark {info['pyspark']} quá cũ: cần PySpark >= {min_spark} (predict_batch_udf có từ 3.4). "
            "Hãy dùng môi trường có Spark mới hơn (ví dụ Kaggle/Colab hiện tại có PySpark 4.x).")
    if not info["java"]:
        raise RuntimeError("Không tìm thấy Java. Spark cần Java 17+ — hãy cài JDK hoặc đặt biến JAVA_HOME.")
    if not info["cuda"]:
        msg = ("Không có GPU: huấn luyện trên CPU. Notebook PhoBERT sẽ tự dùng cấu hình rút gọn (FAST_MODE) "
               "để chạy xong trong thời gian hợp lý; mô hình RNN không bị ảnh hưởng.")
        if require_gpu:
            raise RuntimeError(msg)
        print("CẢNH BÁO:", msg)
    for k, v in info.items():
        print(f"{k:>10}: {v}")
    return info


# ---------------------------------------------------------------------------
# 3. Dữ liệu
# ---------------------------------------------------------------------------
def _has_all_files(folder: str, files) -> bool:
    return all(os.path.isfile(os.path.join(folder, f)) for f in files)


def find_or_download_data(local_dir: str = "./data", kaggle_root: str = "/kaggle/input",
                          url: str = DATA_URL, files=DATA_FILES) -> str:
    """Trả về THƯ MỤC chứa 3 file dữ liệu. Tìm theo thứ tự:
    (1) local_dir, rồi ../data (khi notebook chạy từ thư mục notebooks/ của repo);
    (2) mọi thư mục con của /kaggle/input; (3) tải từ GitHub về local_dir."""
    for folder in (local_dir, os.path.join("..", "data")):
        if _has_all_files(folder, files):
            return folder
    for hit in sorted(glob.glob(os.path.join(kaggle_root, "**", files[0]), recursive=True)):
        folder = os.path.dirname(hit)
        if _has_all_files(folder, files):
            return folder
    os.makedirs(local_dir, exist_ok=True)
    downloaded = []
    try:
        for name in files:
            dest = os.path.join(local_dir, name)
            with urllib.request.urlopen(f"{url}/{name}", timeout=30) as resp, open(dest, "wb") as fh:
                fh.write(resp.read())
            downloaded.append(dest)
    except Exception as exc:  # không có mạng, repo Private, URL sai...
        for path in downloaded:
            os.remove(path)
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu {DATASET_REF} ({', '.join(files)}). Hãy làm MỘT trong các cách:\n"
            f"  1) Đặt 3 file vào thư mục {os.path.abspath(local_dir)}\n"
            f"  2) Trên Kaggle: Add Input → dataset '{DATASET_REF}'\n"
            f"  3) Bật Internet để tải tự động từ {url}\n"
            f"(Lỗi khi tải: {exc})") from exc
    print(f"Đã tải dữ liệu từ {url} về {local_dir}")
    return local_dir


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")  # chữ cái lặp >= 3 lần ("đẹppppp" → "đẹp"); không đụng chữ số


def clean_text(text) -> str:
    """Tiền xử lý cơ bản: NFC, chữ thường, bỏ URL, rút gọn chữ cái lặp, gộp khoảng trắng."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _URL_RE.sub(" ", text)
    text = text.replace("￼", " ")  # ký tự "object replacement" hay gặp khi copy từ Shopee
    text = _REPEAT_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _read_raw(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    df = pd.DataFrame({"text": raw[RAW_TEXT_COL].map(clean_text)})
    for raw_col, aspect in RAW_ASPECT_COLS.items():
        mapped = raw[raw_col].map(ORIG_TO_IDX)
        if mapped.isna().any():
            bad = sorted(raw.loc[mapped.isna(), raw_col].unique().tolist())
            raise ValueError(f"Cột {raw_col} có nhãn lạ {bad}; chỉ chấp nhận {list(ORIG_TO_IDX)}")
        df[aspect] = mapped.astype("int64")
    return df[["text", *ASPECTS]]


def load_dataset(path: str) -> pd.DataFrame:
    """Đọc dữ liệu → cột ["text", *ASPECTS] (nhãn nội bộ 0..3).
    `path` là thư mục chứa 3 file (thêm cột "split") hoặc một file CSV."""
    if os.path.isdir(path):
        parts = []
        for name, split in zip(DATA_FILES, SPLIT_NAMES):
            part = _read_raw(os.path.join(path, name))
            part["split"] = split
            parts.append(part)
        return pd.concat(parts, ignore_index=True)
    return _read_raw(path)


def get_splits(df: pd.DataFrame, seed: int = SEED):
    """Trả về (train, val, test). Dùng cột "split" nếu có, ngược lại chia ngẫu nhiên 80/10/10."""
    if "split" in df.columns:
        return tuple(df[df["split"] == s].drop(columns="split") for s in SPLIT_NAMES)
    order = np.random.RandomState(seed).permutation(len(df))
    n_train, n_val = int(0.8 * len(df)), int(0.1 * len(df))
    idx = (order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:])
    return tuple(df.iloc[np.sort(i)] for i in idx)


def resolve_train_config(base: dict, fast: bool = None) -> dict:
    """Trả về bản sao cấu hình; khi fast (mặc định = FAST_MODE) thì rút gọn để chạy được trên CPU."""
    cfg = dict(base)
    fast = FAST_MODE if fast is None else fast
    if fast:
        cfg["epochs"] = 1
        cfg["max_len"] = min(cfg.get("max_len", 64), 64)
        cfg["train_frac"] = min(cfg.get("train_frac", 1.0), 0.2)
        print("FAST_MODE: dùng cấu hình rút gọn", cfg)
    return cfg


# ---------------------------------------------------------------------------
# 4. Huấn luyện
# ---------------------------------------------------------------------------
def ce_loss(logits, labels):
    """CrossEntropy trung bình trên 8 khía cạnh. logits [B,8,4], labels [B,8]."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def argmax_labels(out):
    """Chuyển logits [B,8,4] → nhãn [B,8]."""
    return out.argmax(dim=-1)


def macro_f1_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Trung bình theo khía cạnh của macro-F1; mỗi khía cạnh chỉ tính trên các nhãn có mặt."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for j in range(y_true.shape[1]):
        labels = sorted(set(y_true[:, j].tolist()) | set(y_pred[:, j].tolist()))
        scores.append(f1_score(y_true[:, j], y_pred[:, j], labels=labels, average="macro", zero_division=0))
    return float(np.mean(scores))


def _to_device(batch: dict, device):
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    labels = batch["labels"].to(device) if "labels" in batch else None
    return inputs, labels


def predict_labels(model, loader, to_labels=argmax_labels, device=None) -> np.ndarray:
    """Dự đoán nhãn [N,8] cho toàn bộ loader."""
    device = device or DEVICE
    model.to(device).eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            inputs, _ = _to_device(batch, device)
            preds.append(to_labels(model(**inputs)).cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros((0, len(ASPECTS)), dtype=np.int64)


def fit(model, train_loader, val_loader, loss_fn=ce_loss, to_labels=argmax_labels,
        epochs: int = 3, lr: float = 2e-5, device=None, weight_decay: float = 0.01) -> dict:
    """Huấn luyện với AdamW; sau mỗi epoch đo macro-F1 trên val; cuối cùng nạp lại epoch tốt nhất.
    Mỗi batch là dict có khóa "labels"; các khóa còn lại được truyền vào model(**batch)."""
    device = device or DEVICE
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_val = np.concatenate([b["labels"].numpy() for b in val_loader])
    history = {"train_loss": [], "val_macro_f1": []}
    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            inputs, labels = _to_device(batch, device)
            loss = loss_fn(model(**inputs), labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
            n += len(labels)
        val_f1 = macro_f1_mean(y_val, predict_labels(model, val_loader, to_labels, device))
        history["train_loss"].append(total / max(n, 1))
        history["val_macro_f1"].append(val_f1)
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch}/{epochs} - loss {total / max(n, 1):.4f} - val macro-F1 {val_f1:.4f} "
              f"- {time.time() - start:.0f}s")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Nạp lại trọng số tốt nhất: val macro-F1 = {best_f1:.4f}")
    return history


# ---------------------------------------------------------------------------
# 5. Spark
# ---------------------------------------------------------------------------
from pyspark.sql import functions as SF  # noqa: E402  (đặt tên SF để không trùng torch.nn.functional)
from pyspark.sql.types import IntegerType, StructField, StructType  # noqa: E402

# Kiểu trả về của UDF dự đoán: 8 trường số nguyên, tên theo ASPECTS
PRED_SCHEMA = StructType([StructField(a, IntegerType()) for a in ASPECTS])


def create_spark(app_name: str = "ABSA", master: str = "local[2]"):
    """Dùng lại SparkSession có sẵn (Databricks...) hoặc tạo session local.
    local[2]: mỗi worker Python nạp một bản mô hình, giới hạn 2 để không hết bộ nhớ GPU."""
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (SparkSession.builder.appName(app_name).master(master)
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .getOrCreate())
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    if _version_tuple(spark.version) < (3, 4):
        raise RuntimeError(f"Spark {spark.version} quá cũ: cần Spark >= 3.4 cho predict_batch_udf.")
    return spark


def to_spark_df(spark, df: pd.DataFrame):
    """pandas → Spark DataFrame: id (long), text (string), 8 cột nhãn (double, cho evaluator)."""
    pdf = df[["text", *ASPECTS]].reset_index(drop=True).copy()
    pdf.insert(0, "id", np.arange(len(pdf), dtype="int64"))
    pdf["text"] = pdf["text"].fillna("").astype(str)
    pdf[ASPECTS] = pdf[ASPECTS].astype("float64")
    return spark.createDataFrame(pdf)


def explode_predictions(sdf, struct_col: str = "pred"):
    """Tách cột struct dự đoán thành 8 cột pred_<aspect> kiểu double."""
    for a in ASPECTS:
        sdf = sdf.withColumn(f"pred_{a}", SF.col(f"{struct_col}.{a}").cast("double"))
    return sdf


def evaluate_with_spark(sdf) -> pd.DataFrame:
    """Đánh giá từng khía cạnh bằng MulticlassClassificationEvaluator.
    Cần cột nhãn <aspect> và cột dự đoán pred_<aspect> (double).
    Trả về bảng: accuracy, f1_weighted, f1_macro (macro trên các nhãn có mặt) + dòng Mean."""
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    sdf = sdf.cache()
    rows = {}
    for a in ASPECTS:
        ev = MulticlassClassificationEvaluator(labelCol=a, predictionCol=f"pred_{a}")
        present = sorted({r[0] for r in sdf.select(a).union(sdf.select(f"pred_{a}")).distinct().collect()})
        f1_by_label = [ev.evaluate(sdf, {ev.metricName: "fMeasureByLabel", ev.metricLabel: float(k)})
                       for k in present]
        rows[a] = {
            "accuracy": ev.evaluate(sdf, {ev.metricName: "accuracy"}),
            "f1_weighted": ev.evaluate(sdf, {ev.metricName: "f1"}),
            "f1_macro": float(np.mean(f1_by_label)),
        }
    sdf.unpersist()
    result = pd.DataFrame.from_dict(rows, orient="index")[["accuracy", "f1_weighted", "f1_macro"]]
    result.loc["Mean"] = result.mean()
    return result


def ship_model_to_spark(spark, model_dir: str) -> str:
    """Gửi thư mục mô hình tới mọi executor (chạy được cả local lẫn cluster).
    Chép sang thư mục tên duy nhất để chạy lại notebook không bị lỗi "file already added".
    Trả về "tên" để dùng trong UDF qua shipped_model_path(<tên>).

    - Cluster (Databricks, YARN...): SparkContext.addFile gửi bản chép tới từng executor.
    - Local mode (Kaggle, Colab, Jupyter trên máy): driver và worker cùng một máy nên worker đọc thẳng
      bản chép qua đường dẫn tuyệt đối. Không gọi addFile ở đây vì trên Windows thiếu winutils của Hadoop,
      một lần addFile lỗi sẽ làm MỌI job Spark sau đó thất bại."""
    base = os.path.basename(os.path.normpath(model_dir))
    name = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    dest = os.path.join(tempfile.gettempdir(), "absa_ship", name)
    shutil.copytree(model_dir, dest)
    if spark.sparkContext.master.startswith("local"):
        return dest
    spark.sparkContext.addFile(dest, recursive=True)
    return name


def shipped_model_path(shipped_name: str) -> str:
    """Dùng BÊN TRONG UDF: trả về thư mục mô hình trên worker (từ SparkFiles, hoặc đường dẫn cục bộ
    nếu ship_model_to_spark đã phải dùng phương án dự phòng)."""
    from pyspark import SparkFiles

    if os.path.isabs(shipped_name) and os.path.isdir(shipped_name):
        return shipped_name
    return SparkFiles.get(shipped_name)
