# Kế hoạch thực hiện: ABSA tiếng Việt trên Spark

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Mục tiêu:** 5 notebook ABSA (RNN và PhoBERT, đóng gói bằng `pandas_udf` / `predict_batch_udf`, đánh giá bằng Spark evaluator, thí nghiệm pooling/loss), mỗi notebook chạy độc lập trên mọi công cụ notebook có Spark.

**Kiến trúc:** Code được viết và kiểm thử dưới dạng các file Python trong `common/`, mỗi file là **một khối dán** nguyên văn vào một ô của notebook (cách A trong thiết kế). Test `pytest` chạy trên chính các file đó. Notebook là phần "lắp ráp": dán các khối, gọi hàm, trình bày Markdown. Script `tools/check_notebooks.py` kiểm tra mọi notebook đang dùng đúng phiên bản khối chung.

**Công nghệ:** Python 3.10+, PyTorch, Hugging Face `transformers` (PhoBERT), `pyvi`, PySpark ≥ 3.4 (`pandas_udf`, `pyspark.ml.functions.predict_batch_udf`, `MulticlassClassificationEvaluator`), pandas, scikit-learn, pytest.

**Thiết kế:** `docs/superpowers/specs/2026-11-13-absa-spark-design.md`. Người thực hiện đọc cả hai tài liệu.

**Người làm:** A (RNN + code chung + loss), B (PhoBERT + NB5). Mỗi task ghi chủ sở hữu và ngày dự kiến.

## Ràng buộc chung

- PySpark ≥ 3.4; Java 17+; không `pip install` đè `pyspark` hoặc `torch` trong notebook.
- 8 khía cạnh theo đúng thứ tự: `Price, Shipping, Outlook, Quality, Size, Shop_Service, General, Others`.
- Nhãn gốc `None=-1, Negative=0, Positive=1, Neutral=2` → nhãn nội bộ `0=None, 1=Negative, 2=Positive, 3=Neutral`.
- `SEED = 42`; chia 80/10/10 nếu dataset không có sẵn split.
- Không viết cứng `/kaggle/working` hay `/kaggle/input/<tên cụ thể>`; mô hình lưu ở `./models/`.
- Dữ liệu tìm theo thứ tự `./data/` → `/kaggle/input/**` → GitHub raw của repo (repo Public).
- Mô hình tới executor qua `addFile` + `SparkFiles.get`; không đọc đường dẫn cục bộ của driver trong UDF.
- Không có GPU → `FAST_MODE = True` → cấu hình rút gọn.
- Mỗi notebook đều có bước huấn luyện; NB5 bản nộp chạy lại toàn bộ thí nghiệm trong một lần (2–3 epoch, `max_len = 128`).
- Chú thích và Markdown bằng tiếng Việt; notebook theo khung 12 mục (thiết kế mục 6.7), ô đầu là `## Hướng dẫn chạy`.
- Tên file nộp: `NhomX_danhsachTen_danhsachMSSV_ABSASpark_NB<k>.ipynb`.
- Thời gian chạy mục tiêu (GPU): NB1/NB2 < 15 phút, NB3/NB4 < 30 phút, NB5 < 2 giờ.

## Quy ước khối dán

- Dòng đầu mỗi file trong `common/`: `# === COMMON <tên file> v<N> ===` (tăng `N` mỗi lần sửa).
- File nào dùng tên từ file khác thì import có bảo vệ, để vừa chạy được trong test vừa dán được vào notebook:

```python
try:  # khi chạy pytest
    from common.common_cells import ASPECTS, PRED_SCHEMA
except ImportError:  # trong notebook: đã có sẵn từ ô code chung phía trên
    pass
```

- Thứ tự dán trong notebook: `common_cells.py` (mục 3) → `rnn_cells.py` hoặc `bert_cells.py` (mục 6) → `losses.py` (chỉ NB5, mục 6) → `udf_pandas.py` hoặc `udf_batch.py` (mục 9).

## Điểm cần review kỹ

Những tình huống thiết kế ngầm yêu cầu nhưng dễ bị bỏ qua. Mỗi dòng đã có test ở task sở hữu:

1. Bình luận rỗng, `None`, toàn emoji → UDF vẫn trả đủ 8 nhãn, không lỗi (Task 2, Task 5).
2. Chạy lại notebook trong cùng phiên (huấn luyện lại, `addFile` lần 2) → không lỗi "file already added", UDF nạp mô hình mới nhất (Task 4).
3. Một lớp không xuất hiện trong tập test của một khía cạnh → `evaluate_with_spark` không trả NaN, macro-F1 tính trên các lớp có mặt (Task 4).
4. Không tải được dữ liệu (không có Internet, repo Private) → báo lỗi rõ ràng, có tên dataset và cách khắc phục (Task 2).
5. Môi trường không có GPU → tự rút gọn cấu hình; Spark < 3.4 hoặc thiếu Java → dừng sớm với thông báo tiếng Việt (Task 2).

---

### Task 1: Repo, dữ liệu, môi trường — A + B, ngày 1

**Files:**
- Create: `README.md`, `.gitignore`, `requirements-dev.txt`, `data/<file dữ liệu gốc>`, `data/SOURCE.md`, `tests/conftest.py`, `tests/fixtures/mini.csv`

**Interfaces:**
- Produces: fixture pytest `spark` (SparkSession `local[2]`, scope `session`); file `tests/fixtures/mini.csv` (20 dòng đầu của dữ liệu gốc, giữ nguyên định dạng và tên cột gốc).

- [ ] **Bước 1 (A):** Tạo repo GitHub `absa-spark` (Public), mời B làm collaborator. Tạo cấu trúc thư mục như thiết kế mục 5.1.
- [ ] **Bước 2 (A):** Tải dataset `cthng123/absa-vietnamese` từ Kaggle, chép vào `data/`. Ghi vào `data/SOURCE.md`: nguồn, giấy phép CC BY 4.0, tác giả. Ghi lại trong README: tên file, tên cột văn bản, tên 8 cột khía cạnh, có sẵn split không, có cột độ tin cậy không, số dòng.
- [ ] **Bước 3 (A):** Tạo `tests/fixtures/mini.csv` từ 20 dòng đầu; đảm bảo có ít nhất một dòng chứa mỗi nhãn `-1, 0, 1, 2`.
- [ ] **Bước 4 (A):** `requirements-dev.txt`: `pytest`, `pyspark>=3.4`, `torch`, `transformers`, `pyvi`, `pandas`, `numpy`, `scikit-learn`, `pyarrow`. `.gitignore`: `models/`, `*.pt`, `*.bin`, `*.safetensors`, `__pycache__/`, `.ipynb_checkpoints/`, `spark-warehouse/`. `tests/conftest.py`: fixture `spark`, thêm thư mục gốc repo vào `sys.path`.
- [ ] **Bước 5 (B):** Xác minh số điện thoại tài khoản Kaggle. Trên Kaggle (GPU + Internet) chạy ô kiểm tra trong thiết kế: phải in `4.0.x`, `openjdk 21`, `5`. Thử `pip install pyvi` và tải `vinai/phobert-base-v2`. Ghi kết quả vào README.
- [ ] **Bước 6:** Chạy `pytest -q` (máy có Java; hoặc trong ô Kaggle: `!git clone <repo> && cd absa-spark && pip install -q pytest pyvi && pytest -q`). Kết quả mong đợi: `no tests ran`, không lỗi import.
- [ ] **Bước 7: Commit** — `git add . && git commit -m "chore: khung repo, dữ liệu, môi trường test"`

### Task 2: `common_cells.py` — dữ liệu và môi trường — A, ngày 1

**Files:**
- Create: `common/__init__.py` (rỗng), `common/common_cells.py`
- Test: `tests/test_common_data.py`

**Interfaces:**
- Produces (trong `common/common_cells.py`):
  - `COMMON_VERSION: int`, `ASPECTS: list[str]`, `LABEL_NAMES = ["None","Negative","Positive","Neutral"]`, `ORIG_TO_IDX = {-1:0, 0:1, 1:2, 2:3}`, `SEED = 42`, `DATASET_REF = "cthng123/absa-vietnamese"`
  - `RAW_TEXT_COL: str`, `RAW_ASPECT_COLS: dict[str, str]` (tên cột gốc → tên trong `ASPECTS`), `DATA_FILENAME: str`, `DATA_URL: str` (raw GitHub) — giá trị lấy từ Bước 2 của Task 1
  - `DEVICE: str` (`"cuda"`/`"cpu"`), `FAST_MODE: bool` (mặc định `DEVICE == "cpu"`)
  - `set_seed(seed: int = SEED) -> None`
  - `check_environment(min_spark: str = "3.4") -> dict` — trả về `{"python","pyspark","java","cuda","fast_mode"}`; ném `RuntimeError` (thông báo tiếng Việt) khi thiếu Java hoặc Spark thấp hơn `min_spark`
  - `find_or_download_data(local_dir="./data", kaggle_root="/kaggle/input", url=DATA_URL, filename=DATA_FILENAME) -> str`
  - `clean_text(text: str | None) -> str`
  - `load_dataset(path: str) -> pd.DataFrame` — cột `["text", *ASPECTS]` (+ `"split"` nếu dataset có sẵn split); nhãn là `int` trong `0..3`; `text` đã qua `clean_text`
  - `get_splits(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` — dùng cột `split` nếu có, ngược lại chia ngẫu nhiên 80/10/10
  - `resolve_train_config(base: dict, fast: bool = FAST_MODE) -> dict`

- [ ] **Bước 1: Viết test thất bại**

```python
from common.common_cells import *

def test_clean_text():
    assert clean_text(None) == ""
    assert clean_text("  Giày ĐẸPPPPP lắm   https://shopee.vn/x ") == "giày đẹp lắm"
    assert isinstance(clean_text("😍😍😍"), str)  # chỉ cần không lỗi

def test_load_dataset_maps_labels(tmp_path):
    df = load_dataset("tests/fixtures/mini.csv")
    assert list(df.columns[:9]) == ["text", *ASPECTS]
    assert set(df[ASPECTS].stack().unique()) <= {0, 1, 2, 3}
    assert len(df) == 20

def test_splits_are_deterministic_and_disjoint():
    df = pd.concat([load_dataset("tests/fixtures/mini.csv")] * 10, ignore_index=True)
    a, b = get_splits(df), get_splits(df)
    assert all(x.index.equals(y.index) for x, y in zip(a, b))
    tr, va, te = a
    assert len(tr) + len(va) + len(te) == len(df)
    assert not set(tr.index) & set(te.index)
    assert abs(len(tr) / len(df) - 0.8) < 0.05

def test_find_data_prefers_local(tmp_path):
    (tmp_path / DATA_FILENAME).write_text("x")
    assert find_or_download_data(local_dir=str(tmp_path), kaggle_root="/khong-co") == str(tmp_path / DATA_FILENAME)

def test_find_data_searches_kaggle_tree(tmp_path):
    p = tmp_path / "ten-bat-ky" / "sub"; p.mkdir(parents=True); (p / DATA_FILENAME).write_text("x")
    assert find_or_download_data(local_dir="/khong-co", kaggle_root=str(tmp_path)).endswith(DATA_FILENAME)

def test_find_data_clear_error_when_unreachable():
    with pytest.raises(FileNotFoundError, match="cthng123/absa-vietnamese"):
        find_or_download_data(local_dir="/khong-co", kaggle_root="/khong-co", url="http://127.0.0.1:9/khong-co.csv")

def test_fast_mode_shrinks_config():
    base = {"epochs": 3, "max_len": 128, "train_frac": 1.0, "batch_size": 32}
    assert resolve_train_config(base, fast=False) == base
    fast = resolve_train_config(base, fast=True)
    assert fast["epochs"] == 1 and fast["max_len"] <= 64 and fast["train_frac"] <= 0.2

def test_check_environment_reports(spark):
    info = check_environment()
    assert set(info) == {"python", "pyspark", "java", "cuda", "fast_mode"}

def test_check_environment_rejects_old_spark(spark):
    with pytest.raises(RuntimeError):
        check_environment(min_spark="99.0")
```

- [ ] **Bước 2:** `pytest tests/test_common_data.py -q` → FAIL (`ImportError`).
- [ ] **Bước 3: Viết code** theo các chữ ký trên. Ghi chú hiện thực:
  - `clean_text`: chuẩn hóa NFC, chữ thường, bỏ URL, rút ký tự lặp ≥3 lần về 1, gộp khoảng trắng.
  - `find_or_download_data`: tìm trong `/kaggle/input` bằng `glob(f"{kaggle_root}/**/{filename}", recursive=True)`. Tải bằng `urllib.request` vào `local_dir` (timeout 30 giây). Thông báo lỗi nêu `DATASET_REF` và 3 cách cung cấp dữ liệu.
  - `resolve_train_config(fast=True)`: `epochs=1`, `max_len=min(max_len, 64)`, `train_frac=min(train_frac, 0.2)`.
  - `check_environment`: lấy phiên bản Java từ `java -version`. Có thêm cờ `require_gpu=False`, chỉ in cảnh báo khi không có GPU, không dừng.
- [ ] **Bước 4:** `pytest tests/test_common_data.py -q` → PASS.
- [ ] **Bước 5: Commit** — `git commit -am "feat(common): đọc dữ liệu, tiền xử lý, kiểm tra môi trường"` (nhớ `git add` file mới).

### Task 3: `common_cells.py` — huấn luyện chung — A, ngày 1

**Files:**
- Modify: `common/common_cells.py`
- Test: `tests/test_common_train.py`

**Interfaces:**
- Consumes: `ASPECTS`, `DEVICE`, `set_seed` (Task 2)
- Produces:
  - `ce_loss(logits: Tensor[B,8,4], labels: LongTensor[B,8]) -> Tensor` (trung bình CE của 8 khía cạnh)
  - `argmax_labels(out: Tensor[B,8,4]) -> LongTensor[B,8]`
  - `macro_f1_mean(y_true: np.ndarray[N,8], y_pred: np.ndarray[N,8]) -> float` — trung bình theo khía cạnh của `f1_score(average="macro")`, mỗi khía cạnh chỉ tính trên các nhãn có trong `y_true ∪ y_pred`
  - `fit(model, train_loader, val_loader, loss_fn=ce_loss, to_labels=argmax_labels, epochs=3, lr=2e-5, device=DEVICE) -> dict` — mỗi batch là `dict` có khóa `"labels"`, các khóa còn lại truyền vào `model(**batch)`. Trả `{"train_loss": [...], "val_macro_f1": [...]}`; cuối cùng nạp lại trạng thái có `val_macro_f1` cao nhất.
  - `predict_labels(model, loader, to_labels=argmax_labels, device=DEVICE) -> np.ndarray[N,8]`

- [ ] **Bước 1: Viết test thất bại**

```python
def test_ce_loss_shape_and_value():
    logits = torch.zeros(2, 8, 4); labels = torch.zeros(2, 8, dtype=torch.long)
    assert torch.isclose(ce_loss(logits, labels), torch.log(torch.tensor(4.0)))

def test_macro_f1_perfect_and_missing_class():
    y = np.array([[0]*8, [1]*8, [2]*8])
    assert macro_f1_mean(y, y) == 1.0
    assert 0.0 <= macro_f1_mean(y, np.zeros_like(y)) < 1.0  # không NaN

def test_fit_learns_toy_problem():
    set_seed(0)
    X = torch.randn(64, 5); Y = (X[:, :1] > 0).long().repeat(1, 8)
    class Toy(torch.nn.Module):
        def __init__(self): super().__init__(); self.l = torch.nn.Linear(5, 32)
        def forward(self, x): return self.l(x).view(-1, 8, 4)
    loader = [{"x": X[i:i+16], "labels": Y[i:i+16]} for i in range(0, 64, 16)]
    hist = fit(Toy(), loader, loader, epochs=30, lr=0.05, device="cpu")
    assert max(hist["val_macro_f1"]) > 0.9
    assert len(hist["train_loss"]) == 30
```

- [ ] **Bước 2:** `pytest tests/test_common_train.py -q` → FAIL.
- [ ] **Bước 3:** Viết các hàm theo chữ ký. `fit` dùng `AdamW`, in loss và macro-F1 mỗi epoch, giữ `copy.deepcopy(model.state_dict())` của epoch tốt nhất.
- [ ] **Bước 4:** `pytest tests/test_common_train.py -q` → PASS.
- [ ] **Bước 5: Commit** — `"feat(common): vòng lặp huấn luyện và macro-F1"`

### Task 4: `common_cells.py` — Spark: tạo session, đánh giá, gửi mô hình — A, ngày 1–2

**Files:**
- Modify: `common/common_cells.py`
- Test: `tests/test_common_spark.py`

**Interfaces:**
- Consumes: `ASPECTS`, `LABEL_NAMES`
- Produces:
  - `create_spark(app_name="ABSA", master="local[2]") -> SparkSession` — `builder.getOrCreate()` (dùng lại session có sẵn), bật `spark.sql.execution.arrow.pyspark.enabled`
  - `PRED_SCHEMA: StructType` — 8 trường `IntegerType`, tên theo `ASPECTS`
  - `to_spark_df(spark, df: pd.DataFrame) -> DataFrame` — cột `id` (long), `text` (string), 8 cột khía cạnh (double)
  - `explode_predictions(sdf, struct_col="pred") -> DataFrame` — thêm `pred_<aspect>` (double) cho mỗi khía cạnh
  - `evaluate_with_spark(sdf) -> pd.DataFrame` — index `ASPECTS + ["Mean"]`, cột `["accuracy", "f1_weighted", "f1_macro"]`; dùng `MulticlassClassificationEvaluator` (`accuracy`, `f1`, và `fMeasureByLabel` với `metricLabel` cho từng nhãn có mặt trong label ∪ prediction của khía cạnh đó)
  - `ship_model_to_spark(spark, model_dir: str) -> str` — chép `model_dir` sang thư mục tên duy nhất `<basename>_<timestamp>`, `addFile(..., recursive=True)`, trả về tên để dùng với `SparkFiles.get`

- [ ] **Bước 1: Viết test thất bại**

```python
def test_evaluate_perfect(spark):
    pdf = pd.DataFrame({"text": list("abcd"), **{a: [0, 1, 2, 0] for a in ASPECTS}})
    sdf = to_spark_df(spark, pdf)
    for a in ASPECTS: sdf = sdf.withColumn(f"pred_{a}", F.col(a))
    res = evaluate_with_spark(sdf)
    assert list(res.index) == ASPECTS + ["Mean"]
    assert (res["accuracy"] == 1.0).all() and (res["f1_macro"] == 1.0).all()

def test_evaluate_missing_class_no_nan(spark):  # nhãn 3 (Neutral) không xuất hiện
    pdf = pd.DataFrame({"text": list("abcd"), **{a: [0, 0, 1, 1] for a in ASPECTS}})
    sdf = to_spark_df(spark, pdf)
    for a in ASPECTS: sdf = sdf.withColumn(f"pred_{a}", F.lit(0.0))
    res = evaluate_with_spark(sdf)
    assert not res.isna().any().any()
    assert res.loc["Mean", "accuracy"] == 0.5

def test_explode_predictions(spark):
    sdf = spark.createDataFrame([(dict(zip(ASPECTS, range(8))),)], StructType([StructField("pred", PRED_SCHEMA)]))
    out = explode_predictions(sdf)
    assert [out.first()[f"pred_{a}"] for a in ASPECTS] == [float(i) for i in range(8)]

def test_ship_model_twice_is_safe(spark, tmp_path):
    d = tmp_path / "m"; d.mkdir(); (d / "w.txt").write_text("v1")
    n1 = ship_model_to_spark(spark, str(d))
    (d / "w.txt").write_text("v2")
    n2 = ship_model_to_spark(spark, str(d))
    assert n1 != n2
    got = spark.sparkContext.parallelize([0], 1).map(
        lambda _: open(os.path.join(SparkFiles.get(n2), "w.txt")).read()).collect()
    assert got == ["v2"]
```

- [ ] **Bước 2:** `pytest tests/test_common_spark.py -q` → FAIL.
- [ ] **Bước 3:** Viết các hàm theo chữ ký. Hàng `Mean` là trung bình 8 hàng phía trên.
- [ ] **Bước 4:** `pytest tests/test_common_spark.py -q` → PASS; `pytest -q` (toàn bộ) → PASS.
- [ ] **Bước 5: Commit** — `"feat(common): Spark session, đánh giá theo khía cạnh, gửi mô hình tới executor"`. Báo B: **`common_cells.py` v1 đã sẵn sàng.**

### Task 5: Hai khối UDF — A, ngày 2 (sáng)

**Files:**
- Create: `common/udf_pandas.py`, `common/udf_batch.py`
- Test: `tests/test_udfs.py`

**Interfaces:**
- Consumes: `ASPECTS`, `PRED_SCHEMA`, `ship_model_to_spark`, `explode_predictions` (Task 4)
- **Hợp đồng predictor** (dùng ở Task 7, 8): `load_predictor(model_dir: str) -> Callable[[list[str]], np.ndarray]`. Predictor nhận danh sách văn bản thô (có thể chứa `""`), tự làm sạch, trả mảng `int` hình `(N, 8)` giá trị `0..3`.
- Produces:
  - `make_pandas_udf(load_predictor, shipped_name: str) -> UserDefinedFunction` — `@pandas_udf(PRED_SCHEMA)` dạng `Iterator[pd.Series] -> Iterator[pd.DataFrame]`. Nạp predictor **một lần** từ `SparkFiles.get(shipped_name)` trước vòng lặp; thay `None` bằng `""`.
  - `make_batch_udf(load_predictor, shipped_name: str, batch_size: int = 64) -> UserDefinedFunction` — `predict_batch_udf(make_predict_fn, return_type=PRED_SCHEMA, batch_size=batch_size)`. `make_predict_fn` nạp predictor từ `SparkFiles.get`; `predict(inputs: np.ndarray) -> dict[str, np.ndarray]` (mỗi khía cạnh một mảng).

- [ ] **Bước 1: Viết test thất bại** (predictor giả, không cần torch)

```python
def fake_loader(model_dir):
    k = int(open(os.path.join(model_dir, "k.txt")).read())
    return lambda texts: np.array([[min(len(t), 3) if k else 0] * 8 for t in texts])

@pytest.fixture
def shipped(spark, tmp_path):
    d = tmp_path / "fake"; d.mkdir(); (d / "k.txt").write_text("1")
    return ship_model_to_spark(spark, str(d))

TEXTS = ["", None, "ab", "abcdef"]
EXPECTED = [0, 0, 2, 3]

@pytest.mark.parametrize("factory", ["pandas", "batch"])
def test_udf_predicts_8_labels_and_handles_empty(spark, shipped, factory):
    udf = make_pandas_udf(fake_loader, shipped) if factory == "pandas" else make_batch_udf(fake_loader, shipped, batch_size=2)
    sdf = spark.createDataFrame(pd.DataFrame({"text": TEXTS})).repartition(2)
    out = explode_predictions(sdf.withColumn("pred", udf("text"))).toPandas().set_index("text", drop=False)
    for t, e in zip(TEXTS, EXPECTED):
        row = out[out["text"].isna()] if t is None else out[out["text"] == t]
        assert [row.iloc[0][f"pred_{a}"] for a in ASPECTS] == [float(e)] * 8
```

Với `predict_batch_udf`, nếu Spark không cho giá trị `None` đi vào predictor, thì `fillna("")` trước khi gọi UDF và ghi rõ điều này trong docstring. Test phải giữ nguyên kỳ vọng: không lỗi, 8 nhãn.

- [ ] **Bước 2:** `pytest tests/test_udfs.py -q` → FAIL.
- [ ] **Bước 3:** Viết 2 file. Mỗi file có docstring tiếng Việt giải thích: vì sao dùng iterator (nạp mô hình một lần); vì sao dùng `SparkFiles` (chạy được trên cluster); `batch_size` có tác dụng gì.
- [ ] **Bước 4:** `pytest tests/test_udfs.py -q` → PASS.
- [ ] **Bước 5: Commit** — `"feat(common): khối pandas_udf và predict_batch_udf"`. Báo B.

### Task 6: Script kiểm tra notebook — A, ngày 2

**Files:**
- Create: `tools/check_notebooks.py`
- Test: `tests/test_check_notebooks.py`

**Interfaces:**
- Produces: `check_notebook(nb_path: str, common_dir: str = "common") -> list[str]` (danh sách lỗi, rỗng là đạt). CLI: `python tools/check_notebooks.py notebooks/*.ipynb` → in lỗi, exit 1 nếu có lỗi.
- Các quy tắc kiểm tra:
  1. Ô đầu tiên là Markdown bắt đầu bằng `## Hướng dẫn chạy`.
  2. Mọi ô code bắt đầu bằng `# === COMMON <file> v<N> ===` phải giống hệt `common/<file>` (bỏ khoảng trắng đầu/cuối).
  3. Không chứa chuỗi `/kaggle/working`.
  4. Có ít nhất một ô code có output.

- [ ] **Bước 1: Viết test thất bại** — dựng notebook giả bằng `nbformat.v4.new_notebook()` trong `tmp_path`, mỗi test cho một quy tắc:

```python
def test_ok_notebook_has_no_problems(tmp_path): ...            # assert check_notebook(p) == []
def test_stale_common_block_detected(tmp_path): ...             # khối v1 cũ khi file đã v2 → lỗi chứa "common_cells.py"
def test_hardcoded_kaggle_working_detected(tmp_path): ...       # lỗi chứa "/kaggle/working"
def test_missing_run_guide_detected(tmp_path): ...              # lỗi chứa "Hướng dẫn chạy"
def test_no_outputs_detected(tmp_path): ...                     # lỗi chứa "output"
```

Mỗi test dùng một thư mục `common_dir` giả trong `tmp_path` có file `common_cells.py` với nội dung biết trước.
- [ ] **Bước 2:** `pytest tests/test_check_notebooks.py -q` → FAIL.
- [ ] **Bước 3:** Viết `check_notebook` (dùng `json`/`nbformat`) và `main()`. Thêm `nbformat` vào `requirements-dev.txt`.
- [ ] **Bước 4:** `pytest tests/test_check_notebooks.py -q` → PASS.
- [ ] **Bước 5: Commit** — `"feat(tools): kiểm tra notebook dùng đúng khối chung"`

### Task 7: `rnn_cells.py` — mô hình RNN — A, ngày 2

**Files:**
- Create: `common/rnn_cells.py`
- Test: `tests/test_rnn_cells.py`

**Interfaces:**
- Consumes: `clean_text`, `ASPECTS`, `DEVICE` (Task 2), hợp đồng predictor (Task 5)
- Produces:
  - `build_vocab(texts: list[str], min_freq: int = 2) -> dict[str, int]` — `"<pad>": 0`, `"<unk>": 1`
  - `encode(texts: list[str], vocab: dict, max_len: int) -> np.ndarray` (int64, `[N, max_len]`)
  - `class RnnAbsa(nn.Module)`: `__init__(vocab_size, emb_dim=200, hidden=128, n_aspects=8, n_classes=4, dropout=0.3)`; `forward(input_ids) -> Tensor[B,8,4]`. Bên trong là BiLSTM + pooling; có thêm nhánh CNN hay không do A quyết định.
  - `make_rnn_loaders(train_df, val_df, vocab, max_len, batch_size=64) -> tuple[DataLoader, DataLoader]` — batch dạng `{"input_ids", "labels"}`
  - `save_rnn(model, vocab, cfg: dict, out_dir: str) -> None` — `model.pt`, `vocab.json`, `config.json`
  - `load_rnn_predictor(model_dir: str) -> Callable[[list[str]], np.ndarray]`

- [ ] **Bước 1: Viết test thất bại**

```python
def test_vocab_and_encode():
    v = build_vocab(["giày đẹp", "giày xấu", "giày đẹp"], min_freq=2)
    assert v["<pad>"] == 0 and v["<unk>"] == 1 and "giày" in v and "xấu" not in v
    x = encode(["giày đẹp quá", ""], v, max_len=5)
    assert x.shape == (2, 5) and x[1].sum() == 0 and x[0, 2] == 1

def test_model_shape():
    m = RnnAbsa(vocab_size=50)
    assert m(torch.randint(0, 50, (3, 7))).shape == (3, 8, 4)

def test_save_load_predictor_roundtrip(tmp_path):
    v = build_vocab(["a b", "a b"], min_freq=1); m = RnnAbsa(len(v))
    save_rnn(m, v, {"max_len": 6, "emb_dim": 200, "hidden": 128}, str(tmp_path))
    pred = load_rnn_predictor(str(tmp_path))(["a b", "", "😍"])
    assert pred.shape == (3, 8) and pred.dtype.kind == "i" and set(np.unique(pred)) <= {0, 1, 2, 3}
```

- [ ] **Bước 2:** `pytest tests/test_rnn_cells.py -q` → FAIL.
- [ ] **Bước 3:** Viết code theo chữ ký. Tách từ theo khoảng trắng sau `clean_text`. Predictor chạy trên `cuda` nếu có, không thì `cpu`.
- [ ] **Bước 4:** `pytest tests/test_rnn_cells.py -q` → PASS.
- [ ] **Bước 5: Commit** — `"feat(rnn): mô hình BiLSTM 8 đầu và predictor"`

### Task 8: `bert_cells.py` — mô hình PhoBERT — B, ngày 1–2

**Files:**
- Create: `common/bert_cells.py`
- Test: `tests/test_bert_cells.py`

**Interfaces:**
- Consumes: `clean_text`, `DEVICE`, hợp đồng predictor (Task 5)
- Produces:
  - `MODEL_NAME = "vinai/phobert-base-v2"`
  - `segment(texts: list[str]) -> list[str]` — `clean_text` rồi `pyvi.ViTokenizer.tokenize`
  - `class PhoBertAbsa(nn.Module)`: `__init__(encoder: nn.Module, hidden_size: int, pooling: str = "cls", gated: bool = False, n_aspects: int = 8)`; `pooling ∈ {"cls", "mean", "max", "attention"}`; `forward(input_ids, attention_mask)`:
    - `gated=False` → `Tensor[B,8,4]`
    - `gated=True` → `{"mention": Tensor[B,8,2], "sentiment": Tensor[B,8,3]}`
    - Pooling `mean`/`max` bỏ qua token padding. `attention` học trọng số trên mọi hidden state của token (query chung hay riêng từng khía cạnh do B chọn, ghi trong docstring).
  - `gated_to_labels(out: dict) -> LongTensor[B,8]` — `mention` argmax = 0 → nhãn 0, ngược lại `sentiment.argmax + 1`
  - `build_phobert_absa(model_name=MODEL_NAME, pooling="cls", gated=False) -> tuple[PhoBertAbsa, tokenizer]`
  - `make_bert_loaders(train_df, val_df, tokenizer, max_len, batch_size=16) -> tuple[DataLoader, DataLoader]` — batch dạng `{"input_ids", "attention_mask", "labels"}`; văn bản đã qua `segment`
  - `save_bert(model, tokenizer, cfg: dict, out_dir: str) -> None`; `load_bert_predictor(model_dir: str, batch_size: int = 32) -> Callable[[list[str]], np.ndarray]`. `cfg` có `pooling`, `gated`, `max_len`, `model_name`. Khi nạp lại, dựng encoder từ config đã lưu, **không tải lại từ Internet**.

- [ ] **Bước 1: Viết test thất bại** — dùng encoder Roberta tí hon để không phải tải PhoBERT:

```python
from transformers import RobertaConfig, RobertaModel
def tiny():
    return RobertaModel(RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, intermediate_size=64, max_position_embeddings=40))

@pytest.mark.parametrize("pooling", ["cls", "mean", "max", "attention"])
def test_shapes(pooling):
    m = PhoBertAbsa(tiny(), 32, pooling=pooling)
    ids = torch.randint(5, 100, (2, 10)); mask = torch.ones_like(ids); mask[1, 6:] = 0
    assert m(ids, mask).shape == (2, 8, 4)

def test_mean_pooling_ignores_padding():
    torch.manual_seed(0); m = PhoBertAbsa(tiny(), 32, pooling="mean").eval()
    ids = torch.randint(5, 100, (1, 10)); mask = torch.ones_like(ids); mask[0, 6:] = 0
    ids2 = ids.clone(); ids2[0, 6:] = 7
    assert torch.allclose(m(ids, mask), m(ids2, mask), atol=1e-5)

def test_gated_outputs_and_labels():
    out = PhoBertAbsa(tiny(), 32, gated=True)(torch.randint(5, 100, (2, 8)), torch.ones(2, 8, dtype=torch.long))
    assert out["mention"].shape == (2, 8, 2) and out["sentiment"].shape == (2, 8, 3)
    fake = {"mention": torch.tensor([[[5., 0.]] * 8]), "sentiment": torch.tensor([[[0., 9., 0.]] * 8])}
    assert gated_to_labels(fake).tolist() == [[0] * 8]
    fake["mention"] = torch.tensor([[[0., 5.]] * 8])
    assert gated_to_labels(fake).tolist() == [[2] * 8]

def test_segment_handles_empty():
    out = segment(["", "Giày đẹp quá"])
    assert len(out) == 2 and out[0] == "" and out[1] != ""
```

(Không kiểm tra kết quả tách từ cụ thể, vì nó phụ thuộc vào `pyvi`.)
- [ ] **Bước 2:** `pytest tests/test_bert_cells.py -q` → FAIL.
- [ ] **Bước 3:** Viết code theo chữ ký.
- [ ] **Bước 4:** `pytest tests/test_bert_cells.py -q` → PASS.
- [ ] **Bước 5 (Kaggle, có Internet):** `build_phobert_absa()` → `save_bert` → `load_bert_predictor(...)(["giày đẹp", ""])` trả về mảng hình `(2, 8)`; sau khi đã lưu, nạp lại không cần Internet.
- [ ] **Bước 6: Commit** — `"feat(bert): PhoBERT 8 đầu, 4 kiểu pooling, gated head, predictor"`

### Task 9: NB1 + NB2 — RNN — A, ngày 2–3

**Files:**
- Create: `notebooks/01_rnn_pandas_udf.ipynb`, `notebooks/02_rnn_predict_batch_udf.ipynb`

**Interfaces:**
- Consumes: `common_cells.py`, `rnn_cells.py`, `udf_pandas.py` (NB1), `udf_batch.py` (NB2)

- [ ] **Bước 1:** Tạo NB1 trên Kaggle theo khung 12 mục. Ô 1 là `## Hướng dẫn chạy`: yêu cầu môi trường, cách cung cấp dữ liệu, thời gian chạy có/không GPU, link Kaggle Public. Ô cài đặt: `%pip install -q` các gói thiếu (không cài `pyspark`, `torch`). Ô 3 dán `common_cells.py` rồi gọi `check_environment()`, `set_seed()`.
- [ ] **Bước 2:** Mục 4–5: `find_or_download_data` → `load_dataset` → `get_splits`. EDA: biểu đồ phân bố nhãn theo khía cạnh, độ dài câu. Ghi lại `max_len` = phân vị 95% độ dài.
- [ ] **Bước 3:** Mục 6–8: dán `rnn_cells.py`; `cfg = resolve_train_config({"epochs": 10, "max_len": <p95>, "batch_size": 64, "lr": 1e-3, "train_frac": 1.0})`; `train = train.sample(frac=cfg["train_frac"], random_state=SEED)` (dòng này áp dụng cho cả NB3–5); `fit` (có vẽ đường loss và macro-F1); `save_rnn(..., "./models/rnn")`.
- [ ] **Bước 4:** Mục 9–11: `create_spark()` → `ship_model_to_spark` → dán `udf_pandas.py` → `make_pandas_udf(load_rnn_predictor, name)` → `to_spark_df(test)` → `withColumn("pred", udf("text"))` → `explode_predictions` → `evaluate_with_spark` → hiển thị bảng.
- [ ] **Bước 5:** Thêm ô kiểm tra cuối notebook:

```python
baseline = macro_f1_mean(test[ASPECTS].values, np.zeros((len(test), 8), dtype=int))  # luôn đoán None
assert list(result.index) == ASPECTS + ["Mean"]
assert result.loc["Mean", "f1_macro"] > baseline, "Mô hình không tốt hơn đoán toàn None"
```

- [ ] **Bước 6:** Mục 12, kết luận. Bật GPU + Internet, chạy "Run All" trên Kaggle → không lỗi, < 15 phút. Tải `.ipynb` về `notebooks/`.
- [ ] **Bước 7:** Tạo NB2 bằng cách nhân bản NB1, thay mục 9 bằng `udf_batch.py` + `make_batch_udf(...)`, sửa tiêu đề/Markdown cho đúng `predict_batch_udf`. Run All → không lỗi.
- [ ] **Bước 8:** `python tools/check_notebooks.py notebooks/01_*.ipynb notebooks/02_*.ipynb` → không lỗi.
- [ ] **Bước 9: Commit** — `"feat(nb): NB1 RNN+pandas_udf, NB2 RNN+predict_batch_udf"`

### Task 10: NB3 + NB4 — PhoBERT — B, ngày 2–3

**Files:**
- Create: `notebooks/03_bert_pandas_udf.ipynb`, `notebooks/04_bert_predict_batch_udf.ipynb`

**Interfaces:**
- Consumes: `common_cells.py`, `bert_cells.py`, `udf_pandas.py` (NB3), `udf_batch.py` (NB4)

- [ ] **Bước 1–2:** Giống Task 9 bước 1–2. Ở ô cài đặt thêm `pyvi`.
- [ ] **Bước 3:** Mục 6–8: dán `bert_cells.py`; `build_phobert_absa(pooling="cls")`; `cfg = resolve_train_config({"epochs": 3, "max_len": 128, "batch_size": 16, "lr": 2e-5, "train_frac": 1.0})`; `fit` → `save_bert(..., "./models/phobert")`.
- [ ] **Bước 4:** Mục 9–11 giống Task 9 bước 4, dùng `load_bert_predictor`. Spark `master="local[2]"` và có chú thích giải thích lý do (mỗi worker nạp một bản mô hình lên GPU).
- [ ] **Bước 5:** Ô kiểm tra cuối giống Task 9 bước 5.
- [ ] **Bước 6:** Run All trên Kaggle GPU → không lỗi, < 30 phút. Thử thêm một lần với `FAST_MODE = True` (giả lập không có GPU) → chạy hết, không lỗi.
- [ ] **Bước 7:** Tạo NB4 từ NB3, thay mục 9 bằng `make_batch_udf(load_bert_predictor, name, batch_size=32)`. Run All → không lỗi.
- [ ] **Bước 8:** `python tools/check_notebooks.py notebooks/03_*.ipynb notebooks/04_*.ipynb` → không lỗi.
- [ ] **Bước 9: Commit** — `"feat(nb): NB3 PhoBERT+pandas_udf, NB4 PhoBERT+predict_batch_udf"`

### Task 11: `losses.py` — Gated / Weighted / Focal — A, ngày 4

**Files:**
- Create: `common/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Consumes: `ce_loss`, `ASPECTS` (Task 3), dạng output gated (Task 8)
- Produces:
  - `gated_loss(out: dict, labels: LongTensor[B,8]) -> Tensor` — CE(mention, `labels > 0`) + CE(sentiment, `labels - 1`), trong đó phần sentiment **chỉ tính trên các vị trí có `labels > 0`** (nếu không có vị trí nào thì phần đó bằng 0)
  - `weighted_ce_loss(logits, labels, class_weights: Tensor[8,4]) -> Tensor`
  - `focal_loss(logits, labels, gamma: float = 2.0) -> Tensor`
  - `DEFAULT_CLASS_WEIGHTS: Tensor[8,4]` — mặc định 1.0; cột Neutral (chỉ số 3) × 0.5; hàng `Others` × 0.5. Nếu Task 1 phát hiện dataset có cột độ tin cậy, thêm `sample_weights` và ghi rõ trong docstring.
  - `make_loss(loss_type: str, class_weights=None, gamma: float = 2.0) -> Callable` — `loss_type ∈ {"ce", "gated", "weighted", "focal"}`

- [ ] **Bước 1: Viết test thất bại**

```python
def test_focal_gamma0_equals_ce():
    lg = torch.randn(4, 8, 4); y = torch.randint(0, 4, (4, 8))
    assert torch.isclose(focal_loss(lg, y, gamma=0.0), ce_loss(lg, y), atol=1e-6)

def test_focal_downweights_easy_examples():
    y = torch.zeros(1, 8, dtype=torch.long); easy = torch.tensor([[[9., 0, 0, 0]] * 8])
    assert focal_loss(easy, y, gamma=2.0) < ce_loss(easy, y)

def test_weighted_all_ones_equals_ce():
    lg = torch.randn(4, 8, 4); y = torch.randint(0, 4, (4, 8))
    assert torch.isclose(weighted_ce_loss(lg, y, torch.ones(8, 4)), ce_loss(lg, y), atol=1e-6)

def test_default_weights_lower_for_neutral_and_others():
    w = DEFAULT_CLASS_WEIGHTS
    assert w[0, 3] == 0.5 and w[ASPECTS.index("Others"), 0] == 0.5 and w[0, 0] == 1.0

def test_gated_ignores_sentiment_when_not_mentioned():
    y = torch.zeros(2, 8, dtype=torch.long)  # không khía cạnh nào được nhắc đến
    m = torch.randn(2, 8, 2)
    a = gated_loss({"mention": m, "sentiment": torch.randn(2, 8, 3)}, y)
    b = gated_loss({"mention": m, "sentiment": torch.randn(2, 8, 3) * 100}, y)
    assert torch.isclose(a, b)

def test_make_loss_dispatch():
    for t in ["ce", "weighted", "focal"]:
        assert make_loss(t)(torch.randn(2, 8, 4), torch.randint(0, 4, (2, 8))).ndim == 0
    with pytest.raises(ValueError): make_loss("khong-co")
```

- [ ] **Bước 2:** `pytest tests/test_losses.py -q` → FAIL.
- [ ] **Bước 3:** Viết code theo chữ ký; docstring tiếng Việt nêu công thức và ý nghĩa từng loss.
- [ ] **Bước 4:** `pytest tests/test_losses.py -q` → PASS.
- [ ] **Bước 5: Commit** — `"feat(losses): gated, weighted, focal loss"`. Báo B.

### Task 12: NB5 — thí nghiệm — B dựng khung + giai đoạn 1, A chạy giai đoạn 2; ngày 3–5

**Files:**
- Create: `notebooks/05_bert_experiments.ipynb`

**Interfaces:**
- Consumes: `common_cells.py`, `bert_cells.py`, `losses.py`, `udf_pandas.py`
- Produces trong notebook: `run_experiment(pooling: str, loss_type: str, cfg: dict) -> dict` trả `{"pooling", "loss", "val_macro_f1", "test_accuracy", "test_macro_f1", "minutes"}`; `results: pd.DataFrame` gom mọi lần chạy.

- [ ] **Bước 1 (B, ngày 3):** Khung notebook (mục 1–5 giống NB3). Dán `bert_cells.py` và `losses.py`. Viết `run_experiment`: `gated = loss_type == "gated"`; `to_labels = gated_to_labels if gated else argmax_labels`; `fit(..., loss_fn=make_loss(loss_type), to_labels=to_labels)`. Cấu hình thí nghiệm: `resolve_train_config({"epochs": 2, "max_len": 128, "batch_size": 16, "lr": 2e-5, "train_frac": 1.0})`.
- [ ] **Bước 2 (B, ngày 4):** Giai đoạn 1, giữ `loss_type="ce"`, chạy `pooling ∈ ["cls", "mean", "attention"]` (thêm `"max"` nếu kịp). `best_pooling = results.loc[results.val_macro_f1.idxmax(), "pooling"]`. Viết Markdown giải thích attention pooling.
- [ ] **Bước 3 (A, ngày 5, tài khoản Kaggle của A):** Giai đoạn 2, cố định `best_pooling`, chạy `loss_type ∈ ["ce", "gated", "weighted", "focal"]`. Gửi B bảng kết quả và nhận xét để B đưa vào Markdown.
- [ ] **Bước 4 (B, ngày 5):** Bảng tổng hợp `results` + biểu đồ cột macro-F1. Chọn tổ hợp tốt nhất theo `val_macro_f1`, huấn luyện lại, lưu `./models/best`, chạy qua `make_pandas_udf` + `evaluate_with_spark`, rồi viết kết luận.
- [ ] **Bước 5 (B):** Run All toàn bộ NB5 trên một tài khoản → không lỗi, < 2 giờ. Ô kiểm tra cuối: `assert len(results) >= 6` và `assert set(results.loss) >= {"ce", "gated", "weighted", "focal"}`.
- [ ] **Bước 6:** `python tools/check_notebooks.py notebooks/05_*.ipynb` → không lỗi.
- [ ] **Bước 7: Commit** — `"feat(nb): NB5 thí nghiệm pooling và loss"`

### Task 13: Rà chéo, chạy thử ngoài Kaggle, nộp — A + B, ngày 6–7

**Files:**
- Modify: `README.md`
- Create: `submission/*.ipynb`

- [ ] **Bước 1 (ngày 6):** A rà NB3–5, B rà NB1–2, theo checklist: đủ 12 mục; chú thích tiếng Việt; có đủ chuỗi train → lưu → UDF → Spark DataFrame → evaluator; bảng kết quả theo khía cạnh. Mở PR hoặc Issue cho từng góp ý; chủ notebook sửa.
- [ ] **Bước 2 (ngày 6):** A chạy NB1, B chạy NB3 trên **Google Colab** từ file `.ipynb` trong repo (không gắn dataset Kaggle). Kỳ vọng: dữ liệu tự tải từ GitHub raw, chạy hết không lỗi.
- [ ] **Bước 3 (ngày 6):** Kiểm tra `pytest -q` → PASS. Kiểm tra `python tools/check_notebooks.py notebooks/*.ipynb` → không lỗi. Chạy lại "Run All" cả 5 notebook trên Kaggle, đặt notebook Kaggle ở chế độ Public, ghi link vào ô `Hướng dẫn chạy` và README.
- [ ] **Bước 4 (ngày 7):** README: bảng kết quả cuối (Mean accuracy / macro-F1 của 5 notebook), phân công, cách chạy. Chép 5 notebook sang `submission/` với tên `NhomX_danhsachTen_danhsachMSSV_ABSASpark_NB1..5.ipynb` (điền số nhóm, tên, MSSV thật). Kiểm tra repo đang Public.
- [ ] **Bước 5: Commit + nộp** — `"chore: bản nộp cuối"`; nộp 5 file trong `submission/`.
