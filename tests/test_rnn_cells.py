import numpy as np
import torch

from common.rnn_cells import RnnAbsa, build_vocab, encode, load_rnn_predictor, make_rnn_loaders, save_rnn
from common.common_cells import ASPECTS


def test_vocab_and_encode():
    v = build_vocab(["giày đẹp", "giày xấu", "giày đẹp"], min_freq=2)
    assert v["<pad>"] == 0 and v["<unk>"] == 1 and "giày" in v and "xấu" not in v
    x = encode(["giày đẹp quá", ""], v, max_len=5)
    assert x.shape == (2, 5) and x.dtype == np.int64
    assert x[1].sum() == 0 and x[0, 2] == 1


def test_encode_truncates():
    v = build_vocab(["a b c d e f"], min_freq=1)
    assert encode(["a b c d e f"], v, max_len=3).shape == (1, 3)


def test_model_shape():
    m = RnnAbsa(vocab_size=50)
    assert m(torch.randint(0, 50, (3, 7))).shape == (3, 8, 4)


def test_model_handles_all_padding_row():
    m = RnnAbsa(vocab_size=50).eval()
    ids = torch.zeros(2, 6, dtype=torch.long)
    ids[0, :3] = 5
    out = m(ids)
    assert out.shape == (2, 8, 4) and torch.isfinite(out).all()
    # câu rỗng → vector đặc trưng bằng 0 → đầu ra chỉ còn bias của lớp phân loại
    assert torch.allclose(out[1], m.heads.bias.view(8, 4))


def test_make_rnn_loaders_batches():
    import pandas as pd
    df = pd.DataFrame({"text": ["giày đẹp", "giao nhanh", "", "size chuẩn"], **{a: [0, 1, 2, 3] for a in ASPECTS}})
    v = build_vocab(df["text"], min_freq=1)
    tr, va = make_rnn_loaders(df, df, v, max_len=4, batch_size=2)
    b = next(iter(tr))
    assert set(b) == {"input_ids", "labels"}
    assert b["input_ids"].shape == (2, 4) and b["labels"].shape == (2, 8)
    assert b["labels"].dtype == torch.long


def test_save_load_predictor_roundtrip(tmp_path):
    v = build_vocab(["a b", "a b"], min_freq=1)
    m = RnnAbsa(len(v))
    save_rnn(m, v, {"max_len": 6, "emb_dim": 200, "hidden": 128}, str(tmp_path))
    pred = load_rnn_predictor(str(tmp_path))(["a b", "", "😍"])
    assert pred.shape == (3, 8) and pred.dtype.kind == "i"
    assert set(np.unique(pred)) <= {0, 1, 2, 3}
    # predictor đã nạp phải cho cùng kết quả với mô hình gốc
    m.eval()
    with torch.no_grad():
        direct = m(torch.from_numpy(encode(["a b"], v, 6))).argmax(-1).numpy()
    assert (pred[:1] == direct).all()
