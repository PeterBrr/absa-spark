import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

from common.bert_cells import (PhoBertAbsa, gated_to_labels, load_bert_predictor, make_bert_loaders,
                               save_bert, segment)
from common.common_cells import ASPECTS


def tiny():
    return RobertaModel(RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, intermediate_size=64, max_position_embeddings=40))


def tiny_tokenizer():
    """Tokenizer tí hon dựng tại chỗ (không tải từ Internet), cùng quy ước <s> … </s> như PhoBERT."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    words = ["giày", "đẹp", "quá", "giao", "hàng", "nhanh", "size", "chuẩn", "giày_đẹp"]
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
                                   pad_token="<pad>", unk_token="<unk>")


@pytest.mark.parametrize("pooling", ["cls", "mean", "max", "attention"])
def test_shapes(pooling):
    m = PhoBertAbsa(tiny(), 32, pooling=pooling)
    ids = torch.randint(5, 100, (2, 10)); mask = torch.ones_like(ids); mask[1, 6:] = 0
    assert m(ids, mask).shape == (2, 8, 4)


@pytest.mark.parametrize("pooling", ["mean", "max", "attention"])
def test_pooling_ignores_padding(pooling):
    torch.manual_seed(0); m = PhoBertAbsa(tiny(), 32, pooling=pooling).eval()
    ids = torch.randint(5, 100, (1, 10)); mask = torch.ones_like(ids); mask[0, 6:] = 0
    ids2 = ids.clone(); ids2[0, 6:] = 7
    assert torch.allclose(m(ids, mask), m(ids2, mask), atol=1e-5)


def test_invalid_pooling_rejected():
    with pytest.raises(ValueError):
        PhoBertAbsa(tiny(), 32, pooling="sum")


def test_gated_outputs_and_labels():
    out = PhoBertAbsa(tiny(), 32, gated=True)(torch.randint(5, 100, (2, 8)), torch.ones(2, 8, dtype=torch.long))
    assert out["mention"].shape == (2, 8, 2) and out["sentiment"].shape == (2, 8, 3)
    fake = {"mention": torch.tensor([[[5., 0.]] * 8]), "sentiment": torch.tensor([[[0., 9., 0.]] * 8])}
    assert gated_to_labels(fake).tolist() == [[0] * 8]
    fake["mention"] = torch.tensor([[[0., 5.]] * 8])
    assert gated_to_labels(fake).tolist() == [[2] * 8]


def test_gated_to_labels_per_aspect():
    mention = torch.tensor([[[0., 1.], [1., 0.], [0., 1.]]])        # nhắc / không / nhắc
    sentiment = torch.tensor([[[3., 0., 0.], [0., 0., 3.], [0., 0., 3.]]])
    assert gated_to_labels({"mention": mention, "sentiment": sentiment}).tolist() == [[1, 0, 3]]


def test_segment_handles_empty():
    out = segment(["", "Giày đẹp quá"])
    assert len(out) == 2 and out[0] == "" and out[1] != ""


def test_segment_handles_none_and_emoji():
    out = segment([None, "😍😍"])
    assert len(out) == 2 and all(isinstance(s, str) for s in out) and out[0] == ""


def test_make_bert_loaders_batches():
    df = pd.DataFrame({"text": ["giày đẹp", "giao hàng nhanh", "", "size chuẩn"],
                       **{a: [0, 1, 2, 3] for a in ASPECTS}})
    tr, va = make_bert_loaders(df, df, tiny_tokenizer(), max_len=4, batch_size=2)
    b = next(iter(va))
    assert set(b) == {"input_ids", "attention_mask", "labels"}
    assert b["input_ids"].shape == b["attention_mask"].shape and b["input_ids"].shape[1] <= 4
    assert b["labels"].shape == (2, 8) and b["labels"].dtype == torch.long
    assert b["labels"][0].tolist() == [0] * 8
    assert sum(len(x["labels"]) for x in tr) == 4


@pytest.mark.parametrize("pooling,gated", [("attention", False), ("cls", True)])
def test_save_load_predictor_roundtrip(tmp_path, monkeypatch, pooling, gated):
    torch.manual_seed(0)
    tok = tiny_tokenizer()
    m = PhoBertAbsa(tiny(), 32, pooling=pooling, gated=gated)
    cfg = {"pooling": pooling, "gated": gated, "max_len": 16, "model_name": "tiny"}
    save_bert(m, tok, cfg, str(tmp_path))
    with open(os.path.join(tmp_path, "config.json"), encoding="utf-8") as fh:
        assert json.load(fh)["pooling"] == pooling
    # nạp lại không được truy cập Internet
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pred = load_bert_predictor(str(tmp_path), batch_size=2)(["giày đẹp quá", "", "😍", "giao hàng nhanh"])
    assert pred.shape == (4, 8) and pred.dtype.kind == "i"
    assert set(np.unique(pred)) <= {0, 1, 2, 3}
    # predictor đã nạp phải cho cùng kết quả với mô hình gốc
    m.eval()
    enc = tok(segment(["giày đẹp quá"]), return_tensors="pt")
    with torch.no_grad():
        out = m(enc["input_ids"], enc["attention_mask"])
    direct = (gated_to_labels(out) if gated else out.argmax(-1)).numpy()
    assert (pred[:1] == direct).all()


def test_paste_bert_blocks_into_one_namespace():
    """NB3–5 dán common_cells → bert_cells → losses → udf_pandas vào cùng một notebook."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ns = {"__name__": "__main__"}
    for name in ["common_cells.py", "bert_cells.py", "losses.py", "udf_pandas.py"]:
        with open(os.path.join(root, "common", name), encoding="utf-8") as fh:
            exec(compile(fh.read(), name, "exec"), ns)
    for fn in ["PhoBertAbsa", "build_phobert_absa", "save_bert", "load_bert_predictor",
               "gated_to_labels", "make_loss", "make_pandas_udf", "fit"]:
        assert callable(ns[fn])
    out = ns["PhoBertAbsa"](tiny(), 32, gated=True)(torch.randint(5, 100, (2, 6)), torch.ones(2, 6, dtype=torch.long))
    assert ns["make_loss"]("gated")(out, torch.randint(0, 4, (2, 8))).ndim == 0
