# === COMMON bert_cells.py v1 ===
# Mô hình PhoBERT cho ABSA — dán NGUYÊN VĂN file common/bert_cells.py vào ô này.
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:  # khi chạy pytest
    from common.common_cells import ASPECTS, clean_text
except ImportError:  # trong notebook: đã có sẵn từ ô code chung phía trên
    pass

MODEL_NAME = "vinai/phobert-base-v2"
POOLINGS = ("cls", "mean", "max", "attention")


def segment(texts) -> list:
    """Làm sạch (clean_text) rồi tách từ bằng pyvi, vd "Hàng hóa tốt" → "hàng_hóa tốt".
    PhoBERT được huấn luyện trên văn bản đã tách từ nên bước này là bắt buộc. Câu rỗng/None → ""."""
    from pyvi import ViTokenizer  # import trễ: chỉ cần khi thật sự tách từ (pip install pyvi)

    out = []
    for t in texts:
        t = clean_text(t)
        out.append(ViTokenizer.tokenize(t) if t else "")
    return out


class _AspectHeads(nn.Module):
    """8 đầu phân loại độc lập: mỗi khía cạnh một ma trận [H, C] riêng. Vào [B, 8, H] → ra [B, 8, C]."""

    def __init__(self, n_aspects: int, hidden: int, n_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_aspects, hidden, n_classes) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_aspects, n_classes))

    def forward(self, feats):
        return torch.einsum("bah,ahc->bac", feats, self.weight) + self.bias


class PhoBertAbsa(nn.Module):
    """Encoder PhoBERT → pooling → 8 đầu phân loại (mỗi khía cạnh một đầu).

    pooling:
      - "cls": vector của token đầu <s>.
      - "mean" / "max": trung bình / lớn nhất trên các token thật (bỏ qua padding theo attention_mask).
      - "attention": MỖI KHÍA CẠNH MỘT VECTOR QUERY RIÊNG (8 query học được). Với khía cạnh a:
            k_t = tanh(W h_t),  α_{a,t} = softmax_t(q_a · k_t / √H)   (token padding bị loại),
            v_a = Σ_t α_{a,t} h_t
        → mỗi khía cạnh tự "nhìn" vào những từ liên quan tới nó (vd "giao", "ship" cho Shipping)
        thay vì dùng chung một vector câu như cls/mean/max.
    gated:
      - False → logits [B, 8, 4] (0=None, 1=Negative, 2=Positive, 3=Neutral).
      - True  → {"mention": [B, 8, 2] (có nhắc tới khía cạnh hay không),
                 "sentiment": [B, 8, 3] (Negative/Positive/Neutral)} — dùng với gated_loss.
    """

    def __init__(self, encoder: nn.Module, hidden_size: int, pooling: str = "cls", gated: bool = False,
                 n_aspects: int = 8, dropout: float = 0.1):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling không hợp lệ: {pooling!r} (chọn {', '.join(POOLINGS)})")
        self.encoder, self.pooling, self.gated, self.n_aspects = encoder, pooling, gated, n_aspects
        self.dropout = nn.Dropout(dropout)
        if pooling == "attention":
            self.att_key = nn.Linear(hidden_size, hidden_size)
            self.att_query = nn.Parameter(torch.randn(n_aspects, hidden_size) * 0.02)
        if gated:
            self.mention_heads = _AspectHeads(n_aspects, hidden_size, 2)
            self.sentiment_heads = _AspectHeads(n_aspects, hidden_size, 3)
        else:
            self.heads = _AspectHeads(n_aspects, hidden_size, 4)

    def _pool(self, h, attention_mask):
        """h [B, L, H] → đặc trưng theo khía cạnh [B, 8, H]."""
        mask = attention_mask.unsqueeze(-1).to(h.dtype)                      # [B, L, 1]
        if self.pooling == "cls":
            v = h[:, 0]
        elif self.pooling == "mean":
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        elif self.pooling == "max":
            v = h.masked_fill(mask == 0, torch.finfo(h.dtype).min).max(1).values
        else:  # attention: query riêng cho từng khía cạnh
            keys = torch.tanh(self.att_key(h))                                # [B, L, H]
            scores = torch.einsum("blh,ah->bal", keys, self.att_query) / math.sqrt(h.shape[-1])
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, torch.finfo(scores.dtype).min)
            return torch.einsum("bal,blh->bah", scores.softmax(-1), h)        # [B, 8, H]
        return v.unsqueeze(1).expand(-1, self.n_aspects, -1)                  # cùng vector cho 8 khía cạnh

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        feats = self.dropout(self._pool(h, attention_mask))
        if self.gated:
            return {"mention": self.mention_heads(feats), "sentiment": self.sentiment_heads(feats)}
        return self.heads(feats)


def gated_to_labels(out: dict):
    """Đầu ra gated → nhãn [B, 8]: không nhắc tới (mention = 0) → 0 (None); ngược lại sentiment + 1."""
    mentioned = out["mention"].argmax(-1)
    return torch.where(mentioned == 0, torch.zeros_like(mentioned), out["sentiment"].argmax(-1) + 1)


def build_phobert_absa(model_name: str = MODEL_NAME, pooling: str = "cls", gated: bool = False):
    """Tải PhoBERT (cần Internet lần đầu) → (PhoBertAbsa, tokenizer)."""
    from transformers import AutoModel, AutoTokenizer

    encoder = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return PhoBertAbsa(encoder, encoder.config.hidden_size, pooling=pooling, gated=gated), tokenizer


class _BertDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        enc = tokenizer(segment(df["text"].tolist()), truncation=True, max_length=max_len)
        self.input_ids = enc["input_ids"]
        self.labels = torch.tensor(df[ASPECTS].to_numpy(dtype=np.int64))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def _bert_collate(pad_id: int):
    """Đệm động: mỗi batch chỉ đệm tới câu dài nhất trong batch (nhanh hơn đệm cố định max_len)."""
    def collate(items):
        width = max(len(x["input_ids"]) for x in items)
        ids = torch.full((len(items), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(items), width), dtype=torch.long)
        for i, x in enumerate(items):
            ids[i, :len(x["input_ids"])] = torch.tensor(x["input_ids"], dtype=torch.long)
            mask[i, :len(x["input_ids"])] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": torch.stack([x["labels"] for x in items])}
    return collate


def make_bert_loaders(train_df, val_df, tokenizer, max_len: int, batch_size: int = 16):
    """DataLoader cho train (xáo trộn) và val; batch {"input_ids", "attention_mask", "labels"}.
    Văn bản được tách từ (segment) trước khi đưa vào tokenizer."""
    collate = _bert_collate(tokenizer.pad_token_id)
    return (DataLoader(_BertDataset(train_df, tokenizer, max_len), batch_size=batch_size, shuffle=True,
                       collate_fn=collate),
            DataLoader(_BertDataset(val_df, tokenizer, max_len), batch_size=batch_size, shuffle=False,
                       collate_fn=collate))


def save_bert(model, tokenizer, cfg: dict, out_dir: str) -> None:
    """Lưu vào out_dir:
      - model.pt     : trọng số toàn mô hình (encoder + pooling + các đầu)
      - encoder/     : cấu hình kiến trúc encoder (để dựng lại KHÔNG cần tải từ Internet)
      - tokenizer/   : tokenizer
      - config.json  : cfg (pooling, gated, max_len, model_name) + hidden_size"""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))
    meta = {**cfg, "pooling": model.pooling, "gated": model.gated,
            "hidden_size": model.encoder.config.hidden_size}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def load_bert_predictor(model_dir: str, batch_size: int = 32):
    """Nạp mô hình đã lưu (không cần Internet) → predictor(list[str]) -> np.ndarray (N, 8) nhãn 0..3."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join(model_dir, "encoder")))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = PhoBertAbsa(encoder, cfg["hidden_size"], pooling=cfg["pooling"], gated=cfg["gated"])
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    model.to(device).eval()
    to_labels = gated_to_labels if cfg["gated"] else (lambda out: out.argmax(-1))
    max_len = cfg["max_len"]

    def predictor(texts) -> np.ndarray:
        preds = [np.zeros((0, len(ASPECTS)), dtype=np.int64)]
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(segment(texts[i:i + batch_size]), truncation=True, max_length=max_len,
                                padding=True, return_tensors="pt")
                out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
                preds.append(to_labels(out).cpu().numpy())
        return np.concatenate(preds)

    return predictor
