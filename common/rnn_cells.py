# === COMMON rnn_cells.py v1 ===
# Mô hình họ RNN (BiLSTM) cho ABSA — dán NGUYÊN VĂN file common/rnn_cells.py vào ô này.
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:  # khi chạy pytest
    from common.common_cells import ASPECTS, clean_text
except ImportError:  # trong notebook: đã có sẵn từ ô code chung phía trên
    pass

PAD, UNK = "<pad>", "<unk>"


def tokenize(text: str) -> list:
    """Tách từ đơn giản theo khoảng trắng sau khi làm sạch (mức âm tiết tiếng Việt)."""
    return clean_text(text).split()


def build_vocab(texts, min_freq: int = 2) -> dict:
    """Từ điển token → id. 0 = <pad>, 1 = <unk>; chỉ giữ token xuất hiện >= min_freq lần."""
    counts = Counter(tok for t in texts for tok in tokenize(t))
    vocab = {PAD: 0, UNK: 1}
    for tok, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):  # sắp xếp cố định
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(texts, vocab: dict, max_len: int) -> np.ndarray:
    """Chuyển văn bản → ma trận id [N, max_len] (cắt bớt nếu dài, đệm 0 nếu ngắn)."""
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(tok, 1) for tok in tokenize(t)][:max_len]
        out[i, :len(ids)] = ids
    return out


class RnnAbsa(nn.Module):
    """Embedding → BiLSTM → (mean + max pooling, bỏ qua padding) → 8 đầu phân loại 4 lớp."""

    def __init__(self, vocab_size: int, emb_dim: int = 200, hidden: int = 128,
                 n_aspects: int = 8, n_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.n_aspects, self.n_classes = n_aspects, n_classes
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # 8 đầu phân loại gộp thành một lớp Linear: đầu ra được reshape thành [B, 8, 4]
        self.heads = nn.Linear(4 * hidden, n_aspects * n_classes)

    def forward(self, input_ids):
        mask = (input_ids != 0).unsqueeze(-1).float()            # [B, L, 1]
        h, _ = self.lstm(self.dropout(self.embedding(input_ids)))  # [B, L, 2H]
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        mx = h.masked_fill(mask == 0, -1e4).max(1).values
        mx = torch.where(mask.sum(1) > 0, mx, torch.zeros_like(mx))  # câu rỗng → 0
        feats = self.dropout(torch.cat([mean, mx], dim=-1))
        return self.heads(feats).view(-1, self.n_aspects, self.n_classes)


class _RnnDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.ids = torch.from_numpy(encode(df["text"].tolist(), vocab, max_len))
        self.labels = torch.tensor(df[ASPECTS].to_numpy(dtype=np.int64))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.labels[i]}


def make_rnn_loaders(train_df, val_df, vocab: dict, max_len: int, batch_size: int = 64):
    """DataLoader cho train (xáo trộn) và val; mỗi batch là {"input_ids", "labels"}."""
    return (DataLoader(_RnnDataset(train_df, vocab, max_len), batch_size=batch_size, shuffle=True),
            DataLoader(_RnnDataset(val_df, vocab, max_len), batch_size=batch_size, shuffle=False))


def save_rnn(model, vocab: dict, cfg: dict, out_dir: str) -> None:
    """Lưu trọng số (model.pt), từ điển (vocab.json) và cấu hình (config.json)."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({**cfg, "vocab_size": len(vocab)}, fh, ensure_ascii=False, indent=2)


def load_rnn_predictor(model_dir: str, batch_size: int = 256):
    """Nạp mô hình đã lưu → hàm predictor(list[str]) -> np.ndarray (N, 8) nhãn 0..3."""
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as fh:
        vocab = json.load(fh)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RnnAbsa(cfg["vocab_size"], emb_dim=cfg.get("emb_dim", 200), hidden=cfg.get("hidden", 128),
                    dropout=cfg.get("dropout", 0.3))
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    model.to(device).eval()
    max_len = cfg["max_len"]

    def predictor(texts) -> np.ndarray:
        preds = [np.zeros((0, len(ASPECTS)), dtype=np.int64)]
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                ids = torch.from_numpy(encode(texts[i:i + batch_size], vocab, max_len)).to(device)
                preds.append(model(ids).argmax(-1).cpu().numpy())
        return np.concatenate(preds)

    return predictor
