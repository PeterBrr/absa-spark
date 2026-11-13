# === COMMON losses.py v1 ===
# Các hàm loss cho thí nghiệm NB5 — dán NGUYÊN VĂN file common/losses.py vào ô này.
from functools import partial

import torch
import torch.nn.functional as F

try:  # khi chạy pytest
    from common.common_cells import ASPECTS, ce_loss
except ImportError:  # trong notebook: đã có sẵn từ ô code chung phía trên
    pass


def gated_loss(out: dict, labels):
    """Gated loss — dùng với mô hình gated=True.

    out["mention"]   [B,8,2]: khía cạnh có được nhắc đến không (0 = không, 1 = có)
    out["sentiment"] [B,8,3]: cảm xúc Negative/Positive/Neutral (= nhãn nội bộ 1/2/3 trừ 1)
    loss = CE(mention, labels > 0) + CE(sentiment, labels - 1) — phần cảm xúc CHỈ tính ở các
    khía cạnh được nhắc đến: khía cạnh không được nhắc thì không có loss cảm xúc.
    """
    mention, sentiment = out["mention"], out["sentiment"]
    loss = F.cross_entropy(mention.reshape(-1, 2), (labels > 0).long().reshape(-1))
    mask = labels > 0
    if mask.any():
        loss = loss + F.cross_entropy(sentiment[mask], labels[mask] - 1)
    else:
        loss = loss + sentiment.sum() * 0.0  # giữ đồ thị tính gradient
    return loss


def weighted_ce_loss(logits, labels, class_weights):
    """CrossEntropy có trọng số theo (khía cạnh, nhãn): nhãn kém tin cậy được trọng số thấp hơn.
    class_weights [8,4]; loss = Σ w·CE / Σ w (trọng số toàn 1 thì bằng CE thường)."""
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                         reduction="none").view_as(labels)
    w = class_weights.to(logits.device)
    aspect_idx = torch.arange(labels.shape[1], device=labels.device).expand_as(labels)
    weights = w[aspect_idx, labels]
    return (weights * ce).sum() / weights.sum()


def focal_loss(logits, labels, gamma: float = 2.0):
    """Focal loss: nhân CE với (1 - p_đúng)^gamma → giảm ảnh hưởng mẫu dễ, tăng trọng số mẫu khó.
    gamma = 0 thì bằng CE thường."""
    logp = F.log_softmax(logits, dim=-1)
    logp_true = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (-(1 - logp_true.exp()) ** gamma * logp_true).mean()


# Dataset không có điểm tin cậy cho từng nhãn → coi các nhãn mơ hồ là "kém tin cậy":
#   - nhãn Neutral (chỉ số 3) dễ bị gán nhầm với Positive/Negative → × 0.5
#   - khía cạnh Others (không gắn với sản phẩm, ranh giới không rõ)  → × 0.5
DEFAULT_CLASS_WEIGHTS = torch.ones(len(ASPECTS), 4)
DEFAULT_CLASS_WEIGHTS[:, 3] *= 0.5
DEFAULT_CLASS_WEIGHTS[ASPECTS.index("Others"), :] *= 0.5


def make_loss(loss_type: str, class_weights=None, gamma: float = 2.0):
    """Chọn hàm loss theo tên: "ce", "gated", "weighted", "focal"."""
    if loss_type == "ce":
        return ce_loss
    if loss_type == "gated":
        return gated_loss
    if loss_type == "weighted":
        return partial(weighted_ce_loss,
                       class_weights=DEFAULT_CLASS_WEIGHTS if class_weights is None else class_weights)
    if loss_type == "focal":
        return partial(focal_loss, gamma=gamma)
    raise ValueError(f"loss_type không hợp lệ: {loss_type!r} (chọn ce / gated / weighted / focal)")
