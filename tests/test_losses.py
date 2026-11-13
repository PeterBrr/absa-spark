import pytest
import torch

from common.common_cells import ASPECTS, ce_loss
from common.losses import DEFAULT_CLASS_WEIGHTS, focal_loss, gated_loss, make_loss, weighted_ce_loss


def test_focal_gamma0_equals_ce():
    lg = torch.randn(4, 8, 4)
    y = torch.randint(0, 4, (4, 8))
    assert torch.isclose(focal_loss(lg, y, gamma=0.0), ce_loss(lg, y), atol=1e-6)


def test_focal_downweights_easy_examples():
    y = torch.zeros(1, 8, dtype=torch.long)
    easy = torch.tensor([[[9., 0, 0, 0]] * 8])
    assert focal_loss(easy, y, gamma=2.0) < ce_loss(easy, y)


def test_weighted_all_ones_equals_ce():
    lg = torch.randn(4, 8, 4)
    y = torch.randint(0, 4, (4, 8))
    assert torch.isclose(weighted_ce_loss(lg, y, torch.ones(8, 4)), ce_loss(lg, y), atol=1e-6)


def test_weighted_lowers_contribution_of_low_weight_labels():
    lg = torch.zeros(1, 8, 4)
    lg[..., 0] = 5.0  # đoán đúng mạnh nhãn 0, sai nhãn 3
    y = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 3]])
    w = torch.ones(8, 4)
    w_low = w.clone()
    w_low[7, 3] = 0.1
    assert weighted_ce_loss(lg, y, w_low) < weighted_ce_loss(lg, y, w)


def test_default_weights_lower_for_neutral_and_others():
    w = DEFAULT_CLASS_WEIGHTS
    assert w.shape == (8, 4)
    assert w[0, 3] == 0.5 and w[ASPECTS.index("Others"), 0] == 0.5 and w[0, 0] == 1.0
    assert w[ASPECTS.index("Others"), 3] == 0.25


def test_gated_ignores_sentiment_when_not_mentioned():
    y = torch.zeros(2, 8, dtype=torch.long)  # không khía cạnh nào được nhắc đến
    m = torch.randn(2, 8, 2)
    a = gated_loss({"mention": m, "sentiment": torch.randn(2, 8, 3)}, y)
    b = gated_loss({"mention": m, "sentiment": torch.randn(2, 8, 3) * 100}, y)
    assert torch.isclose(a, b)


def test_gated_mixed_batch_ignores_unmentioned_positions():
    y = torch.tensor([[0, 2, 0, 1, 3, 0, 0, 2]])  # khía cạnh 0, 2, 5, 6 không được nhắc đến
    m = torch.randn(1, 8, 2)
    s1 = torch.randn(1, 8, 3)
    s2 = s1.clone()
    s2[0, [0, 2, 5, 6]] = torch.randn(4, 3) * 100  # chỉ đổi cảm xúc ở vị trí không được nhắc
    assert torch.isclose(gated_loss({"mention": m, "sentiment": s1}, y),
                         gated_loss({"mention": m, "sentiment": s2}, y))


def test_gated_uses_sentiment_when_mentioned():
    y = torch.full((1, 8), 2, dtype=torch.long)  # tất cả Positive
    m = torch.zeros(1, 8, 2)
    good = torch.tensor([[[0., 9., 0.]] * 8])  # sentiment chỉ số 1 = Positive
    bad = torch.tensor([[[9., 0., 0.]] * 8])
    assert gated_loss({"mention": m, "sentiment": good}, y) < gated_loss({"mention": m, "sentiment": bad}, y)


def test_make_loss_dispatch():
    for t in ["ce", "weighted", "focal"]:
        assert make_loss(t)(torch.randn(2, 8, 4), torch.randint(0, 4, (2, 8))).ndim == 0
    out = {"mention": torch.randn(2, 8, 2), "sentiment": torch.randn(2, 8, 3)}
    assert make_loss("gated")(out, torch.randint(0, 4, (2, 8))).ndim == 0
    with pytest.raises(ValueError):
        make_loss("khong-co")
