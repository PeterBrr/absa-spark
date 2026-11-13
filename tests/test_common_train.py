import numpy as np
import torch

from common.common_cells import argmax_labels, ce_loss, fit, macro_f1_mean, predict_labels, set_seed


def test_ce_loss_shape_and_value():
    logits = torch.zeros(2, 8, 4)
    labels = torch.zeros(2, 8, dtype=torch.long)
    assert torch.isclose(ce_loss(logits, labels), torch.log(torch.tensor(4.0)))


def test_argmax_labels():
    out = torch.zeros(1, 8, 4)
    out[0, :, 2] = 1.0
    assert argmax_labels(out).tolist() == [[2] * 8]


def test_macro_f1_perfect_and_missing_class():
    y = np.array([[0] * 8, [1] * 8, [2] * 8])
    assert macro_f1_mean(y, y) == 1.0
    score = macro_f1_mean(y, np.zeros_like(y))
    assert 0.0 <= score < 1.0 and not np.isnan(score)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(5, 32)

    def forward(self, x):
        return self.l(x).view(-1, 8, 4)


def _toy_loader():
    set_seed(0)
    X = torch.randn(64, 5)
    Y = (X[:, :1] > 0).long().repeat(1, 8)
    return [{"x": X[i:i + 16], "labels": Y[i:i + 16]} for i in range(0, 64, 16)]


def test_fit_learns_toy_problem():
    loader = _toy_loader()
    hist = fit(Toy(), loader, loader, epochs=30, lr=0.05, device="cpu")
    assert max(hist["val_macro_f1"]) > 0.9
    assert len(hist["train_loss"]) == 30


def test_fit_restores_best_epoch_and_predict_labels():
    loader = _toy_loader()
    model = Toy()
    hist = fit(model, loader, loader, epochs=30, lr=0.05, device="cpu")
    pred = predict_labels(model, loader, device="cpu")
    y = torch.cat([b["labels"] for b in loader]).numpy()
    assert pred.shape == (64, 8)
    assert np.isclose(macro_f1_mean(y, pred), max(hist["val_macro_f1"]))
