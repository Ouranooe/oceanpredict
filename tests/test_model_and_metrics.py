from __future__ import annotations

import numpy as np
import torch

from ocean_forecast.losses import masked_mse_loss
from ocean_forecast.metrics import MaskedChannelRMSE, rmse_to_nrmse
from ocean_forecast.models.convlstm import ConvLSTMForecaster


def test_convlstm_output_shape() -> None:
    model = ConvLSTMForecaster(
        input_channels=4,
        output_channels=3,
        hidden_dims=(8, 8),
        kernel_size=3,
        dropout=0.0,
        default_pred_len=6,
    )
    x = torch.randn(2, 5, 4, 16, 12)
    y = model(x)
    assert y.shape == (2, 6, 3, 16, 12)


def test_masked_loss_and_nrmse() -> None:
    pred = torch.zeros(1, 2, 3, 2, 2)
    target = torch.ones(1, 2, 3, 2, 2)
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    loss = masked_mse_loss(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(1.0))

    meter = MaskedChannelRMSE(n_channels=3)
    meter.update(pred, target, mask)
    rmse = meter.compute_rmse()
    assert np.allclose(rmse, np.array([1.0, 1.0, 1.0]))
    nrmse = rmse_to_nrmse(rmse, np.array([2.0, 4.0, 5.0]))
    assert np.allclose(nrmse, np.array([0.5, 0.25, 0.2]))

