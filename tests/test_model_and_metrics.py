from __future__ import annotations

import numpy as np
import torch

from ocean_forecast.losses import masked_mse_loss, masked_physics_loss
from ocean_forecast.metrics import MaskedChannelRMSE, rmse_to_nrmse
from ocean_forecast.models.convlstm import ConvLSTMForecaster
from ocean_forecast.models.registry import build_model


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


def test_other_model_output_shapes() -> None:
    x = torch.randn(2, 5, 4, 16, 12)

    tau = build_model(
        model_name="tau",
        input_channels=4,
        output_channels=3,
        hidden_dim=16,
        dropout=0.0,
        default_pred_len=6,
        max_pred_len=16,
    )
    y_tau = tau(x)
    assert y_tau.shape == (2, 6, 3, 16, 12)

    transformer = build_model(
        model_name="cnn_transformer",
        input_channels=4,
        output_channels=3,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        ff_dim=64,
        dropout=0.0,
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
    )
    y_tf = transformer(x)
    assert y_tf.shape == (2, 6, 3, 16, 12)

    transformer_phys = build_model(
        model_name="cnn_transformer_phys",
        input_channels=4,
        output_channels=3,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        ff_dim=64,
        dropout=0.0,
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
    )
    y_tf_phys = transformer_phys(x)
    assert y_tf_phys.shape == (2, 6, 3, 16, 12)


def test_masked_physics_loss_behavior() -> None:
    mask = torch.ones(1, 4, 4)

    pred_const = torch.zeros(1, 3, 3, 4, 4)
    loss_const = masked_physics_loss(pred_const, mask)
    assert torch.isclose(loss_const, torch.tensor(0.0))

    pred_var = torch.randn(1, 3, 3, 4, 4)
    loss_var = masked_physics_loss(pred_var, mask)
    assert float(loss_var.item()) >= 0.0
