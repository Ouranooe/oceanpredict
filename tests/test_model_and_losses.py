from __future__ import annotations

import numpy as np
import torch

from ocean_forecast.input_features import (
    augment_input_array,
    augment_input_tensor,
    compute_model_input_channels,
)
from ocean_forecast.losses import (
    masked_front_bce_with_logits_loss,
    masked_density_consistency_loss,
    masked_laplacian_smoothness_loss,
    masked_physics_loss,
    masked_smoothness_loss,
    masked_temporal_diffdiv_loss,
)
from ocean_forecast.metrics import masked_front_iou_from_logits
from ocean_forecast.data.dataset import OceanSeqDataset
from ocean_forecast.data.zip_reader import FrameRef
from ocean_forecast.models.cnn_transformer import CNNTransformerForecaster
from ocean_forecast.models.convlstm import ConvLSTMForecaster
from ocean_forecast.models.predformer import PredFormer
from ocean_forecast.models.registry import build_model
from ocean_forecast.models.tau import TAUForecaster


def test_masked_temporal_diffdiv_loss_constant_and_linear_zero() -> None:
    mask = torch.ones(1, 4, 4)
    pred_const = torch.ones(1, 6, 3, 4, 4)
    loss_const = masked_temporal_diffdiv_loss(pred_const, mask)
    assert torch.isclose(loss_const, torch.tensor(0.0))

    t = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1, 1)
    pred_linear = t.expand(1, 6, 3, 4, 4).clone()
    loss_linear = masked_temporal_diffdiv_loss(pred_linear, mask)
    assert torch.isclose(loss_linear, torch.tensor(0.0))


def test_masked_temporal_diffdiv_loss_speed_weight() -> None:
    mask = torch.ones(1, 3, 3)
    pred = torch.zeros(1, 7, 3, 3, 3)
    t = torch.arange(7, dtype=torch.float32).view(1, 7, 1, 1)
    pred[:, :, 2] = t * t

    loss_off = masked_temporal_diffdiv_loss(pred, mask, channels=[0, 1], speed_weight=0.0)
    loss_on = masked_temporal_diffdiv_loss(pred, mask, channels=[0, 1], speed_weight=0.3)
    assert torch.isclose(loss_off, torch.tensor(0.0))
    assert float(loss_on.item()) > 0.0


def test_masked_laplacian_smoothness_loss_constant_zero() -> None:
    mask = torch.ones(1, 5, 6)
    pred = torch.ones(2, 4, 3, 5, 6)
    loss = masked_laplacian_smoothness_loss(pred, mask)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_masked_laplacian_smoothness_loss_random_finite() -> None:
    mask = torch.ones(1, 5, 6)
    pred = torch.randn(2, 4, 3, 5, 6)
    loss = masked_laplacian_smoothness_loss(pred, mask, channels=[0, 1, 2])
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_masked_density_consistency_loss_constant_zero() -> None:
    pred = torch.zeros(1, 6, 3, 3, 3)
    mask = torch.ones(1, 3, 3)
    target_mean = torch.tensor([10.0, 35.0, 1.0], dtype=torch.float32)
    target_std = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32)

    loss = masked_density_consistency_loss(
        pred=pred,
        mask=mask,
        target_mean=target_mean,
        target_std=target_std,
    )
    assert torch.isclose(loss, torch.tensor(0.0))


def test_masked_density_consistency_loss_linear_positive() -> None:
    pred = torch.zeros(1, 6, 3, 2, 2)
    # Linear trend in normalized SST; SSS stays constant.
    pred[:, :, 0] = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1)
    mask = torch.ones(1, 2, 2)
    target_mean = np.array([10.0, 35.0, 1.0], dtype=np.float32)
    target_std = np.array([2.0, 1.0, 0.5], dtype=np.float32)

    loss = masked_density_consistency_loss(
        pred=pred,
        mask=mask,
        target_mean=target_mean,
        target_std=target_std,
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_masked_density_consistency_loss_denorm_manual_and_mask() -> None:
    pred = torch.zeros(1, 3, 3, 2, 2)
    mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    target_mean = torch.tensor([10.0, 30.0, 0.0], dtype=torch.float32)
    target_std = torch.tensor([2.0, 5.0, 1.0], dtype=torch.float32)

    # Only one valid masked location changes over time in SST normalized space.
    pred[:, 0, 0, 0, 0] = 0.0
    pred[:, 1, 0, 0, 0] = 1.0
    pred[:, 2, 0, 0, 0] = 2.0

    loss = masked_density_consistency_loss(
        pred=pred,
        mask=mask,
        target_mean=target_mean,
        target_std=target_std,
        rho0=1027.0,
        alpha=2.0e-4,
        beta=7.5e-4,
        t0=0.0,
        s0=35.0,
    )

    # Denormalized SST delta per step: 2.0. SSS is constant.
    # drho = rho0 * (-alpha * delta_sst) = 1027 * (-2e-4 * 2.0)
    drho = 1027.0 * (-2.0e-4 * 2.0)
    expected = torch.tensor(drho * drho, dtype=torch.float32)
    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-5)


def test_masked_smoothness_loss_matches_legacy() -> None:
    pred = torch.randn(2, 5, 3, 4, 4)
    mask = torch.ones(2, 4, 4)
    legacy = masked_physics_loss(pred, mask, spatial_weight=1.2, temporal_weight=0.8)
    smooth = masked_smoothness_loss(pred, mask, spatial_weight=1.2, temporal_weight=0.8)
    assert torch.allclose(legacy, smooth)


def test_masked_front_bce_with_logits_loss_finite_and_nonnegative() -> None:
    logits = torch.zeros(2, 3, 1, 4, 5, dtype=torch.float32)
    target = torch.zeros(2, 3, 1, 4, 5, dtype=torch.float32)
    target[:, :, :, :2, :2] = 1.0
    mask = torch.ones(2, 4, 5, dtype=torch.float32)
    loss = masked_front_bce_with_logits_loss(
        logits=logits,
        target_mask=target,
        ocean_mask=mask,
        pos_weight=1.5,
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_masked_front_iou_from_logits_perfect_prediction() -> None:
    target = torch.zeros(1, 2, 1, 3, 3, dtype=torch.float32)
    target[:, :, :, 1:, 1:] = 1.0
    logits = torch.where(target > 0.5, torch.full_like(target, 10.0), torch.full_like(target, -10.0))
    mask = torch.ones(1, 3, 3, dtype=torch.float32)
    iou = masked_front_iou_from_logits(
        logits=logits,
        target_mask=target,
        ocean_mask=mask,
        threshold=0.5,
    )
    assert np.isclose(iou, 1.0)


def test_input_feature_augmentation_tensor_and_array() -> None:
    feature_cfg = {"add_mask": True, "add_time_hour": True, "add_time_year": True}
    x = torch.zeros(2, 3, 4, 4, 5)
    t_in = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
    mask = torch.ones(2, 4, 5)
    aug = augment_input_tensor(x=x, t_in=t_in, mask=mask, feature_cfg=feature_cfg)
    assert aug.shape == (2, 3, 9, 4, 5)
    assert torch.isclose(aug[0, 0, 5, 0, 0], torch.tensor(0.0))  # hour_sin at t=0
    assert torch.isclose(aug[0, 0, 6, 0, 0], torch.tensor(1.0))  # hour_cos at t=0

    x_np = np.zeros((3, 4, 4, 5), dtype=np.float32)
    t_np = np.array([0, 1, 2], dtype=np.int64)
    mask_np = np.ones((4, 5), dtype=np.float32)
    aug_np = augment_input_array(x=x_np, t_in=t_np, mask=mask_np, feature_cfg=feature_cfg)
    assert aug_np.shape == (3, 9, 4, 5)


def test_tau_and_cnn_transformer_output_shape_with_augmented_input() -> None:
    feature_cfg = {"add_mask": True, "add_time_hour": True, "add_time_year": True}
    in_channels = compute_model_input_channels(4, feature_cfg)
    x = torch.randn(2, 5, in_channels, 8, 8)

    tau = TAUForecaster(
        input_channels=in_channels,
        output_channels=3,
        hidden_dim=32,
        dropout=0.0,
        max_pred_len=16,
        default_pred_len=6,
        tau_block_layers=1,
    )
    y_tau = tau(x, pred_len=4)
    assert y_tau.shape == (2, 4, 3, 8, 8)

    model = CNNTransformerForecaster(
        input_channels=in_channels,
        output_channels=3,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.0,
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
    )
    y = model(x, pred_len=4)
    assert y.shape == (2, 4, 3, 8, 8)


def test_dual_head_output_shapes_for_all_models() -> None:
    x = torch.randn(2, 5, 4, 8, 8)

    convlstm = ConvLSTMForecaster(
        input_channels=4,
        output_channels=3,
        hidden_dims=(16, 16),
        default_pred_len=4,
        front_aux_enabled=True,
    )
    tau = TAUForecaster(
        input_channels=4,
        output_channels=3,
        hidden_dim=32,
        max_pred_len=16,
        default_pred_len=6,
        front_aux_enabled=True,
    )
    cnn = CNNTransformerForecaster(
        input_channels=4,
        output_channels=3,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
        front_aux_enabled=True,
    )
    predformer = PredFormer(
        input_channels=4,
        output_channels=3,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        patch_size=4,
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
        front_aux_enabled=True,
    )

    for model in (convlstm, tau, cnn, predformer):
        out = model(x, pred_len=4)
        assert isinstance(out, dict)
        assert out["field"].shape == (2, 4, 3, 8, 8)
        assert out["front_mask_logits"].shape == (2, 4, 1, 8, 8)


class _DummyReaderForFrontMask:
    def __init__(self, frames: np.ndarray, mask: np.ndarray):
        self.frames = frames.astype(np.float32)
        self.mask = mask.astype(bool)

    def read_frame(self, ref: FrameRef):
        return self.frames[int(ref.time_idx)], self.mask


def _make_refs(n: int) -> list[FrameRef]:
    base = np.datetime64("2010-01-01T00:00:00")
    refs: list[FrameRef] = []
    for i in range(n):
        refs.append(
            FrameRef(
                timestamp=base + np.timedelta64(i, "h"),
                hour_index=i,
                zip_path="unused.npy",
                member_name="2010",
                time_idx=i,
            )
        )
    return refs


def test_dataset_front_mask_shape_and_land_zero() -> None:
    t_total, c, h, w = 6, 4, 4, 5
    frames = np.zeros((t_total, c, h, w), dtype=np.float32)
    for t in range(t_total):
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        frames[t, 0] = yy + 0.1 * t
        frames[t, 1] = xx + 0.2 * t
    ocean_mask = np.ones((h, w), dtype=np.float32)
    ocean_mask[:, 0] = 0.0

    stats = {
        "input_mean": np.zeros((4,), dtype=np.float32),
        "input_std": np.ones((4,), dtype=np.float32),
        "target_mean": np.zeros((4,), dtype=np.float32),
        "target_std": np.ones((4,), dtype=np.float32),
        "ocean_mask": ocean_mask,
    }
    ds = OceanSeqDataset(
        reader=_DummyReaderForFrontMask(frames=frames, mask=ocean_mask > 0.5),
        refs=_make_refs(t_total),
        window_starts=[0],
        input_len=3,
        pred_len=2,
        stats=stats,
        front_seg_enabled=True,
        front_seg_quantile=0.9,
    )
    sample = ds[0]
    front = sample["front_mask"].numpy()
    assert front.shape == (2, 1, h, w)
    assert np.all(front[:, :, :, 0] == 0.0)


def test_cnn_transformer_probsparse_forward_shape_and_ratio() -> None:
    x = torch.randn(2, 10, 4, 8, 8)
    model = CNNTransformerForecaster(
        input_channels=4,
        output_channels=3,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.0,
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
        temporal_attention="probsparse",
        probsparse_factor=2.0,
        probsparse_min_k=2,
    )
    y = model(x, pred_len=4)
    assert y.shape == (2, 4, 3, 8, 8)
    assert 0.0 < float(model.last_sparse_query_ratio) < 1.0


def test_cnn_transformer_probsparse_short_sequence_fallback() -> None:
    x = torch.randn(1, 4, 4, 8, 8)
    model = CNNTransformerForecaster(
        input_channels=4,
        output_channels=3,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.0,
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
        temporal_attention="probsparse",
        probsparse_factor=2.0,
        probsparse_min_k=3,
    )
    y = model(x, pred_len=3)
    assert y.shape == (1, 3, 3, 8, 8)
    assert float(model.last_sparse_query_ratio) >= 0.999


def test_cnn_transformer_local_fusion_enabled_shape_and_param_growth() -> None:
    x = torch.randn(2, 6, 4, 8, 8)
    base = CNNTransformerForecaster(
        input_channels=4,
        output_channels=3,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.0,
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
    )
    fused = CNNTransformerForecaster(
        input_channels=4,
        output_channels=3,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.0,
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
        local_fusion_enabled=True,
        local_dilations=[1, 2],
        local_fusion_type="film",
    )
    params_base = sum(p.numel() for p in base.parameters())
    params_fused = sum(p.numel() for p in fused.parameters())
    assert params_fused > params_base

    y = fused(x, pred_len=4)
    assert y.shape == (2, 4, 3, 8, 8)


def test_predformer_forward_shape() -> None:
    feature_cfg = {"add_mask": True, "add_time_hour": True, "add_time_year": True}
    in_channels = compute_model_input_channels(4, feature_cfg)
    x = torch.randn(2, 5, in_channels, 8, 8)

    model = PredFormer(
        input_channels=in_channels,
        output_channels=3,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        patch_size=4,
        temporal_kernel="binary_ts",
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
    )
    y = model(x, pred_len=4)
    assert y.shape == (2, 4, 3, 8, 8)


def test_registry_build_predformer() -> None:
    feature_cfg = {"add_mask": True, "add_time_hour": True, "add_time_year": True}
    in_channels = compute_model_input_channels(4, feature_cfg)
    x = torch.randn(1, 5, in_channels, 8, 8)

    model = build_model(
        model_name="predformer",
        input_channels=in_channels,
        output_channels=3,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        patch_size=4,
        temporal_kernel="binary_ts",
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
    )
    y = model(x, pred_len=3)
    assert y.shape == (1, 3, 3, 8, 8)


def test_predformer_pred_len_override() -> None:
    x = torch.randn(1, 5, 4, 8, 8)
    model = PredFormer(
        input_channels=4,
        output_channels=3,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        patch_size=4,
        temporal_kernel="binary_ts",
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
    )
    y = model(x, pred_len=7)
    assert y.shape == (1, 7, 3, 8, 8)


def test_predformer_non_divisible_patch_supported() -> None:
    x = torch.randn(1, 5, 4, 7, 8)
    model = PredFormer(
        input_channels=4,
        output_channels=3,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        patch_size=4,
        temporal_kernel="binary_ts",
        max_input_len=16,
        max_pred_len=16,
        default_pred_len=6,
    )
    y = model(x, pred_len=4)
    assert y.shape == (1, 4, 3, 7, 8)
